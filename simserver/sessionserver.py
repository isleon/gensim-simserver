#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU AGPL v3 - http://www.gnu.org/licenses/agpl.html


"""
"Find similar" service, using gensim (=vector spaces) for backend.

The server performs 3 main functions:

1. converts documents to semantic representation (TF-IDF, LSA, LDA...)
2. indexes documents in the vector representation, for faster retrieval
3. for a given query document, return ids of the most similar documents from the index

SessionServer objects are transactional, so that you can rollback/commit an entire
set of changes.

The server is ready for concurrent requests (thread-safe). Indexing is incremental
and you can query the SessionServer even while it's being updated, so that there
is virtually no down-time.

"""

from __future__ import with_statement

import os
import logging
import threading
import shutil
import gensim
from simserver import SimServer

logger = logging.getLogger('gensim.similarities.simserver')


TOP_SIMS = 100 # when precomputing similarities, only consider this many "most similar" documents
SHARD_SIZE = 65536 # spill index shards to disk in SHARD_SIZE-ed chunks of documents
DEFAULT_NUM_TOPICS = 400 # use this many topics for topic models unless user specified a value
JOURNAL_MODE = 'OFF' # don't keep journals in sqlite dbs


class SessionServer(gensim.utils.SaveLoad):
    """
    Similarity server on top of :class:`SimServer` that implements sessions = transactions.

    A transaction is a set of server modifications (index/delete/train calls) that
    may be either committed or rolled back entirely.

    Sessions are realized by:

    1. cloning (=copying) a SimServer at the beginning of a session
    2. serving read-only queries from the original server (the clone may be modified during queries)
    3. modifications affect only the clone
    4. at commit, the clone becomes the original
    5. at rollback, do nothing (clone is discarded, next transaction starts from the original again)
    """
    def __init__(self, basedir, autosession=True, use_locks=True):
        self.basedir = basedir
        self.autosession = autosession
        self.use_locks = use_locks
        self.lock_update = threading.RLock() if use_locks else gensim.utils.nocm
        self.locs = ['a', 'b'] # directories under which to store stable.session data
        try:
            stable = open(self.location('stable')).read().strip()
            self.istable = self.locs.index(stable)
        except:
            self.istable = 0
            logger.info("stable index pointer not found or invalid; starting from %s" %
                        self.loc_stable)
        try:
            os.makedirs(self.loc_stable)
        except:
            pass
        self.write_istable()
        self.stable = SimServer(self.loc_stable, use_locks=self.use_locks)
        self.session = None

    def location(self, name):
        return os.path.join(self.basedir, name)

    @property
    def loc_stable(self):
        return self.location(self.locs[self.istable])

    @property
    def loc_session(self):
        return self.location(self.locs[1 - self.istable])

    def __contains__(self, docid):
        return docid in self.stable

    def __str__(self):
        return "SessionServer(\n\tstable=%s\n\tsession=%s\n)" % (self.stable, self.session)

    def __len__(self):
        return len(self.stable)

    def keys(self):
        return self.stable.keys()

    @gensim.utils.synchronous('lock_update')
    def check_session(self):
        """
        Make sure a session is open.

        If it's not and autosession is turned on, create a new session automatically.
        If it's not and autosession is off, raise an exception.
        """
        if self.session is None:
            if self.autosession:
                self.open_session()
            else:
                msg = "must open a session before modifying %s" % self
                raise RuntimeError(msg)

    @gensim.utils.synchronous('lock_update')
    def open_session(self):
        """
        Open a new session to modify this server.

        You can either call this fnc directly, or turn on autosession which will
        open/commit sessions for you transparently.
        """
        if self.session is not None:
            msg = "session already open; commit it or rollback before opening another one in %s" % self
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info("opening a new session")
        logger.info("removing %s" % self.loc_session)
        try:
            shutil.rmtree(self.loc_session)
        except:
            logger.info("failed to delete %s" % self.loc_session)
        logger.info("cloning server from %s to %s" %
                    (self.loc_stable, self.loc_session))
        shutil.copytree(self.loc_stable, self.loc_session)
        self.session = SimServer(self.loc_session, use_locks=self.use_locks)
        self.lock_update.acquire() # no other thread can call any modification methods until commit/rollback

    @gensim.utils.synchronous('lock_update')
    def buffer(self, *args, **kwargs):
        """Buffer documents, in the current session"""
        self.check_session()
        result = self.session.buffer(*args, **kwargs)
        return result

    @gensim.utils.synchronous('lock_update')
    def index(self, *args, **kwargs):
        """Index documents, in the current session"""
        self.check_session()
        result = self.session.index(*args, **kwargs)
        if self.autosession:
            self.commit()
        return result

    @gensim.utils.synchronous('lock_update')
    def train(self, *args, **kwargs):
        """Update semantic model, in the current session."""
        self.check_session()
        result = self.session.train(*args, **kwargs)
        if self.autosession:
            self.commit()
        return result

    @gensim.utils.synchronous('lock_update')
    def drop_index(self, keep_model=True):
        """Drop all indexed documents from the session. Optionally, drop model too."""
        self.check_session()
        result = self.session.drop_index(keep_model)
        if self.autosession:
            self.commit()
        return result

    @gensim.utils.synchronous('lock_update')
    def delete(self, docids):
        """Delete documents from the current session."""
        self.check_session()
        result = self.session.delete(docids)
        if self.autosession:
            self.commit()
        return result

    @gensim.utils.synchronous('lock_update')
    def optimize(self):
        """Optimize index for faster by-document-id queries."""
        self.check_session()
        result = self.session.optimize()
        if self.autosession:
            self.commit()
        return result

    @gensim.utils.synchronous('lock_update')
    def write_istable(self):
        with open(self.location('stable'), 'w') as fout:
            fout.write(os.path.basename(self.loc_stable))

    @gensim.utils.synchronous('lock_update')
    def commit(self):
        """Commit changes made by the latest session."""
        if self.session is not None:
            logger.info("committing transaction in %s" % self)
            tmp = self.stable
            self.stable, self.session = self.session, None
            self.istable = 1 - self.istable
            self.write_istable()
            tmp.close() # don't wait for gc, release resources manually
            self.lock_update.release()
        else:
            logger.warning("commit called but there's no open session in %s" % self)

    @gensim.utils.synchronous('lock_update')
    def rollback(self):
        """Ignore all changes made in the latest session (terminate the session)."""
        if self.session is not None:
            logger.info("rolling back transaction in %s" % self)
            self.session.close()
            self.session = None
            self.lock_update.release()
        else:
            logger.warning("rollback called but there's no open session in %s" % self)

    @gensim.utils.synchronous('lock_update')
    def set_autosession(self, value=None):
        """
        Turn autosession (automatic committing after each modification call) on/off.
        If value is None, only query the current value (don't change anything).
        """
        if value is not None:
            self.rollback()
            self.autosession = value
        return self.autosession

    @gensim.utils.synchronous('lock_update')
    def close(self):
        """Don't wait for gc, try to release important resources manually"""
        try:
            self.stable.close()
        except:
            pass
        try:
            self.session.close()
        except:
            pass

    def __del__(self):
        self.close()

    @gensim.utils.synchronous('lock_update')
    def terminate(self):
        """Delete all files created by this server, invalidating `self`. Use with care."""
        logger.info("deleting entire server %s" % self)
        self.close()
        try:
            shutil.rmtree(self.basedir)
            logger.info("deleted server under %s" % self.basedir)
            # delete everything from self, so that using this object fails results
            # in an error as quickly as possible
            for val in self.__dict__.keys():
                try:
                    delattr(self, val)
                except:
                    pass
        except Exception, e:
            logger.warning("failed to delete SessionServer: %s" % (e))

    def find_similar(self, *args, **kwargs):
        """
        Find similar articles.

        With autosession off, use the index state *before* current session started,
        so that changes made in the session will not be visible here. With autosession
        on, close the current session first (so that session changes *are* committed
        and visible).
        """
        if self.session is not None and self.autosession:
            # with autosession on, commit the pending transaction first
            self.commit()
        return self.stable.find_similar(*args, **kwargs)

    def get_tfidf(self, *args, **kwargs):
        if self.session is not None and self.autosession:
            # with autosession on, commit the pending transaction first
            self.commit()
        return self.stable.get_tfidf(*args, **kwargs)

    # add some functions for remote access (RPC via Pyro)
    def debug_model(self):
        return self.stable.model

    def status(self): # str() alias
        return str(self)
