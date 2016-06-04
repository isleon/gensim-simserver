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
import gensim
from sqlitedict import SqliteDict # needs sqlitedict: run "sudo easy_install sqlitedict"
from simmodel import SimModel
from simindex import SimIndex


logger = logging.getLogger('gensim.similarities.simserver')


TOP_SIMS = 100 # when precomputing similarities, only consider this many "most similar" documents
SHARD_SIZE = 65536 # spill index shards to disk in SHARD_SIZE-ed chunks of documents
DEFAULT_NUM_TOPICS = 400 # use this many topics for topic models unless user specified a value
JOURNAL_MODE = 'OFF' # don't keep journals in sqlite dbs


def merge_sims(oldsims, newsims, clip=None):
    """Merge two precomputed similarity lists, truncating the result to `clip` most similar items."""
    if oldsims is None:
        result = newsims or []
    elif newsims is None:
        result = oldsims
    else:
        result = sorted(oldsims + newsims, key=lambda item: -item[1])
    if clip is not None:
        result = result[:clip]
    return result


class SimServer(object):
    """
    Top-level functionality for similarity services. A similarity server takes
    care of::

    1. creating semantic models
    2. indexing documents using these models
    3. finding the most similar documents in an index.

    An object of this class can be shared across network via Pyro, to answer remote
    client requests. It is thread safe. Using a server concurrently from multiple
    processes is safe for reading = answering similarity queries. Modifying
    (training/indexing) is realized via locking = serialized internally.
    """
    def __init__(self, basename, use_locks=False):
        """
        All data will be stored under directory `basename`. If there is a server
        there already, it will be loaded (resumed).

        The server object is stateless in RAM -- its state is defined entirely by its location.
        There is therefore no need to store the server object.
        """
        if not os.path.isdir(basename):
            raise ValueError("%r must be a writable directory" % basename)
        self.basename = basename
        self.use_locks = use_locks
        self.lock_update = threading.RLock() if use_locks else gensim.utils.nocm
        try:
            self.fresh_index = SimIndex.load(self.location('index_fresh'))
        except:
            logger.debug("starting a new fresh index")
            self.fresh_index = None
        try:
            self.opt_index = SimIndex.load(self.location('index_opt'))
        except:
            logger.debug("starting a new optimized index")
            self.opt_index = None
        try:
            self.model = SimModel.load(self.location('model'))
        except:
            self.model = None
        self.payload = SqliteDict(self.location('payload'), autocommit=True, journal_mode=JOURNAL_MODE)
        self.flush(save_index=False, save_model=False, clear_buffer=True)
        logger.info("loaded %s" % self)

    def location(self, name):
        return os.path.join(self.basename, name)

    @gensim.utils.synchronous('lock_update')
    def flush(self, save_index=False, save_model=False, clear_buffer=False):
        """Commit all changes, clear all caches."""
        if save_index:
            if self.fresh_index is not None:
                self.fresh_index.save(self.location('index_fresh'))
            if self.opt_index is not None:
                self.opt_index.save(self.location('index_opt'))
        if save_model:
            if self.model is not None:
                self.model.save(self.location('model'))
        self.payload.commit()
        if clear_buffer:
            if hasattr(self, 'fresh_docs'):
                try:
                    self.fresh_docs.terminate() # erase all buffered documents + file on disk
                except:
                    pass
            self.fresh_docs = SqliteDict(journal_mode=JOURNAL_MODE) # buffer defaults to a random location in temp
        self.fresh_docs.sync()

    def close(self):
        """Explicitly close open file handles, databases etc."""
        try:
            self.payload.close()
        except:
            pass
        try:
            self.model.close()
        except:
            pass
        try:
            self.fresh_index.close()
        except:
            pass
        try:
            self.opt_index.close()
        except:
            pass
        try:
            self.fresh_docs.terminate()
        except:
            pass

    def __del__(self):
        """When the server went out of scope, make an effort to close its DBs."""
        self.close()

    @gensim.utils.synchronous('lock_update')
    def buffer(self, documents):
        """
        Add a sequence of documents to be processed (indexed or trained on).

        Here, the documents are simply collected; real processing is done later,
        during the `self.index` or `self.train` calls.

        `buffer` can be called repeatedly; the result is the same as if it was
        called once, with a concatenation of all the partial document batches.
        The point is to save memory when sending large corpora over network: the
        entire `documents` must be serialized into RAM. See `utils.upload_chunked()`.

        A call to `flush()` clears this documents-to-be-processed buffer (`flush`
        is also implicitly called when you call `index()` and `train()`).
        """
        logger.info("adding documents to temporary buffer of %s" % (self))
        for doc in documents:
            docid = doc['id']
#            logger.debug("buffering document %r" % docid)
            if docid in self.fresh_docs:
                logger.warning("asked to re-add id %r; rewriting old value" % docid)
            self.fresh_docs[docid] = doc
        self.fresh_docs.sync()

    @gensim.utils.synchronous('lock_update')
    def train(self, corpus=None, method='auto', clear_buffer=True, params=None):
        """
        Create an indexing model. Will overwrite the model if it already exists.
        All indexes become invalid, because documents in them use a now-obsolete
        representation.

        The model is trained on documents previously entered via `buffer`,
        or directly on `corpus`, if specified.
        """
        if corpus is not None:
            # use the supplied corpus only (erase existing buffer, if any)
            self.flush(clear_buffer=True)
            self.buffer(corpus)
        if not self.fresh_docs:
            msg = "train called but no training corpus specified for %s" % self
            logger.error(msg)
            raise ValueError(msg)
        if method == 'auto':
            numdocs = len(self.fresh_docs)
            if numdocs < 1000:
                logging.warning("too few training documents; using simple log-entropy model instead of latent semantic indexing")
                method = 'logentropy'
            else:
                method = 'lsi'
        if params is None:
            params = {}
        self.model = SimModel(self.fresh_docs, method=method, params=params)
        self.flush(save_model=True, clear_buffer=clear_buffer)

    @gensim.utils.synchronous('lock_update')
    def index(self, corpus=None, clear_buffer=True):
        """
        Permanently index all documents previously added via `buffer`, or
        directly index documents from `corpus`, if specified.

        The indexing model must already exist (see `train`) before this function
        is called.
        """
        if not self.model:
            msg = 'must initialize model for %s before indexing documents' % self.basename
            logger.error(msg)
            raise AttributeError(msg)

        if corpus is not None:
            # use the supplied corpus only (erase existing buffer, if any)
            self.flush(clear_buffer=True)
            self.buffer(corpus)

        if not self.fresh_docs:
            msg = "index called but no indexing corpus specified for %s" % self
            logger.error(msg)
            raise ValueError(msg)

        if not self.fresh_index:
            logger.info("starting a new fresh index for %s" % self)
            self.fresh_index = SimIndex(self.location('index_fresh'), self.model.num_features)
        self.fresh_index.index_documents(self.fresh_docs, self.model)
        if self.opt_index is not None:
            self.opt_index.delete(self.fresh_docs.keys())
        logger.info("storing document payloads")
        for docid in self.fresh_docs:
            payload = self.fresh_docs[docid].get('payload', None)
            if payload is None:
                # HACK: exit on first doc without a payload (=assume all docs have payload, or none does)
                break
            self.payload[docid] = payload
        self.flush(save_index=True, clear_buffer=clear_buffer)

    @gensim.utils.synchronous('lock_update')
    def optimize(self):
        """
        Precompute top similarities for all indexed documents. This speeds up
        `find_similar` queries by id (but not queries by fulltext).

        Internally, documents are moved from a fresh index (=no precomputed similarities)
        to an optimized index (precomputed similarities). Similarity queries always
        query both indexes, so this split is transparent to clients.

        If you add documents later via `index`, they go to the fresh index again.
        To precompute top similarities for these new documents too, simply call
        `optimize` again.

        """
        if self.fresh_index is None:
            logger.warning("optimize called but there are no new documents")
            return # nothing to do!

        if self.opt_index is None:
            logger.info("starting a new optimized index for %s" % self)
            self.opt_index = SimIndex(self.location('index_opt'), self.model.num_features)

        self.opt_index.merge(self.fresh_index)
        self.fresh_index.terminate() # delete old files
        self.fresh_index = None
        self.flush(save_index=True)

    @gensim.utils.synchronous('lock_update')
    def drop_index(self, keep_model=True):
        """Drop all indexed documents. If `keep_model` is False, also dropped the model."""
        modelstr = "" if keep_model else "and model "
        logger.info("deleting similarity index " + modelstr + "from %s" % self.basename)

        # delete indexes
        for index in [self.fresh_index, self.opt_index]:
            if index is not None:
                index.terminate()
        self.fresh_index, self.opt_index = None, None

        # delete payload
        if self.payload is not None:
            self.payload.close()

            fname = self.location('payload')
            try:
                if os.path.exists(fname):
                    os.remove(fname)
                    logger.info("deleted %s" % fname)
            except Exception, e:
                logger.warning("failed to delete %s" % fname)
        self.payload = SqliteDict(self.location('payload'), autocommit=True, journal_mode=JOURNAL_MODE)

        # optionally, delete the model as well
        if not keep_model and self.model is not None:
            self.model.close()
            fname = self.location('model')
            try:
                if os.path.exists(fname):
                    os.remove(fname)
                    logger.info("deleted %s" % fname)
            except Exception, e:
                logger.warning("failed to delete %s" % fname)
            self.model = None
        self.flush(save_index=True, save_model=True, clear_buffer=True)

    @gensim.utils.synchronous('lock_update')
    def delete(self, docids):
        """Delete specified documents from the index."""
        logger.info("asked to drop %i documents" % len(docids))
        for index in [self.opt_index, self.fresh_index]:
            if index is not None:
                index.delete(docids)
        self.flush(save_index=True)

    def is_locked(self):
        return self.use_locks and self.lock_update._RLock__count > 0

    def vec_by_id(self, docid):
        for index in [self.opt_index, self.fresh_index]:
            if index is not None and docid in index:
                return index.vec_by_id(docid)


    def find_similar(self, doc, min_score=0.0, max_results=100):
        """
        Find `max_results` most similar articles in the index, each having similarity
        score of at least `min_score`. The resulting list may be shorter than `max_results`,
        in case there are not enough matching documents.

        `doc` is either a string (=document id, previously indexed) or a
        dict containing a 'tokens' key. These tokens are processed to produce a
        vector, which is then used as a query against the index.

        The similar documents are returned in decreasing similarity order, as
        `(doc_id, similarity_score, doc_payload)` 3-tuples. The payload returned
        is identical to what was supplied for this document during indexing.

        """
        logger.debug("received query call with %r" % doc)
        if self.is_locked():
            msg = "cannot query while the server is being updated"
            logger.error(msg)
            raise RuntimeError(msg)
        sims_opt, sims_fresh = None, None
        for index in [self.fresh_index, self.opt_index]:
            if index is not None:
                index.topsims = max_results
        if isinstance(doc, basestring):
            # query by direct document id
            docid = doc
            if self.opt_index is not None and docid in self.opt_index:
                sims_opt = self.opt_index.sims_by_id(docid)
                if self.fresh_index is not None:
                    vec = self.opt_index.vec_by_id(docid)
                    sims_fresh = self.fresh_index.sims_by_vec(vec, normalize=False)
            elif self.fresh_index is not None and docid in self.fresh_index:
                sims_fresh = self.fresh_index.sims_by_id(docid)
                if self.opt_index is not None:
                    vec = self.fresh_index.vec_by_id(docid)
                    sims_opt = self.opt_index.sims_by_vec(vec, normalize=False)
            else:
                raise ValueError("document %r not in index" % docid)
        else:
            if 'topics' in doc:
                # user supplied vector directly => use that
                vec = gensim.matutils.any2sparse(doc['topics'])
            else:
                # query by an arbitrary text (=tokens) inside doc['tokens']
                vec = self.model.doc2vec(doc) # convert document (text) to vector
            if self.opt_index is not None:
                sims_opt = self.opt_index.sims_by_vec(vec)
            if self.fresh_index is not None:
                sims_fresh = self.fresh_index.sims_by_vec(vec)

        merged = merge_sims(sims_opt, sims_fresh)
        logger.debug("got %s raw similars, pruning with max_results=%s, min_score=%s" %
            (len(merged), max_results, min_score))
        result = []
        for docid, score in merged:
            if score < min_score or 0 < max_results <= len(result):
                break
            result.append((docid, float(score), self.payload.get(docid, None)))
        return result


    def __str__(self):
        return ("SimServer(loc=%r, fresh=%s, opt=%s, model=%s, buffer=%s)" %
                (self.basename, self.fresh_index, self.opt_index, self.model, self.fresh_docs))


    def __len__(self):
        return sum(len(index) for index in [self.opt_index, self.fresh_index]
                   if index is not None)


    def __contains__(self, docid):
        """Is document with `docid` in the index?"""
        return any(index is not None and docid in index
                   for index in [self.opt_index, self.fresh_index])

    def get_tfidf(self, *args, **kwargs):
        return self.model.get_tfidf(*args, **kwargs)

    def status(self):
        return str(self)

    def keys(self):
        """Return ids of all indexed documents."""
        result = []
        if self.fresh_index is not None:
            result += self.fresh_index.keys()
        if self.opt_index is not None:
            result += self.opt_index.keys()
        return result

    def memdebug(self):
        from guppy import hpy
        return str(hpy().heap())
#endclass SimServer
