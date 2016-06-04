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

import numpy

import gensim
from sqlitedict import SqliteDict # needs sqlitedict: run "sudo easy_install sqlitedict"


logger = logging.getLogger('gensim.similarities.simserver')


TOP_SIMS = 100 # when precomputing similarities, only consider this many "most similar" documents
SHARD_SIZE = 65536 # spill index shards to disk in SHARD_SIZE-ed chunks of documents
DEFAULT_NUM_TOPICS = 400 # use this many topics for topic models unless user specified a value
JOURNAL_MODE = 'OFF' # don't keep journals in sqlite dbs

class SimIndex(gensim.utils.SaveLoad):
    """
    An index of documents. Used internally by SimServer.

    It uses the Similarity class to persist all document vectors to disk (via mmap).
    """
    def __init__(self, fname, num_features, shardsize=SHARD_SIZE, topsims=TOP_SIMS):
        """
        Spill index shards to disk after every `shardsize` documents.
        In similarity queries, return only the `topsims` most similar documents.
        """
        self.fname = fname
        self.shardsize = int(shardsize)
        self.topsims = int(topsims)
        self.id2pos = {} # map document id (string) to index position (integer)
        self.pos2id = {} # reverse mapping for id2pos; redundant, for performance
        self.id2sims = SqliteDict(self.fname + '.id2sims', journal_mode=JOURNAL_MODE) # precomputed top similar: document id -> [(doc_id, similarity)]
        self.qindex = gensim.similarities.Similarity(self.fname + '.idx', corpus=None,
            num_best=None, num_features=num_features, shardsize=shardsize)
        self.length = 0

    def save(self, fname):
        tmp, self.id2sims = self.id2sims, None
        super(SimIndex, self).save(fname)
        self.id2sims = tmp


    @staticmethod
    def load(fname):
        result = gensim.utils.SaveLoad.load(fname)
        result.fname = fname
        result.check_moved()
        result.id2sims = SqliteDict(fname + '.id2sims', journal_mode=JOURNAL_MODE)
        return result


    def check_moved(self):
        output_prefix = self.fname + '.idx'
        if self.qindex.output_prefix != output_prefix:
            logger.info("index seems to have moved from %s to %s; updating locations" %
                (self.qindex.output_prefix, output_prefix))
            self.qindex.output_prefix = output_prefix
            self.qindex.check_moved()


    def close(self):
        "Explicitly release important resources (file handles, db, ...)"
        try:
            self.id2sims.close()
        except:
            pass
        try:
            del self.qindex
        except:
            pass


    def terminate(self):
        """Delete all files created by this index, invalidating `self`. Use with care."""
        try:
            self.id2sims.terminate()
        except:
            pass
        import glob
        for fname in glob.glob(self.fname + '*'):
            try:
                os.remove(fname)
                logger.info("deleted %s" % fname)
            except Exception, e:
                logger.warning("failed to delete %s: %s" % (fname, e))
        for val in self.__dict__.keys():
            try:
                delattr(self, val)
            except:
                pass


    def index_documents(self, fresh_docs, model):
        """
        Update fresh index with new documents (potentially replacing old ones with
        the same id). `fresh_docs` is a dictionary-like object (=dict, sqlitedict, shelve etc)
        that maps document_id->document.
        """
        docids = fresh_docs.keys()
        vectors = (model.docs2vecs(fresh_docs[docid] for docid in docids))
        logger.info("adding %i documents to %s" % (len(docids), self))
        self.qindex.add_documents(vectors)
        self.qindex.save()
        self.update_ids(docids)


    def update_ids(self, docids):
        """Update id->pos mapping with new document ids."""
        logger.info("updating %i id mappings" % len(docids))
        for docid in docids:
            if docid is not None:
                pos = self.id2pos.get(docid, None)
                if pos is not None:
                    logger.info("replacing existing document %r in %s" % (docid, self))
                    del self.pos2id[pos]
                self.id2pos[docid] = self.length
                try:
                    del self.id2sims[docid]
                except:
                    pass
            self.length += 1
        self.id2sims.sync()
        self.update_mappings()


    def update_mappings(self):
        """Synchronize id<->position mappings."""
        self.pos2id = dict((v, k) for k, v in self.id2pos.iteritems())
        assert len(self.pos2id) == len(self.id2pos), "duplicate ids or positions detected"


    def delete(self, docids):
        """Delete documents (specified by their ids) from the index."""
        logger.debug("deleting %i documents from %s" % (len(docids), self))
        deleted = 0
        for docid in docids:
            try:
                del self.id2pos[docid]
                deleted += 1
                del self.id2sims[docid]
            except:
                pass
        self.id2sims.sync()
        if deleted:
            logger.info("deleted %i documents from %s" % (deleted, self))
        self.update_mappings()


    def sims2scores(self, sims, eps=1e-7):
        """Convert raw similarity vector to a list of (docid, similarity) results."""
        result = []
        if isinstance(sims, numpy.ndarray):
            sims = abs(sims) # TODO or maybe clip? are opposite vectors "similar" or "dissimilar"?!
            for pos in numpy.argsort(sims)[::-1]:
                if pos in self.pos2id and sims[pos] > eps: # ignore deleted/rewritten documents
                    # convert positions of resulting docs back to ids
                    result.append((self.pos2id[pos], sims[pos]))
                    if len(result) == self.topsims:
                        break
        else:
            for pos, score in sims:
                if pos in self.pos2id and abs(score) > eps: # ignore deleted/rewritten documents
                    # convert positions of resulting docs back to ids
                    result.append((self.pos2id[pos], abs(score)))
                    if len(result) == self.topsims:
                        break
        return result


    def vec_by_id(self, docid):
        """Return indexed vector corresponding to document `docid`."""
        pos = self.id2pos[docid]
        return self.qindex.vector_by_id(pos)


    def sims_by_id(self, docid):
        """Find the most similar documents to the (already indexed) document with `docid`."""
        result = self.id2sims.get(docid, None)
        if result is None:
            self.qindex.num_best = self.topsims
            sims = self.qindex.similarity_by_id(self.id2pos[docid])
            result = self.sims2scores(sims)
        return result


    def sims_by_vec(self, vec, normalize=None):
        """
        Find the most similar documents to a given vector (=already processed document).
        """
        if normalize is None:
            normalize = self.qindex.normalize
        norm, self.qindex.normalize = self.qindex.normalize, normalize # store old value
        self.qindex.num_best = self.topsims
        sims = self.qindex[vec]
        self.qindex.normalize = norm # restore old value of qindex.normalize
        return self.sims2scores(sims)


    def merge(self, other):
        """Merge documents from the other index. Update precomputed similarities
        in the process."""
        other.qindex.normalize, other.qindex.num_best = False, self.topsims
        # update precomputed "most similar" for old documents (in case some of
        # the new docs make it to the top-N for some of the old documents)
        logger.info("updating old precomputed values")
        pos, lenself = 0, len(self.qindex)
        for chunk in self.qindex.iter_chunks():
            for sims in other.qindex[chunk]:
                if pos in self.pos2id:
                    # ignore masked entries (deleted, overwritten documents)
                    docid = self.pos2id[pos]
                    sims = self.sims2scores(sims)
                    self.id2sims[docid] = merge_sims(self.id2sims[docid], sims, self.topsims)
                pos += 1
                if pos % 10000 == 0:
                    logger.info("PROGRESS: updated doc #%i/%i" % (pos, lenself))
        self.id2sims.sync()

        logger.info("merging fresh index into optimized one")
        pos, docids = 0, []
        for chunk in other.qindex.iter_chunks():
            for vec in chunk:
                if pos in other.pos2id: # don't copy deleted documents
                    self.qindex.add_documents([vec])
                    docids.append(other.pos2id[pos])
                pos += 1
        self.qindex.save()
        self.update_ids(docids)

        logger.info("precomputing most similar for the fresh index")
        pos, lenother = 0, len(other.qindex)
        norm, self.qindex.normalize = self.qindex.normalize, False
        topsims, self.qindex.num_best = self.qindex.num_best, self.topsims
        for chunk in other.qindex.iter_chunks():
            for sims in self.qindex[chunk]:
                if pos in other.pos2id:
                    # ignore masked entries (deleted, overwritten documents)
                    docid = other.pos2id[pos]
                    self.id2sims[docid] = self.sims2scores(sims)
                pos += 1
                if pos % 10000 == 0:
                    logger.info("PROGRESS: precomputed doc #%i/%i" % (pos, lenother))
        self.qindex.normalize, self.qindex.num_best = norm, topsims
        self.id2sims.sync()


    def __len__(self):
        return len(self.id2pos)

    def __contains__(self, docid):
        return docid in self.id2pos

    def keys(self):
        return self.id2pos.keys()

    def __str__(self):
        return "SimIndex(%i docs, %i real size)" % (len(self), self.length)
#endclass SimIndex