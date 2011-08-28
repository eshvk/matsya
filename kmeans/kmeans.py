#/usr/bin/env python
# Filename: kmeans.py
# Author: Eshwaran Vijaya Kumar
"""
An implementation of the KMeans algorithm, the module defines the following
class:

- `KMeans`, the KMeans driver class.


How To Use This Module
----------------------

1. Invoke as a script passing it a CSC matrix where the columns are vectors.

2. `import matsya.kmeans` and initialize `KMeans`.

"""

__docformat__ = 'restructuredtext'

import math
import logging
import sys
import time

import argparse
import cPickle
import numpy as np
import scipy.sparse as ssp
import scipy.io as spio


class KMeans:

    """
    Class that performs batch KMeans.

    Note that the vectors should ideally be normalized tf-idf vectors for
    Spherical KMeans.

    Parameters:

    - `data` : CSC Matrix with rows as features and columns as points.
    - `k` : Number of clusters to generate. ( Integer )
    - `n` : Number of iterations before stopping. ( Integer )
    - `delta` : Convergence parameter. ( Recommended : [0.01, 0.9]
    - `rc` : True generates centroids by splitting matrix
            deterministically, False uses random groups of columns.
    - `v` : Enables debug setting.
    - `cl` : True performs classical KMeans, while false performs
                spherical KMeans.


    References:

        1. I. S. Dhillon and D. S. Modha. Concept decompositions for large
        sparse text data using clustering. Machine Learning, 42:143-175, 2001.

    """

    def __init__(self, data, k, n, delta, rc, cl, v):

        """
            Initialize the `KMeans` class.

            Parameters:

            - `data`: CSC Matrix with rows as features and columns as points.
            - `k`: Number of clusters to generate.
            - `n`: Number of iterations before stopping.
            - `delta`: Convergence parameter.
            - `rc`: True generates centroids by splitting matrix
                    deterministically, False uses random groups of columns.
            - `v`: Enables debug setting.
            - `cl`: True performs classical KMeans, while false performs
                spherical KMeans.

        """
        self.data = data
        self.k = k
        self.n = n
        self.delta = delta
        self.rc = rc
        verbose = v
        self.cl = cl
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        if verbose:
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug("Starting KMeans debugging...")

    def _chunks(self, l, n):
        """Chunks up list `l` into divisions of `n`.

            Parameters:

            - `l` : A list.
            - `n` : The splits required.

            Returns:

            - A list of lists.

        """
        return [l[i:i + n] for i in range(0, len(l), n)]

    def converged(self, newQ):
        """
            Check convergence.

            Check whether difference between `currQ`, `newQ` less than delta.

            Parameters:

            - `newQ`: A new quality measure.

            Returns:

            - `True`or `False`

        """
        if math.fabs(self.Q - newQ) < self.delta:
            return True
        else:
            return False

    def get_centroids(self, clusters):
        """ Uses `clusters` to generate `centroids`.

            Returns:

             -   dict containing: `centroids`, `centroiddict`.

        """
        if self.cl is True:
            normalize = False
        else:
            normalize = True
        k = len(clusters)
        newcentroids = np.mat(np.zeros((self.data.shape[0], k)))
        invnorms = np.zeros(k)
        normsII = np.arange(0, k, 1)
        normsJJ = normsII
        centroiddict = {}
        ii = 0
        for centroid, v_ids in clusters.iteritems():
            for v in v_ids:
                newcentroids[:, ii] = newcentroids[:, ii] +\
                        self.data[:, v].todense()
            newcentroids[:, ii] = newcentroids[:, ii] * (float(1) / len(v_ids))
            normcentroid = math.sqrt(newcentroids[:, ii].T *\
                                     newcentroids[:, ii])
            if normcentroid is not 0:
                invnorms[ii] =\
                    1 / (math.sqrt(newcentroids[:, ii].T *\
                                   newcentroids[:, ii]))
            else:
                invnorms[ii] = 0
            assert(centroid not in centroiddict),\
                    'Logic in get_centroids is wrong'
            centroiddict[centroid] = ii
            ii = ii + 1
        if normalize is True:
            diag = ssp.coo_matrix((invnorms, (normsII, normsJJ)),\
                                                    shape=(k, k)).tocsc()
            return {'centroids': ssp.csc_matrix(newcentroids) * diag, \
                    'centroiddict': centroiddict}
        else:
            return {'centroids': ssp.csc_matrix(newcentroids), \
                    'centroiddict': centroiddict}

    def get_closest(self, distances):
        """ Find closest point from `distances`.

            Parameters:

            - `distances`: Dict of points.

            Returns:

            - Closest point key.

        """
        sortedkeys = distances.keys()
        sortedkeys.sort(cmp=lambda a, b: cmp(distances[a], distances[b]))
        MIN = 0
        MAX = len(sortedkeys) - 1
        if self.cl is True:
            return sortedkeys[MIN]
        else:
            return sortedkeys[MAX]

    def get_deterministic_partitions(self):
        """ Divide up the vectors into `k` partitions deterministically. """
        nvectors = self.data.shape[1]
        numsplits = int(math.floor(nvectors / self.k))
        v_ids = range(0, nvectors, 1)
        v_idslist = self._chunks(v_ids, numsplits)
        ii = 0
        self.clusters = {}
        while ii < self.k:
            self.clusters[ii] = v_idslist[ii]
            ii = ii + 1
        if self.cl is True:
            result = self.get_centroids(self.clusters)
        else:
            result = self.get_centroids(self.clusters)
        self.centroids = result['centroids']
        self.centroiddict = result['centroiddict']

    def get_dist(self, a, b):
        """ Returns distance between two points.

            Parameters:

            - `a`: A vector in CSC form.
            - `b`: A vector in CSC form.

            Returns:

            - : Distance between the two points.

        """
        if self.cl is True:
            return math.sqrt(((a - b).T * (a - b)).todense())
        else:
            return (a.T * b).todense()

    def get_Q(self, centroids, centroiddict, clusters):
        """ Computes objective function that measures quality of clusters.

            Parameters:

            - `centroids`: A sparse csc matrix.
            - `centroiddict`: A map from centroid ID to matrix column number.
            - `clusters`: A map from centroid ID to list of vectors.

            Returns:

            - `Q`: A measure of clustering.
        """
        Q = 0
        for c_id, v_ids in clusters.iteritems():
            for v in v_ids:
                cv_id = centroiddict[c_id]
                Q += self.get_dist(self.data[:, v], centroids[:, cv_id])
        return Q

    #TODO: (Eshwaran) Generate global mean vector and generate centroids by
    # random perturbations of this vector and then compute clusters
    def get_randomized_partitions(self):
        """ Divide up the vectors among the k partitions randomly. """
        nvectors = self.data.shape[1]
        numsplits = int(math.floor(nvectors / self.k))
        v_ids = range(0, nvectors, 1)
        np.random.shuffle(v_ids)
        v_idslist = self._chunks(v_ids, numsplits)
        ii = 0
        self.clusters = {}
        while ii < self.k:
            self.clusters[ii] = v_idslist[ii]
            ii = ii + 1
        if self.cl is True:
            result = self.get_centroids(self.clusters)
        else:
            result = self.get_centroids(self.clusters)
        self.centroids = result['centroids']
        self.centroiddict = result['centroiddict']

    def run(self):
        """ Runs kmeans.

            Returns:

            - `results` : dict containing `clusters`, `centroids`,
                `centroiddict`.

        """
        assert (self.data.shape[1] > self.k), "Number of clusters requested\
        greater than number of vectors"
        self.logger.debug("Data is of dimensions:\
                     (%d, %d)", self.data.shape[0], self.data.shape[1])
        self.logger.debug("Generating %d clusters ...", self.k)
        if self.rc:
            self.logger.debug("Generating centroids by randomized partioning")
            self.get_randomized_partitions()
        else:
            self.logger.debug("Generating centroids by arbitrary partitioning")
            self.get_deterministic_partitions()
        self.Q = self.get_Q(self.centroids, self.centroiddict, self.clusters)
        ii = 0
        new_clusters = {}
        while ii < self.n:
            self.logger.debug("Iteration %d", ii)
            newclusters = {}
            jj = 0
            while jj < self.data.shape[1]:
                actualk = len(self.clusters)
                if self.k is not actualk:
                    self.logger.debug("Number of clusters is %d and not k=%d",
                                      actualk, self.k)
                dcentroids = {}
                for cid, cv_id in self.centroiddict.iteritems():
                    dcentroids[cid] = self.get_dist(self.data[:, jj], \
                                                   self.centroids[:, cv_id])
                closestcluster = self.get_closest(dcentroids)
                if closestcluster in newclusters:
                    newclusters[closestcluster].append(jj)
                else:
                    newclusters[closestcluster] = [jj]
                jj = jj + 1
            self.logger.debug("Going to get new centroids...")
            if self.cl is True:
                result = self.get_centroids(newclusters)
            else:
                result = self.get_centroids(newclusters)
            newcentroids = result['centroids']
            newcentroiddict = result['centroiddict']
            self.logger.debug("Going to check convergence...")
            newQ = self.get_Q(newcentroids, newcentroiddict, newclusters)
            if self.converged(newQ):
                break
            else:
                self.centroids = newcentroids
                self.centroiddict = newcentroiddict
                self.clusters = newclusters
                self.Q = newQ
            ii = ii + 1

        return {'clusters': self.clusters, 'centroiddict': \
                    self.centroiddict, 'centroids': self.centroids}


def gen_args():
    """
        A formatted argument generator.

        Returns:
        _ `parser`: An argparse object.
    """
    parser = argparse.ArgumentParser(description='KMeans Clusterer')
    parser.add_argument('-data', type=file, help='CSC Matrix in MMF')
    parser.add_argument('-v', action='store_true', default=False, \
                        dest='verbose', help='Verbose output. Default = No')
    parser.add_argument('-classical', action='store_true', default=False, \
                        dest='classical', help='Select type of\
                        KMeans to use Spherical or Euclidean. Default:\
                        Spherical')
    parser.add_argument('-k', metavar='k', action='store', type=int, \
                        dest='k', default=None, help='Number of clusters to\
                        generate. No input leads to finding k.')
    parser.add_argument('-n', metavar='n', action='store', type=int,\
                        dest='n', help='Max number of iterations')
    parser.add_argument('-delta', metavar='delta', action='store', \
                        default=0.005, type=float, dest='delta', help='Quit\
                        see if difference in objective function is less than\
                        delta. Default=0.005')
    parser.add_argument('-rc', action='store_true', default=False, \
                        dest='randomcentroids', help='Generate centroids by\
                        partitioning matrix deterministically or randomize\
                        selection of columns. Default=false')
    parser.add_argument('-sessionid', action='store', dest='sessionid', \
                        default=str(int(time.time() * 100000)), \
                        help='Generate unique session id. Default=time\
                        dependent')
    return parser


def main():
    """
        Main entry point to script to perform kmeans.

        Returns:

        - `0` or `1` on success or failure respectively.
        - Saves `centroids`, `centroiddict`, and `clusters` in working dir.

    """
    parser = gen_args()
    args = parser.parse_args()
    sessionid = args.sessionid
    data = spio.mmread(args.data).tocsc()
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if args.k:
        k = args.k
    kmeans = KMeans(data, k, args.n, args.delta, args.randomcentroids, \
                    args.classical, args.verbose)
    result = kmeans.run()
    clusters = result['clusters']
    centroids = result['centroids']
    centroiddict = result['centroiddict']
    cPickle.dump(clusters, open("data_clusters_" + sessionid + '.pck', 'w'))
    cPickle.dump(centroiddict, open("centroid_dict_" + \
                                    sessionid + '.pck', 'w'))
    spio.mmwrite(open("data_centroids_" + sessionid + '.mtx', 'w'), \
                 centroids, comment="CSC Matrix", field='real')
    logger.info(" %d Clusters Generated ", len(clusters))
    return 0

if __name__ == "__main__":
    sys.exit(main())
