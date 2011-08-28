#/usr/bin/env python
# Filename: cocluster.py
# Author: Eshwaran Vijaya Kumar
"""
An implementation of the Spectral Co-Clustering algorithm, the module defines
the following class:

- `SpectralCoClusterer`, the Spectral CoClustering driver class.


How To Use This Module
----------------------

1. Invoke as a script passing it a Term-Document matrix.

2. `import matsya.cocluster` and initialize `SpectralCoClusterer`.

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
import scipy.linalg as spla

try:
    from .. import kmeans
except ValueError:
    import kmeans


class SpectralCoClusterer:

    """
    Generate cluster of words and documents.

    Perform Spectral Co-clustering on input term-document matrix. Note
    that it works well in practice for TF input.

    Parameters:

    - `A` : Term - Doc adjacency matrix.
    - `k` : Number of clusters to generate. ( Integer )
    - `n` : Number of iterations before stopping. ( Integer )
    - `delta` : Convergence parameter. ( Recommended : [0.01, 0.9]
    - `rc` : True generates centroids by splitting matrix
            deterministically, False uses random groups of columns.
    - `cl` : True performs classical KMeans, while false performs
                spherical KMeans.
    - `v` : Enables debug setting.

    References:

        1. Dhillon, I. (2001). Co-clustering documents and words using
        bipartite spectral graph partitioning. In Proceedings of the seventh
        ACM SIGKDD international conference on Knowledge discovery and data
        mining (KDD) (pp.269 - 274). New York: ACM Press.
    """

    def __init__(self, A, k, n, delta, rc, cl, v):
        """
            Initialize the `SpectralCoClusterer` class.

            Parameters:

            - `A` : Term - Doc adjacency matrix.
            - `k` : Number of clusters to generate. ( Integer )
            - `n` : Number of iterations before stopping. ( Integer )
            - `delta` : Convergence parameter. ( Recommended : [0.01, 0.9]
            - `rc` : True generates centroids by splitting matrix
                    deterministically, False uses random groups of columns.
            - `cl` : True performs classical KMeans, while false performs
                     spherical KMeans.
            - `v` : Enables debug setting.

        """
        self.A = A
        self.k = k
        self.n = n
        self.delta = delta
        self.rc = rc
        self.cl = cl
        self.verbose = v
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        self.logger.debug("Starting Spectral Co-Clustering debugging...")

    def run(self):
        """ Main method that drives Spectral Co-Clustering. """
        self.nfeatures = self.A.shape[0]
        self.ndocs = self.A.shape[1]
        self.logger.debug("Word By Documentmatrix A has dim:(%d,%d)", \
                          self.nfeatures, self.ndocs)
        self.logger.debug("Generating normalized Adjacency Matrix, A_n")
        self.gen_An()
        self.logger.debug("Finding SVD of An")
        un, s, vnt = spla.svd(self.An.todense())
        self.logger.debug('Shape of un (%d,%d)', un.shape[0], un.shape[1])
        vn = vnt.T
        self.logger.debug('Shape of vn (%d,%d)', vn.shape[1], vn.shape[1])
        self.logger.debug("Generating Z matrix")
        self.get_Z(un, vn)
        data = (self.Z.T).tocsc()
        kmeans = kmeans.KMeans(data, self.k, self.n, self.delta, self.rc, \
                               self.cl, self.verbose)
        result = kmeans.run()
        self.centroids = result['centroids']
        self.centroid_dict = result['centroiddict']
        self.clusters = result['clusters']
        self.cluster_dict = self._get_cluster_dict()
        self.logger.debug('Number of co-clusters produced: %d', \
                                                        len(self.clusters))
        return {'centroids' : self.centroids, \
                'centroiddict' : self.centroid_dict, \
                'clusters' : self.clusters, \
                'clusterdict' : self.cluster_dict}

    def gen_An(self):
        """ Generates normalized Adjacency Matrix An = D1 * A * D2. """
        self._get_inv_sqrt_D1()
        self.logger.debug('D1 dimensions are (%d,%d)', self.D1.shape[0], \
                          self.D1.shape[1])
        self._get_inv_sqrt_D2()
        self.logger.debug('D2 dimensions are (%d,%d)', self.D2.shape[0], \
                          self.D2.shape[1])
        self.An = self.D1 * self.A * self.D2
        self.logger.debug('An dimensions are (%d,%d)', self.An.shape[0], \
                          self.An.shape[1])

    def _get_cluster_dict(self):
        """ Construct dict mapping vector IDs to docs, features.

            Returns:

                `cluster_dict` : A dict mapping vector IDS to a tuple where
                first element is id of doc/feature and second element is `f` or
                `d`.
        """
        cluster_dict = {}
        for ii in range(self.ndocs + self.nfeatures):
            if self._is_feature(ii):
                cluster_dict[ii] = (self._get_fid(ii), 'f')
            else:
                cluster_dict[ii] = (self._get_did(ii), 'd')
        return cluster_dict

    def _get_inv_sqrt_D1(self):
        """ Compute term degree matrix and take its inverse square root. """
        numwords = self.A.shape[0]
        d = np.zeros(numwords)
        II = np.arange(0, numwords, 1)
        JJ = II
        for ii in range(numwords):
            temp = math.sqrt(self.A[ii, :].todense().sum())
            d[ii] = 1 / temp
        self.D1 = ssp.coo_matrix((d, (II, JJ)), \
                                 shape=(numwords, numwords)).tocsc()

    def _get_inv_sqrt_D2(self):
        """ Compute doc degree matrix and take its inverse square root. """
        numdocs = self.A.shape[1]
        d = np.zeros(numdocs)
        II = np.arange(0, numdocs, 1)
        JJ = II
        for ii in range(numdocs):
            temp = math.sqrt(self.A[:, ii].todense().sum())
            d[ii] = 1 / temp
        self.D2 = ssp.coo_matrix((d, (II, JJ)), \
                                 shape=(numdocs, numdocs)).tocsc()

    def get_Z(self, un, vn):
        """ Get matrix Z.

            Parameters:

            - `un` : Left singular vectors of An.
            - `vn` : Right singular vectors of An.
        """
        self.l = int(math.ceil(math.log(self.k, 2)))
        self._prune_un(un)
        self._prune_vn(vn)
        self._get_Z_features()
        self._get_Z_docs()
        self.Z = ssp.vstack([self.Zfeatures, self.Zdocs])

    def _prune_un(self, un):
        """ Select `l` singular vectors from `un`.

            Parameters:

            - `un` : Left singular vectors of An.
        """
        self.U = ssp.csc_matrix(un[:, 1:(self.l + 1)])

    def _prune_vn(self, vn):
        """ Select `l` singular vectors from `vn`.

            Parameters:

            - `vn` : Right singular vectors of An.
        """
        self.V = ssp.csc_matrix(vn[:, 1:(self.l + 1)])

    def _get_Z_features(self):
        """ Compute the features part of Z."""
        self.Zfeatures = self.D1 * self.U

    def _get_Z_docs(self):
        """ Compute the documents part of Z."""
        self.Zdocs = self.D2 * self.V

    def _is_feature(self, z):
        """ Checks if `z` is feature or doc.

            Parameters:

            - `z` : The integer to check.

            Returns:

            - : Boolean
        """
        if z < self.nfeatures:
            return True
        else:
            return False

    def _get_fid(self, z):
        """ Get the feature ID corresponding to the feature.

            Parameters:

            - `z` : The integer to convert.

            Returns:

            - `z` : Corresponding feature id.
        """
        assert (z < self.nfeatures), 'Run _is_feature prior to _get_fid'
        return z

    def _get_did(self, z):
        """ Get the doc ID corresponding to the doc.

            Parameters:

            - `z` : The integer to convert.

            Returns:

            -  : Corresponding doc id.
        """
        assert (z >= self.nfeatures), 'Run iszfeaure prior to _get_did'
        return z - self.nfeatures


def gen_args():
    """
        A formatted argument generator.

        Returns:
        _ `parser` : An argparse object.
    """
    parser = argparse.ArgumentParser(description='Spectral Co-Clusterer')
    parser.add_argument('-A', type=file, help='CSC Term-Doc Matrix in MMF')
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
        Main entry point to script to perform spectral co-clustering.

        Returns:

        - `0` or `1` on success or failure respectively.
        - Saves `centroids`, `centroiddict`, `clusters` and `clusterdict` in \
                working dir.

    """
    parser = gen_args()
    args = parser.parse_args()
    sessionid = args.sessionid
    A = spio.mmread(args.A).tocsc()
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if args.k:
        k = args.k
    spcc = SpectralCoClusterer(A, k, args.n, args.delta, \
                               args.randomcentroids, \
                               args.classical, args.verbose)
    result = spcc.run()
    clusters = result['clusters']
    centroids = result['centroids']
    centroid_dict = result['centroiddict']
    cluster_dict = result['clusterdict']
    cPickle.dump(clusters, open("clusters_" + sessionid + '.pck', 'w'))
    cPickle.dump(centroid_dict, open("centroid_dict_" + \
                                    sessionid + '.pck', 'w'))
    cPickle.dump(cluster_dict, open("cluster_dict_" + \
                                    sessionid + '.pck', 'w'))
    spio.mmwrite(open("centroids_" + sessionid + '.mtx', 'w'), \
                 centroids, comment="CSC Matrix", field='real')
    logger.info(" %d Clusters Generated ", len(clusters))
    return 0

if __name__ == "__main__":
    sys.exit(main())
