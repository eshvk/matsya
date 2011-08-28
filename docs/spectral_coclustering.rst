Spectral Co-Clustering
======================

Spectral Co-Clustering is an algorithm that simultaneously extracts groups of
features and data points that contain those features. For example, one would be
able to extract groups of words and documents from a corpus. Other examples
where co-clustering would be useful is in Biology.

The Math
--------

Spectral Algorithms
:::::::::::::::::::

A spectral algorithm is one where one attempts to use the information contained
within singular vectors of some normalized form of the adjacency matrix of a
graph in order to extract clusters from a Graph. More information can be found
here_.

Spectral Co-Clustering
:::::::::::::::::::::::

Spectral clustering algorithms typically require that one has an adjacency
matrix of a graph. This natural extension of this assumption in the field of
text mining would be to construct a graph of documents. Unfortunately, this is
an O(n^2) operation unless one were to do something clever like LSH_. The
spectral co-clustering algorithm views the term-document matrix as the adjacency
matrix of a bipartite graph: One where the set of all nodes contains all
features and all documents.


CODE
----
.. automodule:: cocluster
.. autoclass::  SpectralCoClusterer
   :members:

.. _here: http://en.wikipedia.org/wiki/Spectral_graph_theory
.. _LSH: http://en.wikipedia.org/wiki/Locality-sensitive_hashing
