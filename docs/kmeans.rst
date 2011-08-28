KMeans
======

KMeans_ is a geometric iterative algorithm where the objective is to partitition
points into several non-overlapping subsets. Roughly speaking, the idea is to
find mathematically "comfortable positions" for the centroids of these points.

The Math
--------

In more formal terms, the idea is to try to find a local minima for the KMeans
objective function over several iterations. The maximum number of iterations and
the objective function value one is prepared to accept are user driven.
Typically, settings are non-convex which means that we run into local minima
driving poor or drastically varying results. Having said that, the algorithm is
easy to implement.

Objective Function
:::::::::::::::::::

The typical objective function that we try to optimize is the total sum of the
distances between each centroid and each point that "belong" to it.

Distance Measures
::::::::::::::::::


Euclidean Distance
~~~~~~~~~~~~~~~~~~~

The euclidean distance is merely the L2 norm.

Cosine Similarity
~~~~~~~~~~~~~~~~~

This is not a true norm_, the idea is to assume that the vectors are projected
onto a unit hypersphere and to optimize such that the angle between the centroid
and all the vectors belonging to that centroid is minimized.

Cosine Distance
~~~~~~~~~~~~~~~~~

The cosine distance maps the output of cosine similarity to range [0,2].

CODE
----
.. automodule:: kmeans
.. autoclass::  KMeans
   :members:

.. _KMeans: http://en.wikipedia.org/wiki/K-means_clustering
.. _norm: http://en.wikipedia.org/wiki/Norm_%28mathematics%29
