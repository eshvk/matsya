========
matsya
========

matsya is a collection of unsupervised text mining routines writted in numpy 
initially to cluster data for [Firefox Input](http://input.mozilla.com). I 
have generalized these routines to work in any general context. As far as I 
have seen, I have not found a co-clustering routine implementing the specific 
algorithm written in here, so this module has *some* novelty.

Although in the interests of portability, I have packaged matsya and made it
available to install via pip, some of the routines that depend on NLTK will not
work out of the box. So once you install matsya, fire up python and install the
following NLTK data files. 


For more details on how to install NLTK data files, go
[here](http://www.nltk.org/data).

Documentation is put up [here](http://matsya.eshvk.me)
