# Improving the Wfchef Pattern Detection and Replication Algorithms
Our goal is to re-implement the Wfchef workflow pattern detection and replication algorithms [1]  (part of the larger [Wfcommons](https://github.com/wfcommons/wfcommons) framework) as divide and conquer algorithms so that the tool can be used to detect patterns and generate realistic synthetic workflows for larger applications.

## Team Members
- [Jared Coleman](https://jaredraycoleman.com)
- [Tainã Coleman](https://tainacoleman.com)

## Preliminary Results
This work presents an algorithm that attempts to find all patterns in a Direct Acyclic Graph (DAG) simultaneously in one pass. We start with a DAG that represents an application, or a scientific workflow, and rather than flooding on every pair of nodes that have the same typehash (see definition in [1]) as it is done in wfChef, our goal is to find all the patterns comparing the nodes only once in the top-down pass through the graph to eliminate the redundant computation, and divide the work to be computed in parallel for speed-up purposes. 

Figure 1 shows our preliminary results of the speedup possible with this kind of one-pass algorithm for 7 different applications: 1000genome, Cycles, Epigenomics, Montage, Seismology, Soykb and Srasearch.

![Figure 1: Preliminary results for possible speedup between WfChef approach (old) and our new approach (new).](https://user-images.githubusercontent.com/38535400/207549262-dc547603-b441-4bf6-be8a-3045e6cd0089.jpg)

We have not yet succeeded in finding an algorithm of this form. Our most recent efforts find all patterns, but also find many false positives. The next step is to perform a speed analysis for single-CPU and parallel computing set-ups.     


## References
[1] Coleman, T., Casanova, H., & da Silva, R. F. (2021, September). WfChef: Automated generation of accurate scientific workflow generators. In 2021 IEEE 17th International Conference on eScience (eScience) (pp. 159-168). IEEE.
