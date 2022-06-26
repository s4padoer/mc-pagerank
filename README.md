# mc-pagerank
Monte-Carlo PageRank Implementation

The code allows to update random walks on a directed graph partially, depending on successive changes in the underlying graph. 
New or deleted nodes only require a subset of the random walk to be updated - this subset is identified, deleted and updated. Based on the random walks, an
estimate of the underlying stationary distribution can be calculated.
