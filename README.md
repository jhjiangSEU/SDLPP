# SDLPP
## How to start?
Just run demo.m.


## Data sets
Because the data sets are too large, we only provide a synthetic data set slashdot with r=1,2,3 and a real-world data set lost.
- Real-world data sets are publicly available at: http://palm.seu.edu.cn/zhangml/Resources.htm#data.
- Synthetic data sets are derived from multi-label benchmark data sets by retaining examples with only one relevant label.
   Specically, given a multi-class example (x_i, y_i), its corresponding partial label training example (xi, W_i) is generated by 
   randomly adding r class labels from Y into Si. You can get these multi-label data sets at: http://mulan.sourceforge.net/datasets-mlc.html,
   http://waikato.github.io/meka/datasets/, and https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/.


## Declare
EuDist2.m is from [1] and getProperDim.m is from [2].

- [1] Xiao-Fei He and Partha Niyogi. 2003. Locality preserving projections. In Advances in Neural Information Processing Systems 16. MIT Press, Cambridge, MA, 153–160.
- [2] Wei-Xuan Bao, Jun-Yi Hang, and Min-Ling Zhang. 2021. Partial label dimensionality reduction via confidence-based dependence maximization. In Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. Virtual Event, 46–54.
