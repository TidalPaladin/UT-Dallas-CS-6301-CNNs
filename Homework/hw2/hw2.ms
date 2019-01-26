.TL
CS6301.005 Homework 02
.AU
Scott Chase Waggener
scw180000

.SH
Problem 1
.LP
.PP
When multiplying MxK and KxN matrices we will produce a MxN output matrix
where each element was the result of a linear combination of K elements.
As such, we must compute M*N sums of K elements.
Similarly, for memory operations we must bring in the original matrices and output
the resulting matrix, leading to a sum over the dimensions of these matrices.

.EQ
O sub {compute} (M, N, K) = M*N*K
.EN
.EQ
O sub {memory} (M, N, K) = M*K + K*N + M*N
.EN

The proof that arithmetic intensity is as follows: s

