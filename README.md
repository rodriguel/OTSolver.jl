# OTSolver
Implementation of the GenCol algorithm introduced in 

> Friesecke, G., Schulz, A. S., & Vögler, D. (2022). Genetic column generation: fast computation of high-dimensional multimarginal optimal transport problems. SIAM Journal on Scientific Computing, 44(3), A1632-A165

Given $N$ the number of marginals, $d \geq 1$ their dimension, $L$ the size of the discretisation of each marginals (beware that $L$ is the discretisation along each dimension of $\mathbb{R}^d$, so that the number of points in the discretisation is actually given by $L^d$), we build a sparse domain ``D = Domain(N, L, d)`` which is modeled as a sparse tensor, contained in the field ``D.grid``.


## To do
- This code was origally written to deal with (an arbitrary number of) one-dimensional marginals. Whie it does allow these marginals to be of any dimension $d \geq 1$, I think the implementation is certainly not optimal.
