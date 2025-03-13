include("../src/OTSolver.jl")

using LinearAlgebra
L = 10
N = 2
d = 2
function init()
    # Random.seed!(1245)
    D = OTSolver.Domain(L, N, d)
    dims = D.dims 

    # Populate with the diagonal 
    for i = 1:L^d
        ii = [i for _ = 1:N]
        idx = OTSolver.indices_to_linear(dims, ii)
        D.grid[idx] = 1.
    end 

    n = N*L^d
    for i = 1:n
     ii = Tuple(rand(1:L) for _ in 1:N)
        idx = OTSolver.indices_to_linear(dims, ii)
        D.grid[idx] = 1.
    end 

    # Coulomb cost 
    ϵ = 0.1 # Cutoff
    coulomb(x, y) = 1/max(norm(x - y), ϵ)

    function cost(xs)
        l = length(xs)
        res = 0
        for i = 1:l-1
            for j = i+1:l
                res += coulomb(xs[i], xs[j])
            end
        end
        return res 
    end 

    # Building the initial cost matrix 
    C = OTSolver.build_cost(D, cost)

    # Building the MOT problem 
    Π = OTSolver.sptzeros(Float64, D.dims)

    for i = 1:L
        ii = Tuple(i for _ in 1:N)
        idx = OTSolver.indices_to_linear(dims, ii)
        Π[idx] = 1.
    end 
    # Marginals = uniform
    m = [ones(L^d)/L^d for i = 1:N]
    # Marginals = beta 
    #f(x) = x*(1-x)
    #m = [f.(D.xs) for i = 1:N]
    # Kantorovich potential
    ψs = zeros(N, L^d)


    return OTSolver.Problem(C, m, Π, ψs, D, cost);
end

P = init()

wniter = 1000
P = init()
@time vals_gero = OTSolver.solve(P, niter, 2)