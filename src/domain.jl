

mutable struct Domain
    d       # Dimension of the underlying space
    L       # Number of discretisation points [the same for each marginals, sry ;) !]
    N       # Number of marginals 
    dims    # (L, \dots, L) [a little bit silly]
    xs      # The $(r_1, \dots, r_N) of the grid [it is always one-dimensional, even if d > 1]
    grid    # The grid itself -- a sparse tensor with 0 and 1's
end

# -- Returns the ... xs (I don't know what's the name for that)
@inline _xs(L) = [l/L + 1/(2L) for l = 0:L-1]

# -- Constructor for the Domain
function Domain(L, N, d = 1)
    xs = _xs(L)
    dims = [L^d for i = 1:N]            
    grid = sptzeros(Float64, dims)
    # -- see here, in higher dimension things are flatten in 1D, hence the L^d 
    # -- [my first implementation of the problem was 1D, hence ... the not so intuitive implentation]
    return Domain(d, L, N, dims, xs, grid)
end


# -- Size of the domain Ω, i.e. number of 1s in the grid 
@inline Base.size(D :: Domain) = nnz(D.grid.data)

# -- Remove a point of the domain Ω (make it "inactive" if you prefer)
@inline function remove!(D :: Domain, indices)
    D.grid[indices] = 0
    # -- I add this, but this is not necessary because of the setindex implementation which automatically drops zeros
    dropzeros!(D.grid)
end

# -- Add a point to the domain Ω (make it "active" if you prefer)
@inline function add!(D :: Domain, indices)
    D.grid[indices] = 1
end 

# -- OK, I give (i_1, \dots, i_N), this gives me (x_1, \dots, x_N) (in any dimension d \geq 1)
@inline function to_grid(D :: Domain, indices)
    x = []
    dim_int = [D.L for _ in 1:D.d]
    for i in indices 
        i_ = linear_to_indices(dim_int, i)
        xi = [D.xs[k] for k in i_]
        push!(x, xi)
    end
    return x 
end 

# -- Speaks for itself 
@inline function cost(D :: Domain, c :: Function, indices)
    x = to_grid(D, indices)
    return c(x)
end 

# -- Again, speaks for itselft
function build_cost(D :: Domain, c :: Function)
    idxs = keys(D.grid.data.data)   # -- Fetch linear indices in the sparse vector [maybe should do otherwise]
    C = sptzeros(Float64, D.dims)
    for i in idxs
        pos = linear_to_indices(D.dims, i)
        C[i] = cost(D, c, pos)
    end
    return C
end

# -- I think this is even more stupid [ I mean, could probably do otherwise]
function indices_where_A_zero_but_B_not(A::LargeST{T}, B::LargeST{G}) where {T, G}
    indices = []
    non_zero_indices_B, _ = findnz(B.data)
    for index in non_zero_indices_B
        if A[index] == zero(T)
            push!(indices, index)
        end
    end
    return indices
end
