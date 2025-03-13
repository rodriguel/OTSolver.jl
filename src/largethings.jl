
include("utils.jl")

# -- If tensors are big, linear indices may too large for Int type
myInt = Union{Int, BigInt}

# -- Structure that implements sparse vectors
struct LargeSV{T}
    n::myInt                # Length of the sparse vector
    data::Dict{myInt, T}    # Dictionnary to store the (non-zeros) values
end


# -- Structure that implements sparse tensors
struct LargeST{T}
    N :: Int                # Number of dimensions
    dims::Vector{<: myInt}  # Dimensions 
    data::LargeSV{T}        # Sparse vector (flattened tensor in linear indices)
end


# -- Constructor that yields an empty sparse vector of length n
function LargeSV{T}(n) where T
    data = Dict{myInt, T}()
    return LargeSV(n, data)
end 

# -- Constructure that yields an empty sparse vector with dims provided
function LargeST{T}(dims::Vector{<: myInt}) where T
    N = length(dims)
    data = LargeSV{T}(prod(dims))
    LargeST(N, dims, data)
end

# -- Exactly the same as the constructor, mainly for convenience
function spzeros(T, n)
    return LargeSV{T}(n)
end 

# -- Idem for sparse tensor, mainly for convenience
function sptzeros(T, dims)
    LargeST{T}(dims)
end 

# -- Retrieve the length of the sparse vector
function Base.length(v :: LargeSV{T}) where T
    return v.n
end

# -- getindex implementation 
function Base.getindex(v::LargeSV{T}, i) where T
    @assert 1 <= i <= v.n "Index out of bounds"
    return get(v.data, i, zero(T))
end

# -- getindex for sparse tensor, with support for slices ":" [Un peu codé avec les pieds...]
function Base.getindex(A::LargeST{T}, indices...) where {T}
    N = A.N

    # -- if single index, then it's linear indexing
    if length(indices) == 1
        @assert 1 ≤ length(indices) ≤ prod(A.dims) "Out of index"
        idx = indices[1]
        return A.data[idx]
    else 
        # -- ok, it's not a single index, so  we need to work
        @assert length(indices) == N "Number of indices must match tensor dimensions"
    
        # -- is it a subtensor ? 
        sub_tensor = any(isa.(indices, Colon))


        if sub_tensor
            # -- oh no ! It is a subtensor, so we need to work my_random_neighborhood [Chatgpt-ed and corrected by me, I was lazy]
            nd = myInt[]
            for i in 1:A.N
                if isa(indices[i], Colon)
                    push!(nd, A.dims[i])
                end 
            end 

            new_dims = map((dim, idx) -> isa(idx, Colon) ? dim : 1, A.dims, indices)
            new_data = spzeros(T, prod(new_dims))
            
            for (linear_index, value) in A.data.data
                old_indices = linear_to_indices(A.dims, linear_index)
                match = true
                new_indices = []
                for (i, (idx, old_idx)) in enumerate(zip(indices, old_indices))
                    if isa(idx, myInt)
                        if idx != old_idx
                            match = false
                            break
                        else
                            push!(new_indices, 1)
                        end
                    elseif isa(idx, Colon)
                        push!(new_indices, old_idx)
                    end
                end
                
                if match
                    new_linear_index = indices_to_linear(new_dims, new_indices)
                    new_data[new_linear_index] = value
                end
            end
            
            return LargeST(length(nd), nd, new_data)
        else
            # Ok, it is not a subtensor, so we are happy !
            idx = indices_to_linear(A.dims, indices)
            return A[idx]
        end
    end 
end

# -- setindex implementation 
function Base.setindex!(v::LargeSV{T}, val, i) where T
    @assert 1 <= i <= v.n "Index out of bounds"
    if val == zero(T)
        delete!(v.data, i)
    else 
        v.data[i] = val
    end 
end

function Base.setindex!(A::LargeST{T}, value, indices) where T
    if length(indices) == 1
        @assert 1 ≤ length(indices) ≤ A.data.n "Out of index"
        idx = indices[1]
        A.data[idx] = value
    else 
        @assert length(indices) == A.N "Number of indices must match tensor dimensions"
        idx = indices_to_linear(A.dims, indices)
        A.data[idx] = value
    end
end


# -- Number of non-zeros entry in the sparse vector
@inline nnz(v :: LargeSV{T}) where T = length(v.data)
@inline nnz(A :: LargeST{T}) where T = nnz(A.data)
# -- Idem for tensors, but it's called "size" (oups !)
@inline Base.size(A::LargeST{T}) where T = nnz(A.data)
# -- Retrieve keys and vals of non-zero entries 
function findnz(v :: LargeSV{T}) where T
    return keys(v.data), values(v.data)
end 
# -- Idem for sparse tensors 
function findnz(A :: LargeST{T}) where T
    return findnz(A.data)
end
# -- Retrieve the support of a tensor
function support(A :: LargeST{T}) where T
    B = sptzeros(T, A.dims)
    ks, _ = findnz(A)
    for idx in ks 
        B[idx] = 1.
    end 
    return B
end 


# -- Remove all zeros entries [shouldn't be used, implemented in setindex automatically]
function dropzeros!(v::LargeSV{T}) where T
    for (k, val) in collect(v.data)
        if val ==  zero(T)
            delete!(v.data, k)
        end
    end
    return v
end
# Overloading for sparse tensors
@inline dropzeros!(A :: LargeST{T}) where T = dropzeros!(A.data)



# -- Multiplication with a scalar
function mult(lhs, rhs::LargeSV{T}) where T
    new_data = Dict{myInt, T}()
    for (key, val) in rhs.data
        new_data[key] = lhs * val
    end
    return LargeSV{T}(rhs.n, new_data)
end

# -- In place multiplication with a scalar
function mult!(lhs, rhs::LargeSV{T}) where T
    for (key, val) in rhs.data
        rhs.data[key] = lhs * val
    end 
end 

# -- Element-wise multiplication between two sparse vectors
function mult(lhs::LargeSV{T}, rhs::LargeSV{G}) where {T, G}
    @assert lhs.n == rhs.n "Sparse vectors do not share the same number of elements"
    new_data = Dict{myInt, T}()
    for key in keys(lhs.data)
        if haskey(rhs.data, key)
            new_data[key] = lhs.data[key] * rhs.data[key]
        else
            new_data[key] = zero(T) # Multiply by the default value for missing keys
        end
    end
    return LargeSV(lhs.n, new_data)
end

# -- Overloading "*" symbol
function Base.:*(lhs, rhs)
    mult(lhs, rhs)
end 

# -- Idem for tensors 
function Base.:*(A::LargeST{T}, B::LargeST{G}) where {T, G}
    result_data = A.data * B.data
    return LargeST(A.N, A.dims, result_data)
end

# -- Overloading the Base.sum
function Base.sum(v :: LargeSV{T}) where T
    res = zero(T)
    for val in values(v.data)
        res += val
    end 
    return res 
end 
# -- Overload for tensors
@inline Base.sum(A :: LargeST{T}) where T = sum(A.data)

# -- Summing over all but one dimension of a tensor
function Base.sum(A::LargeST{T}, dim) where T
    l = A.dims[dim]
    result_data = zeros(T, l)
    for i = 1:l
        idx = [(j == dim) ? i : Colon() for j = 1:A.N]
        result_data[i] = sum(A[idx...])
    end 
    return result_data
end 