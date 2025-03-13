
# -- Transform (i_1, \dots, i_N) to a linear index 

function indices_to_linear(dims, indices)
    N = length(dims)
    @assert length(indices) ==  N "Number of indices must match tensor dimensions"
    idx = indices[1] - 1
    stride = 1
    for i in 2:N
        stride *= dims[i-1]
        idx += (indices[i] - 1) * stride
    end
    return idx + 1
end

function linear_to_indices(dims, linear) 
    @assert 1 ≤ linear ≤ prod(dims) "Linear index out of range"
    indices = []
    remaining = linear - 1 
    for dim in reverse(dims)
        push!(indices, rem(remaining, dim) + 1)
        remaining = div(remaining, dim)
    end
    return indices
end

# -- Stupid little function 
function change_dimension(tuple, i, new_value) 
    return [tuple[1:i-1]..., new_value, tuple[i+1:end]...]
end


