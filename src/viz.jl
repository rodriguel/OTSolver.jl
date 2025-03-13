# -- Retrieve the two point marginal wrt to marginals i and j
function two_point_density(A :: LargeST{T}, i, j) where T
    l_i, l_j = A.dims[i], A.dims[j]
    res = zeros(l_i, l_j)
    for k = 1:l_i
        for l = 1:l_j
            idx = []
            for a = 1:A.N
                if a == i
                    push!(idx, k)
                elseif a == j
                    push!(idx, l)
                else
                    push!(idx, Colon())
                end
            end 
            res[k,l] = sum(A[idx...])
        end
    end 
    return res 
end 