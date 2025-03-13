# Implementation of the Problem
struct Problem
    C       # Cost (sparse) tensors
    m       # Marginals 
    Π       # Current (optimal) transport plan
    ψ       # Current (optimal) Kantorovich potential
    Ω       # The domain ... Ω, evidently
    c       # The cost function c(r_1, \dots, r_N)
end 

# -- Function that implements the tail-clearing procedure 
@inline function clearing!(P :: Problem, β)
    d = P.Ω.d
    if size(P.Ω) > β * P.Ω.N * P.Ω.L^d
        # -- again, this could be "dropped" since the setindex now automatically implements this
        dropzeros!(P.Π)
        # Fetch indices where the plan is non-zero
        indices_plan, _ = findnz(P.Π)
        # Fetch indices where the domain is active
        indices_grid, _ = findnz(P.Ω.grid)
        # Difference (in the set theory sense) of these two sets of indices 
        indices = setdiff(indices_grid, indices_plan)

        for i in indices 
            P.Ω.grid[i] = 0
        end
        # -- again, the following is probabily useless [I need to take care of this...]
        dropzeros!(P.Ω.grid)
    end 
end 


# -- Solving the (small) LP problem
function solve_lp!(P :: Problem)
    N = P.Ω.N
    supp = P.Ω.grid
    dims = supp.dims
    nb_vars = size(P.Ω)

    # Creating the LP to solve 
    model = Model(HiGHS.Optimizer);
    set_silent(model)
    # Setting up the variables
    @variable(model, sol[1:nb_vars] .>= 0);
    # The reason why I implemented the sparse tensors and vectors with general type T :
    Π = sptzeros(JuMP.AffExpr, dims)    
    # Ok, fetch point of the support (this is the domain Ω, simply...)
    idxs, _ = findnz(supp)
    # ... still in the "setting up variables thing"...
    for (i, idx) in enumerate(idxs)
        Π[idx] = sol[i]
        set_start_value(sol[i], P.Π[idx])
    end 
    # Ok, now, I implement the marginal constraints 
    Π_Ω = Π*supp
    for i = 1:N
        @constraint(model, sum(Π_Ω, i) .== P.m[i])  
    end 

    # Build the objective and optimize !!! 
    @objective(model, Min, sum(Π_Ω*P.C));
    optimize!(model);

    # We retrieve the solution
    val = objective_value(model)
    sol = value.(sol)
    for (i, idx) in enumerate(idxs)
        P.Π[idx] = sol[i]
    end 

    return sol, val
end 

# -- Function that checks if the dual constraint is violated smwhr
@inline function is_violated(P :: Problem, idx)
    sum_ψs = 0
    x = to_grid(P.Ω, idx)
    for (i, j) in enumerate(idx)
        sum_ψs += P.ψ[i, j]
    end 
    (P.c(x) - sum_ψs < 0) ? true : false
end 

# -- Like the previous one, implements the Hamiltonian c(r_1, \dots, r_N) - \phi...
function hamiltonian(P :: Problem, indices...) # [should use the ... this is completely silly, anyway]
    x = to_grid(P.Ω, indices)
    cost = P.c(x)
    sum_ψs = 0
    for (i,j) in enumerate(indices)
        sum_ψs += P.ψ[i, j]
    end
    return cost - sum_ψs
end


# -- Update cost given the domain has been updated
function update_cost(P :: Problem)
    idxs = indices_where_A_zero_but_B_not(P.C, P.Ω.grid)
    for idx in idxs
        idx_grid = linear_to_indices(P.Ω.grid.dims, idx)
        x = to_grid(P.Ω, idx_grid)
        P.C[idx] = P.c(x)
    end 
end



# -- Solving the (small) dual of the LP (= Kantorovich duality)
function solve_dual!(P :: Problem)
    d = P.Ω.d
    L = P.Ω.L^d;        # [Again, this is shady... ]
    N = P.Ω.N
    supp = P.Ω.grid
    dims = supp.dims

    # ... idem as before
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, ψs[1:N, 1:L])
    for i = 1:N
        for j = 1:L
        set_start_value(ψs[i, j], P.ψ[i,j])
        end
    end
    
    idxs, _ = findnz(supp.data)
    # Adding the usual good 'ld constraints
    for (i, idx) in enumerate(idxs)
        is = linear_to_indices(dims, idx)
        sum_ψs = sum([ψs[j, is[j]] for j = 1:N])
        @constraint(model, P.C[idx] >= sum_ψs)
    end

    # Setting up the objective and optimize !
    sum_ints_ψs = sum([ψs[j, :]'P.m[j] for j = 1:N])
    @objective(model, Max, sum_ints_ψs)
    optimize!(model)

    # ...
    val = objective_value(model)
    sol = value.(ψs)
    for i = 1:N
        for j = 1:L
        P.ψ[i, j] = sol[i, j]
        end
    end 

    return sol, val
end

# -- Solving the entire LP (to be used for small N @ small L)
function exact_solve(P :: Problem)
    model = Model(HiGHS.Optimizer);

    # Big (Big!) tensors -- will certainly crash
    C = zeros(P.Ω.dims...)
    Π = zeros(JuMP.AffExpr, P.Ω.dims...)
    I = CartesianIndices(C)
    for i in I
        x = [P.Ω.xs[j] for j in Tuple(i)]
        C[i] = P.c(x...)
        Π[i] = @variable(model)
        @constraint(model, Π[i] >= 0)
    end
    # Adding the marginal constraints
    for i = 1:P.Ω.N
        dims_ = filter(j -> j != i, [k for k = 1:P.Ω.N])
        mi = dropdims(sum(Π, dims = dims_), dims = Tuple(dims_))
        @constraint(model,  mi .== P.m[i])  
    end 
    # Let it get slow :) !
    @objective(model, Min, sum(Π.*C));
    optimize!(model);
    sol, val = value.(Π), objective_value(model)
    return sol, val

end 

# -- This is a completely chatgpt-ed implementation of the north west rule corner [my head hearts...]
function northwest_corner_rule(marginals::Vector{Vector{T}}) where {T<:Real}
    N = length(marginals)  # Number of dimensions
    sizes = map(length, marginals)  # Size of each dimension
    X = sptzeros(T, sizes)  # Initialize tensor
    
    # Convert marginals to mutable arrays
    remaining = [copy(μ) for μ in marginals]

    # Multi-dimensional index tracking
    indices = fill(1, N)

    while all(i -> indices[i] ≤ sizes[i], 1:N)
        # Determine amount to place at current position
        min_value = minimum(remaining[i][indices[i]] for i in 1:N)
        X[indices] = min_value

        # Subtract the allocated amount from the marginals
        for i in 1:N
            remaining[i][indices[i]] -= min_value
        end

        # Advance indices using NW rule logic
        for i in 1:N
            if remaining[i][indices[i]] == 0
                indices[i] += 1
                break
            end
        end
    end

    return X
end


##########################################################################
#### --- OK, in this section, I implement the original GenCol algo'.  ####
##########################################################################

# -- Function that picks a "parent" at random in Ω
@inline function pick_parent(P::Problem)
    idx, _ = findnz(P.Ω.grid)
    i = rand(idx)
    res = linear_to_indices(P.Ω.grid.dims, i)
    return res 
end

# -- Function that picks a "children" given a parent [= meaning a neighbor that differs by only one dimension ]
# [It is still completely valid in dimension d > 1 ? I mean, it is what I should do ?]
function random_unoccupied_neighborhood(A :: LargeST{T}, indices) where T
    N = A.N 
    dims = A.dims 
    dim = rand(1:N)
    o = rand((-1, 1))
    offset = [(j == dim) ? o : 0 for j = 1:N]
    neighbor = indices .+ offset
    if all(1 .<= neighbor .<= dims) && A[neighbor...] == zero(T)
        return neighbor
    end 
    nothing 
end 
# -- Basically just a wrapper, don't known why I did that ...
@inline function pick_children(P :: Problem, idx)
    random_unoccupied_neighborhood(P.Ω.grid, idx)
end

# ------ Here's a slighty modified version for which I can prove convergence of the algorithm in the marginal setting 
function _random_unoccupied_neighborhood(A::LargeST{T}, indices) where {T}
    dims = A.dims
    N = A.N
    neighborhood_offsets = shuffle(collect(Iterators.product(((-1:1) for _ in 1:N)...)))
    for offset in neighborhood_offsets
        neighbor_index = indices .+ offset
        # [ Probably (certainly) not very efficient, I should think about that...]
        if all(1 .<= neighbor_index .<= dims) && any(abs.(offset) .!= 0) && sum(abs.(collect(offset))) <= N-1
            if A[neighbor_index...] == zero(T)
               return neighbor_index
            end
        end
    end
    return nothing
end


# -- The full update
function update!(P :: Problem)
    # Pick a parent 
    idx = pick_parent(P)
    # Pick a child
    child = pick_children(P, idx)
    if child !== nothing # [Maybe I should...do smthg about that ?]
        # Does the child violates the constraint ?
        if is_violated(P, child)
            # It did ! Add it, then, don't waste your time !
            P.Ω.grid[child] = 1.
            return true
        end
    end 
    return false
end

# -- The final function, that solves the problem (niter - times)
# (β is the tail-clearing parameter)
function solve(P :: Problem, niter, β = 3)
    vals = zeros(niter); val = 0
    updated = true
    for k in ProgressBar(1:niter)
        # Solve small LPs
        if updated 
            _, val = solve_lp!(P);
            solve_dual!(P);
        end
        # Keep value of objective 
        vals[k] = val
        # Ok, update the domain 
        updated = update!(P)
        # If an update has been made, update the cost matrix
        if updated 
            update_cost(P)
        end
        # Finally, the tail-clearing procedure 
        clearing!(P, β)
    end
    return vals
end 


###########################################################################
#### --- OK, in this section, I implement the Argmax-ed GenCol algo'.  ####
###########################################################################

# -- Pick argmax along a dimension 
@inline function pick_argmax(P :: Problem, idx_parent)
    # Choose at random a dimension 
    dim = rand(1:P.Ω.N)
    d = P.Ω.d
    L = P.Ω.L^d # -- again and again, I should make this more intuitive
    vals = zeros(L)
    for i = 1:L
        idx = change_dimension(idx_parent, dim, i)
        vals[i] = -hamiltonian(P, idx...)
    end 
    max_index = argmax(vals)
    return vals[max_index], dim, max_index
end 

# -- This is one is for the Pricing Problem [not used, then, in practice]
function _pick_argmax(P::Problem, idx_parent, i_dim)
    vals = zeros(P.Ω.L)
    for i = 1:P.Ω.L
        idx = change_dimension(idx_parent, i_dim, i)
        vals[i] = -hamiltonian(P, idx...)
    end 
    return argmax(vals)
end 

# -- Update function for Argmax-ed GenCol
function update_argmax(P :: Problem)
    # Pick parent
    idx = pick_parent(P)
    # Pick argmax along random dimension
    val, dim, i_max = pick_argmax(P, idx)
    new_idx = change_dimension(idx, dim, i_max)
    # And add the culprit if it violates the constraint 
    if val > 0 && P.Ω.grid[new_idx...] == 0
        add!(P.Ω, new_idx)
        return true 
    end 
    return false
end 

# -- And finally the full procedure 
function solve_argmax(P :: Problem, niter, β = 3)
    vals = zeros(niter); val = 0
    updated = true 
    for i in ProgressBar(1:niter)
        # Solving small LPs
        if updated 
            _, val = solve_lp!(P);
            solve_dual!(P);
        end
        vals[i] = val
        # Update using Argmax...
        updated = update_argmax(P)
        # If the domain changes, update the cost tensor
        if updated 
            update_cost(P)
        end
        # Tail-clearing procedure 
        clearing!(P, β)
    end
    return vals
end 

##########################################################################
#### --- OK, in this section, I implement the Annealed GenCol algo'.  ####
##########################################################################


# -- Implementation of the simulated annealing (honestly, chatgpt-ed quickly)

# 1) A function that generate a random element in the (full grid)
function generate_initial_solution(A::LargeST{T}) where T
    len = prod(A.dims)
    #zind = setdiff(1:len, A.data.nzind) # -- [if you want to start with a point in Ω]
    res = rand(1:len)
    return linear_to_indices(A.dims, res)
end

# 2) ...picks a random neighorhood, the name says it all
function my_random_neighborhood(A::LargeST{T}, indices...) where {T}    
    dim = rand(1:A.N)
    L = A.dims[dim]
    pos = indices[dim]
    if pos == L
        return change_dimension(indices, dim, L - 1)
    elseif pos == 1
        return change_dimension(indices, dim, 2)
    else 
        ofs = rand((-1,1))
        return change_dimension(indices, dim, pos + ofs)
    end 
end 

# 3) the chatgpt-ed SA algorithm
function simulated_annealing(tensor::LargeST{T}, objective_function :: Function, max_iterations::Int, initial_temperature::Float64, cooling_rate::Float64) where T
    current_solution = generate_initial_solution(tensor)
    current_objective = objective_function(current_solution)
    best_solution = current_solution
    best_objective = current_objective
    temperature = initial_temperature
    for iter in 1:max_iterations
        new_solution = my_random_neighborhood(tensor, current_solution...)
        if new_solution === nothing 
            break
        end 
        new_objective = objective_function(new_solution)
        # -- I add this : if the constraint is violated, just stop the annealing... [Is it intelligent or stupid or we don't really care ?]
        if new_objective < 0
            return new_solution, new_objective
        end
        if new_objective < current_objective || rand() < exp((current_objective - new_objective) / temperature)
            current_solution = new_solution
            current_objective = new_objective
        end
        if new_objective < best_objective
            best_solution = new_solution
            best_objective = new_objective
        end
        temperature *= cooling_rate
    end
    return best_solution, best_objective
end


# -- Update function for Annealed GenCol
# T = is starting temperature and tc is cooling factor 
function update_annealing!(P :: Problem, niter, T, tc)
    energy = is -> hamiltonian(P, is...)
    sol, val = simulated_annealing(P.Ω.grid, energy, niter, T, tc)
    if val < 0 && P.Ω.grid[sol...] == 0
        P.Ω.grid[sol] = 1.
        return true
    else 
        return false
    end 
end 

# -- And, finally, the full algorithm
function solve_annealing(P :: Problem, niter, niter_annealing, T, tc, β = 3)
    vals = zeros(niter); val = 0
    updated = true
    for i in ProgressBar(1:niter)
        if updated 
            _, val = solve_lp!(P);
            solve_dual!(P);
        end
        vals[i] = val
        updated = update_annealing!(P, niter_annealing, T, tc)
        if updated
            update_cost(P)
        end 
        clearing!(P, β)
    end
    return vals
end 


############################################################################################
#### --- This is a mesh-refinement strategy, just for the heck of it  (not implemented) ####
############################################################################################

function mesh_refinement(tensor::LargeST{T}) where {T,N}
    new_dims = map(x -> 2 * x, tensor.dims)
    new_data = spzeros(prod(new_dims))
    
    for (linear_index, value) in zip(tensor.data.nzind, tensor.data.nzval)
        old_indices = linear_to_indices(tensor.dims, linear_index)
        
        for offset in Iterators.product(ntuple(_ -> 0:1, N)...)
            new_indices = ntuple(i -> 2 * old_indices[i] - 1 + offset[i], N)
            new_linear_index = indices_to_linear(new_dims, new_indices...)
            new_data[new_linear_index] = value
        end
    end

    return SparseTensor(new_data, new_dims)
end

function mesh_refinement(vector::Vector{T}) where {T}
    new_length = 2 * length(vector)
    new_vector = zeros(T, new_length)
    
    for i in 1:length(vector)
        value = vector[i]
        new_vector[2 * i - 1] = value
        new_vector[2 * i] = value
    end
    
    return new_vector
end


function refinement(P :: Problem)
    N = P.Ω.N; L = P.Ω.L
    new_Ω = mesh_refinement(P.Ω.grid)
    new_D = Domain(2*L, N, Tuple((2*L for _ in 1:N)), _xs(2*L), new_Ω)
    new_C = build_cost(new_D, P.c)
    new_m = [mesh_refinement(P.m[i])/2 for i = 1:N]
    new_Π = mesh_refinement(P.Π)
    new_ψs = zeros(N, 2*L)
    return Problem(new_C, new_m, new_Π, new_ψs, new_D, P.c);
end



