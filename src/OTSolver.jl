module OTSolver 
    using JuMP, Random, ProgressBars, HiGHS
    include("utils.jl")
    include("largethings.jl")
    include("domain.jl")
    include("problem.jl")
    include("viz.jl")
end 