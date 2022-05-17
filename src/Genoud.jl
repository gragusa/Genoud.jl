module Genoud
using Parameters
using StatsFuns
using StatsBase
using OptimBase
import OptimBase: converged, f_converged, g_converged, f_tol, g_tol
using GalacticOptim
using GalacticOptimJL
using Random
using Printf
using Sobol
using LinearAlgebra
#using MathProgBase
#using Ipopt
# package code goes here

const rtol  = sqrt(eps(one(1.0)))
const FLOAT = eltype(one(1.0))

const XNM = ["X₁", "X₂", "X₃", "X₄", "X₅", "X₆", "X₇", "X₈ ", "X₉", "X₁₀",
"X₁₁", "X₁₂", "X₁₃", "X₁₄", "X₁₅", "X₁₆", "X₁₇", "X₁₈ ", "X₁₉", "X₂₀",
"X₂₁", "X₂₂", "X₂₃", "X₂₄", "X₂₅", "X₂₆", "X₂₇", "X₂₈ ", "X₂₉", "X₃₀",
"X₃₁", "X₃₂", "X₃₃", "X₃₄", "X₃₅", "X₃₆", "X₃₇", "X₃₈ ", "X₃₉", "X₄₀",
"X₄₁", "X₄₂", "X₄₃", "X₄₄", "X₄₅", "X₄₆", "X₄₇", "X₄₈ ", "X₄₉", "X₅₀",
"X₅₁", "X₅₂", "X₅₃", "X₅₄", "X₅₅", "X₅₆", "X₅₇", "X₅₈ ", "X₅₉", "X₆₀",
"X₆₁", "X₆₂", "X₆₃", "X₆₄", "X₆₅", "X₆₆", "X₆₇", "X₆₈ ", "X₆₉", "X₇₀"]

## Constant parameters
const NUMBER_OF_TRIES_HC = 1000
const DEBUG = false

struct Domain
    m::Array{Float64, 2}
    p::Array{Float64, 2}   ## Padding for boundary mutation
end

function Domain(x::Array{Float64, 1})
    n = length(x)
    Domain([x.-10*ones(Float64, n) x.+10*ones(Float64, n)],
    [√eps(Float64)*ones(Float64, n) -√eps(Float64)*ones(Float64, n)])
end

function Domain(x::Array{Float64, 2})
    n = length(x)
    Domain(x, [√eps(Float64)*ones(Float64, n) -√eps(Float64)*ones(Float64, n)])
end

Base.length(d::Domain) = size(d.m, 1)::Int64

@with_kw struct Operators{T}
    cloning::T = 50
    uniform_mut::T = 50
    boundary_mut::T = 50
    nonuniform_mut::T = 50
    whole_mut::T = 50
    polytope_cross::T = 50
    simple_cross::T = 50
    heuristic_cross::T = 50
    localmin_cross::T = 0
end

@with_kw struct Options{T, F, B}
#    max_generations::T = 100
    #wait_generations::T = 10
    hard_generation_limit::B = true
    check_gradient::B = false
    freltol::F = 1e-05
    greltol::F = 1e-05
    fabstol::F = 1e-05
    gabstol::F = 1e-05
    boundary_enforcement::B = false
    data_type_int::B = false
    print_level::T = 1
    optim_burnin::T = 0
    hessian::B = false
    pmix::F = 0.5
    bmix::F = 6.0
    initial_selection::B = false
end

mutable struct GenoudOutput
    bestindiv
    bestfitns
    fvals
    xvals
    ftols
    gtols
    grad
    bestgen
    d::Domain
    op::Operators
    opts
end

function Base.sum(op::Operators)
    s = 0
    for fnm in fieldnames(typeof(op))
        s += getfield(op, fnm)
    end
    s
end

function operator_rate(op)
    fnm = fieldnames(typeof(op))
    rate = Vector{Float64}(undef, length(fnm))
    for f in enumerate(fnm)
        rate[f[1]] = getfield(op, f[2])
    end
    rate./sum(op)
end

function Base.in(x::Array, d::Domain)
    isin = true
    for j in eachindex(x)
        if !(x[j] >= d.m[j, 1] && x[j] <= d.m[j, 2])
            isin = false
            break
        end
    end
    isin
end


_rand(a,b) = a + (b-a)*rand()
isinbound(x::Float64, j, d::Domain) = x <= d.m[j,2] && x >= d.m[j,1]

cloning(x) = x

function uniform_mut!(x, d::Domain)
    k = length(d)::Int64
    j = rand(1:k)
    x[j] = _rand(d.m[j,1], d.m[j,2])
    x
end

function boundary_mut!(x, d::Domain)
    k = length(d)::Int64
    j = rand(1:k)
    r = rand(1:2)
    x[j] = d.m[j,r] + d.p[j,r]
    x
end

function nonuniform_mut!(x, bmix, d::Domain, generation, max_generations, boundary_enforcement)
    k = length(d)::Int64
    j = rand(1:k)
    p = rand()*(1-generation/(max_generations+1))^bmix
    r = rand(1:2)
    mx = (1-p)*x[j]+p*d.m[j,r]
    if boundary_enforcement
        x[j] = ifelse(isinbound(mx, j, d), mx, x[j])
    else
        x[j] = mx
    end
    x
end

function polytope_cross(x, d::Domain)
    k = length(d)::Int64
    m = max(2, k)
    p = rand(m)
    p = p./sum(p)
    z = zeros(k)
    for s in 1:k, j in 1:m
        z[s] += p[j]*x[j, s]
    end
    z
end

function simple_cross(x, d::Domain)
    k = length(d)::Int64
    p = 0.5
    z = zeros(k)
    for s in 1:k, j in 1:2
        z[s] += 0.5*x[j, s]
    end
    z
end

function whole_mut!(x, bmix, d::Domain, generation, max_generations, boundary_enforcement)
    k = length(d)::Int64
    for j in 1:k
        p = rand()*(1-generation/max_generations)^bmix
        r = rand(1:2)
        mx = (1-p)*x[j]+p*d.m[j,r]
        if boundary_enforcement
            x[j] = ifelse(isinbound(mx, j, d), mx, x[j])
        else
            x[j] = mx
        end
    end
    x
end


function heuristic_cross(x, d::Domain)
    k = length(d)::Int64
    z = Array{Float64}(undef, k)
    attempts = 0
    while true
        p = rand()
        for s = 1:k
            z[s] = p*(x[1,s]-x[2, s]) + x[1,s]
        end
        if z ∈ d
            break
        end
        if attempts > NUMBER_OF_TRIES_HC
            for s = 1:k
                z[s] = (x[1,s] + x[2,s])/2
            end
            break
        end
        attempts += 1
    end
    z
end

function Random.rand(d::Domain)
    k = length(d)
    u = Array{Float64}(k)
    rand!(u)
    for j in 1:k
        u[j] = d.m[j,1] + (d.m[j,2]-d.m[j,1])*u[j]
    end
    x
end

function Random.rand!(X, d::Domain)
    k = length(d)
    u = rand(size(X, 2))
    for j in 1:k
        copy!(view(X, j, :), d.m[j,1] + (d.m[j,2]-d.m[j,1]).*u)
        rand!(u)
    end
    X
end


function checkdomain(d::Domain, x)
    @assert size(d.m, 1) == length(x) ""
    @assert all(d.m[:,1] .<= d.m[:,2]) ""
    @assert all(d.m[:,1] .<= x) && all(d.m[:,2] .>= x) ""
end


function mutation(population, fitness, smplidx, fitidx, idx, domains::Domain, generation, max_generations, boundary_enforcement, bmix)
    offspring = copy(population)
    _, k = size(population)
    ## Perform cloning
    for i in 1:idx[1]
        for j = 1:k
            offspring[i,j] = population[i,j]
        end
    end
    ## Uniform mutation
    for i in idx[1]+1:idx[2]
        for j = 1:k
            offspring[i,j] = population[i,j]
        end
        uniform_mut!(view(offspring, i, :), domains)
    end
    ## Boundary mutation
    for i in idx[2]+1:idx[3]
        for j = 1:k
            offspring[i,j] = population[i,j]
        end
        boundary_mut!(view(offspring, i, :), domains)
    end
    
    ## Non uniform mutation
    for i in idx[3]+1:idx[4]
        for j in 1:k
            offspring[i,j] = population[i,j]
        end        
        nonuniform_mut!(view(offspring, i, :), bmix, domains, generation, max_generations, boundary_enforcement)
    end
    ## Polytope crossover
    for i in idx[4]+1:idx[5]
        if i + k > idx[5]
            p = i+k-idx[5]
            pidx = (idx[5]-k-p):(idx[5]-p)
        else
            pidx = i:i+k
        end
        offspring[i,:] = polytope_cross(view(population, pidx, :), domains)
    end
    ## Simple Cross
    for i in (idx[5]+1):idx[6]
        if i + 2 > idx[5]
            p = i+2-idx[5]
            tidx = (idx[5]-2-p):(idx[5]-p)
        else
            tidx = i:i+2
        end
        offspring[i,:] = simple_cross(view(population, tidx, :), domains)
    end
    ## Whole mutation
    for i in idx[6]+1:idx[7]
        for j = 1:k
            offspring[i,j] = population[i,j]
        end
        whole_mut!(view(offspring, i, :), bmix, domains, generation, max_generations, boundary_enforcement)
    end
    
    ## Heuristic mutation
    for i in idx[7]+1:idx[8]
        offspring[i,:] = heuristic_cross(view(population, i:i+1, :), domains)
    end
    return offspring
end

function print_problem_info(op, opts, popsize, d, wait_generations, max_generations)
    k = length(d)
    println("Domains:")
    for j in 1:k
        println(d.m[j,1], " <= ", "X", j, " <=", d.m[j,2])
    end
    
    opnames = ["Cloning...........................  ",
    "Uniform mutation..................  ",
    "Boundary mutation.................  ",
    "Nonuniform mutation...............  ",
    "Polytope crossover................  ",
    "Simple crossover..................  ",
    "Whole nonuniform mutation.........  ",
    "Heuristic crossover...............  ",
    "Local-minimum crossover...........  "]
    println("Operators:")
    for j in enumerate(fieldnames(typeof(op)))
        println("       (",j[1],") ", opnames[j[1]], getfield(op, j[2]))
    end
    println("")
    println("Population size.....................:  ", popsize)
    println("Maximum number of generation........:  ", max_generations, ifelse(opts.hard_generation_limit, " (HARD)", ""))
    println("Maximum nonchanging generations.....:  ", wait_generations)
    println("Convergence tolerance...............:  (f rel): ", opts.freltol, "(f abs): ", opts.fabstol)
    println("")
    # if sense == :Max
    #     println("Maximization problem")
    # elseif sense == :Min
        println("Minimization problem.")
    # end
end

function print_generation_info(generation, fitness, population, bestindiv, bestfitns, print_level)
    printstyled("Generation ", string(generation)*"\n"; color = :blue)
    printstyled("Fitness (best)...:   "*string(bestfitns)*"\n"; color = :cyan)
    println("  mean...........:   ", mean(fitness))
    println("  variance.......:   ", var(fitness))
    println("  unique.........:   ", length(unique(fitness)))
    if print_level > 1
        meanindiv = mean(population, 2)
        varindiv = var(population, 2)
        for k in 1:size(population, 1)
            printstyled(XNM[k]*":\n"; color = :cyan)
            println("  best...........:   ", bestindiv[k])
            println("  mean...........:   ", meanindiv[k])
            println("  var............:   ", varindiv[k])
        end
        println("")
    end
end

function splits(op::Operators, popsize, k)
    ## Calculate operator rate and indexes
    rate = operator_rate(op)
    op_split = round.(Int, rate*popsize)
    ## Oprator that need positive population
    op_split[6] = isodd(op_split[6]) ? (op_split[1] -= 1; op_split[6] + 1) : op_split[6]
    op_split[8] = isodd(op_split[8]) ? (op_split[1] -= 1; op_split[8] + 1) : op_split[8]
    
    if op_split[5] != 1
        r = k-rem(op_split[5]÷k, k)
        op_split[5] += r
        op_split[1] -= r
    end
    
    sum_split = sum(op_split)
    if sum_split == popsize
        op_split[1] -= 1
    elseif sum_split < popsize
        op_split[1] += popsize - (sum_split + 1)
    elseif sum_split > popsize
        op_split[1] -= (sum_split + 1) - popsize
    end
    cumsum(op_split)[1:end-1]
end

function _describe(x)
    q = quantile(x, [.25, .5, .75])
    [minimum(x); q; maximum(x)]
end


# mutable struct NonDifferentiable <: NLSolversBase.AbstractObjective
#     f::Function
#     x0::Array{Float64, 1}
# end

#NonDifferentiable(f::Function, x0::Vector) = NonDifferentiable(f, float(x0))



# struct Genoud_MPB{T, F} <: MathProgBase.AbstractNLPEvaluator 
#     f::T
#     g!::F
# end

# Genoud_MPB(d::Optim.OnceDifferentiable) = Genoud_MPB(d.f, d.g!)
# Genoud_MPB(d::Optim.NonDifferentiable) = Genoud_MPB(d.f, identity)


# MathProgBase.features_available(g::Genoud_MPB) = [:Grad]
# function MathProgBase.initialize(d::Genoud_MPB, requested_features::Vector{Symbol})
#     for feat in requested_features
#         if !(feat in [:Grad])
#             error("Unsupported feature $feat")
#         end
#     end
# end 
# MathProgBase.jac_structure(g::Genoud_MPB) = Int[],Int[]
# MathProgBase.eval_jac_g(g::Genoud_MPB, J, x) = nothing
# MathProgBase.eval_f(g::Genoud_MPB, x) = g.f(x)
# MathProgBase.eval_grad_f(g::Genoud_MPB, gr, x) = g.g!(gr, x)    

# genoud(d::NLSolversBase.AbstractObjective, popsize::Int64, domain::Domain; kwargs...) = genoud(d, popsize, domain.m[:,1], domain.m[:,2]; kwargs...)



function genoud(prob::GalacticOptim.OptimizationProblem; 
    popsize::Int64 = 100, 
    optimize_best = typeof(prob.f.adtype) <: SciMLBase.NoAD ? false : true,
    operators::Operators = Genoud.Operators(),
    opt = Options(), max_generations::Int64  = 100, 
    wait_generations::Int64 = 10, solver = BFGS())
    
    f0  = prob.f
    x0 = prob.u0
    p  = prob.p
    lb = prob.lb
    ub = prob.ub
    domain = Domain([float.(lb) float.(ub)])

    f = GalacticOptim.instantiate_function(f0, x0, f0.adtype, p)

    

    ## Check
    ## checkdomain(domain, x0)
    
    ## Number of parameters
    k = length(x0)
    ## Get splits for operator application
    idx = splits(operators, popsize, k)
    ## Tolerances
    freltol = opt.freltol
    fabstol = opt.fabstol
    greltol = opt.greltol
    gabstol = opt.gabstol

    boundary_enforcement = opt.boundary_enforcement
    hard_generation_limit = opt.hard_generation_limit    
    optim_burnin = opt.optim_burnin
    print_level = opt.print_level
    initial_selection = opt.initial_selection
    check_gradient = opt.check_gradient
    bmix = opt.bmix::Float64
    pmix = opt.pmix::Float64
    ## Set the solver
    
    # Initialize population
    s = SobolSeq(lb, ub)
    population  = collect(reduce(hcat, next!(s) for i = 1:popsize)')
    offspring   = similar(population)     ## Should pass this down
    fitness     = Vector{Float64}(undef, popsize)                       
    smplidx     = collect(1:popsize)
    #=
    ## Initialize storages
    =#    
    ftols = [+Inf]
    gtols = [+Inf]
    gtol  = 0.0
    ftol  = 0.0
    grx   = Vector{Float64}(undef, k)

    #=
    ## Print problem info
    =#
    print_level > 0 && print_problem_info(operators, opt, popsize, domain, wait_generations, max_generations)
    #=
    ## Set generation
    =#
    generation = 0
    #=
    ## Calculate initial fitness value
    =#
    for i in 1:popsize
        fitness[i] = f(population[i,:], p)
    end
    fitidx = StatsBase.competerank(fitness)
    permidx = sortperm(fitidx)
    #=
    ## Resample population
    =#
    smplprob = Vector{Float64}(undef, popsize)
    if initial_selection
        smplprob .= pmix.*((1-pmix).^(fitidx-1))
        sample!(1:popsize, Weights(smplprob), smplidx)
        population = population[smplidx, :]
        fitness = fitness[smplidx]
        DEBUG && Base.show(_describe(smplprob))
    end
    #=
    ## Best individual
    =#
    bestfitns, idxmin = findmin(fitness)
    bestindiv =  population[idxmin, :]
    xvals = [bestindiv]
    fvals = [bestfitns]
    
    DEBUG && println("\n Best individual:\n")
    DEBUG && Base.show(bestindiv)
    DEBUG && println("Best fitness:\n")
    DEBUG && show(bestfitns)
    DEBUG && check_gradient && println("Gradient:\n")
    DEBUG && check_gradient && show(bestfitns)
    
    print_level >0 && print_generation_info(generation, fitness, population, bestindiv, bestfitns, print_level)
    
    while true
        #=
        Mutate population
        =#
        population = mutation(population, fitness, smplidx, fitidx, idx, domain, generation, 
                              max_generations, boundary_enforcement, bmix)
        #=
        Calculate fitness
        =#
        for i in 1:popsize
            fitness[i] = f(population[i, :], p)
        end
        #=
        Rank fitness and get best individuals
        =#
        sortperm!(permidx, fitness, rev = false)
        StatsBase._competerank!(fitidx, fitness, permidx)
        current_bestfitns, minidx = findmin(fitness)
        current_bestindiv =  population[minidx, :]
        
        generation += 1
        #=
        Apply solver to best individual of mutated population
        =#
        if optim_burnin < generation + 1 && optimize_best
            try
                DEBUG && println("Running BFGS on:\n")
                DEBUG && show(current_bestindiv)
                copy!(prob.u0, current_bestindiv)
                sol = solve(prob, solver)
                DEBUG && println("SOLVER OUTPUT")
                DEBUG && @show sol.original
                solution = sol.minimum
                if solution < current_bestfitns
                    current_bestindiv = sol.u
                    current_bestfitns = solution
                end
            catch exception
                DEBUG && (println("SOLVER FAILED WITH:\n"); println(exception))
                print_level > 0 && printstyled("Solver on best individual failed\n", color = :red)
            end
        end
        #=
        Print info
        =#
        print_level > 0 && print_generation_info(generation, fitness, population, bestindiv, bestfitns, print_level)
        #=
        Calculate tolarances
        =#
        DEBUG && @show current_bestfitns
        DEBUG && @show bestfitns

        ftol = norm(current_bestfitns-bestfitns)/(fabstol + freltol*max(norm(bestfitns), norm(current_bestfitns)))

        if check_gradient
            grx = f.grad(current_bestindiv, p0)
            gtol = norm(grx)/(gabstol + greltol*norm(grx))
        end
        #=
        ## Store tolerance level
        =#
        push!(fvals, bestfitns)
        push!(xvals, bestindiv)
        push!(ftols, ftol)        
        check_gradient && push!(gtols, gtol)
        #=
        ## Copy current to actual
        =#
        bestfitns = copy(current_bestfitns)
        bestindiv = copy(current_bestindiv)
        #=
        Check exit conditions
        =#
        if ftol < 1 && gtol < 1
            if length(fvals) > wait_generations && maximum(abs.(fvals[end-wait_generations+2:end] .- fvals[end-wait_generations+1])) <= freltol
                break
            end
        end
        
        if generation > max_generations
            break
        elseif generation == max_generations
            if hard_generation_limit
                print_level > 1 && warn("Number of max generation limit reached, but the fitness value is still changing")
                break
            else
                print_level > 1 && warn("Number of max generation limit reached, increasing max number of generation from "*string(max_generations)*" to "*string(max_generations+10))
                generation += 10
            end
        end
        
        ## Resample population
        sortperm!(permidx, fitness, rev = false)
        StatsBase._competerank!(fitidx, fitness, permidx)
        smplprob .= pmix.*((1.0.-pmix).^(fitidx.-1))
        sample!(1:popsize, Weights(smplprob), smplidx)
        population = population[smplidx,:]
        #=
        Carry over best of previous generation
        =#
        population[1, :] = bestindiv
        fitness[1] = bestfitns
        
    end
    bestgen = findmin(fvals)[2]-1
    GenoudOutput(bestindiv,
    bestfitns,
    fvals,
    xvals,
    ftols,
    gtols,
    grx,
    bestgen,
    domain,
    operators,
    opt)
end

OptimBase.f_converged(r::GenoudOutput) = last(r.ftols) <= r.opts.freltol
OptimBase.g_converged(r::GenoudOutput) = last(r.gtols) <= r.opts.greltol

OptimBase.f_tol(r::GenoudOutput) = r.opts.freltol
OptimBase.g_tol(r::GenoudOutput) = r.opts.fabstol

function OptimBase.converged(r::GenoudOutput)
    f_converged(r) || r.opts.check_gradient && g_converged(r)
end

generations(r::GenoudOutput) = length(r.ftols) - 1
#
function Base.show(io::IO, r::GenoudOutput)
    @printf io "Results of Genoud Optimization Algorithm\n"
    optimum = r.bestindiv
    # if r.sense == :Min
        if length(join(optimum, ",")) < 40
            @printf io " * Minimizer: [%s]\n" join(optimum, ",")
        else
            @printf io " * Minimizer: [%s, ...]\n" join(optimum[1:2], ",")
        end
        @printf io " * Minimum: %e\n" r.bestfitns
    # else
    #     if length(join(optimum, ",")) < 40
    #         @printf io " * Maximizer: [%s]\n" join(optimum, ",")
    #     else
    #         @printf io " * Maximizer: [%s, ...]\n" join(optimum[1:2], ",")
    #     end
    #     @printf io " * Maximum: %e\n" r.bestfitns
    # end
    
    @printf io " * Pick generation: %d\n" r.bestgen
    @printf io " * Convergence: %s\n" converged(r)
    @printf io "   * |f(x) - f(x')| / |f(x)| < %.1e: %s\n" f_tol(r) f_converged(r)
    if r.opts.check_gradient
        @printf io "   * |g(x)| < %.1e: %s\n" g_tol(r) g_converged(r)
    end
    @printf io "   * Number of Generations: %s\n" generations(r)
end

function Base.isapprox(x::Array{Float64, 1}, y::Float64)
    for j in eachindex(x)
        isapprox(x[j], y) || return false
    end
    true
end

bestindiv(g::Genoud.GenoudOutput) = g.bestindiv
bestfitns(g::Genoud.GenoudOutput) = g.bestfitns
bestgen(g::Genoud.GenoudOutput)   = g.bestgen

export genoud, OnceDifferentiable, NonDifferentiable
end  #module
