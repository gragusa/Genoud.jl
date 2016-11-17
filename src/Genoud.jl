module Genoud
using MathProgBase
using OptimMPB
using Parameters
using StatsFuns
using StatsBase
# package code goes here

const rtol  = sqrt(eps(1.0))
const FLOAT = eltype(1.0)

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

immutable Domain
  m::Array{Float64, 2}
  p::Array{Float64, 2}   ## Padding for boundary mutation
end

Base.length(d::Domain) = size(d.m, 1)::Int64

function Domain(x::Array{Float64, 1})
  n = length(x)
  Domain([x.-10*ones(Float64, n) x.+10*ones(Float64, n)],
         [√eps(Float64)*ones(Float64, n) -√eps(Float64)*ones(Float64, n)])
end

function Domain(x::Array{Float64, 2})
  n = length(x)
  Domain(x, [√eps(Float64)*ones(Float64, n) -√eps(Float64)*ones(Float64, n)])
end

@with_kw immutable GenoudOperators{T}
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

@with_kw immutable GenoudOptions{T, F, B}
    max_generations::T = 100
    wait_generations::T = 10
    hard_generation_limit::B = true
    check_gradient::B = false
    f_tol::F = 0.001
    g_tol::F = 0.001
    boundary_enforcement::B = false
    data_type_int::B = false
    print_level::T = 1
    optim_burnin::T = 0
    hessian::B = false
    pmix::F = 0.5
    bmix::F = 6.0
    initial_selection::B = true
end

type GenoudOutput
  bestindiv
  bestfitns
  fvals
  xvals
  ftols
  gtols
  grad
  bestgen
  sense
  d::Domain
  op::GenoudOperators
  opts::GenoudOptions
end

function Base.sum(op::GenoudOperators)
  s = 0
  for fnm in fieldnames(op)
    s += getfield(op, fnm)
  end
  s
end

function operator_rate(op)
  fnm = fieldnames(op)
  rate = Array{Float64}(length(fnm))
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
  p   = rand(m)
  p   = p./sum(p)
  z   = zeros(k)
  for s in 1:k, j in 1:m
     z[s] += p[j]*x[s, j]
  end
  z
end

function simple_cross(x, d::Domain)
  k = length(d)::Int64
  p = 0.5
  z = zeros(k)
  for s in 1:k, j in 1:2
    z[s] += 0.5*x[s, j]
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
  z = Array{Float64}(k)
  attempts = 0
  while true
    p = rand()
    for s = 1:k
      z[s] = p*(x[s,1]-x[s,2]) + x[s,1]
    end
    if z ∈ d
      break
    end
    if attempts > NUMBER_OF_TRIES_HC
      for s = 1:k
        z[s] = (x[s, 1] + x[s, 2])/2
      end
      break
    end
    attempts += 1
  end
  z
end


function Base.Random.rand(d::Domain)
  k = length(d)
  u = Array{Float64}(k)
  rand!(u)
  for j in 1:k
    u[j] = d.m[j,1] + (d.m[j,2]-d.m[j,1])*u[j]
  end
  x
end

function Base.Random.rand!(X, d::Domain)
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

function initialpopulation(d::Domain, n::Int)
    k = length(d)
    X = Array{Float64}(k, n)
    rand!(X, d)
    X
end

function mutation(population, fitness, smplidx, fitidx, idx, domains::Domain, generation, max_generations, boundary_enforcement, bmix)
  offspring = copy(population)
  k, sizepop = size(population)
  ## Perform cloning
  for i in 1:idx[1]
      for j = 1:k
        offspring[j,i] = population[j,i]
      end
  end
  ## Uniform mutation
  for i in idx[1]+1:idx[2]
    for j = 1:k
      offspring[j,i] = population[j,i]
    end
    uniform_mut!(view(offspring, :, i), domains)
  end

  ## Boundary mutation
  for i in idx[2]+1:idx[3]
    for j = 1:k
      offspring[j,i] = population[j,i]
    end
    boundary_mut!(view(offspring, :, i), domains)
  end

  ## Non uniform mutation
  for i in idx[3]+1:idx[4]
    for j = 1:k
      offspring[j,i] = population[j,i]
    end
    nonuniform_mut!(view(offspring, :, i), bmix, domains, generation, max_generations, boundary_enforcement)
  end
  # kidx = Array{Int64}(k)
  # tidx = Array{Int64}(2)
  ## Polytope crossover
  for i in idx[4]+1:idx[5]
    if i + k > idx[5]
       p = i+k-idx[5]
       pidx = (idx[5]-k-p):(idx[5]-p)
    else
      pidx = i:i+k
    end
    offspring[:,i] = polytope_cross(view(population, :, pidx), domains)
  end
  ## Simple Cross
  for i in (idx[5]+1):idx[6]
    if i + 2 > idx[5]
       p = i+2-idx[5]
       tidx = (idx[5]-2-p):(idx[5]-p)
    else
      tidx = i:i+2
    end
    offspring[:,i] = simple_cross(view(population, :, tidx), domains)
  end
  ## Whole mutation
  for i in idx[6]+1:idx[7]
    for j = 1:k
      offspring[j,i] = population[j,i]
    end
    whole_mut!(view(offspring, :, i), bmix, domains, generation, max_generations, boundary_enforcement)
  end

  ## Heuristic mutation
  for i in idx[7]+1:idx[8]
    offspring[:,i] = heuristic_cross(view(population, :, i:i+1), domains)
  end
  return offspring
end


function print_problem_info(op, opts, sizepop, d, sense)
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
  for j in enumerate(fieldnames(op))
    println("       (",j[1],") ", opnames[j[1]], getfield(op, j[2]))
  end
  println("")
  println("Population size.....................:  ", sizepop)
  println("Hard maximum number of generation...:  ", opts.hard_generation_limit)
  println("Maximum nonchanging generations.....:  ", opts.wait_generations)
  println("Convergence tolerance...............:  ", opts.f_tol)
  println("")
  if sense == :Max
    println("Maximization problem")
  elseif sense == :Min
    println("Minimization problem.")
  end
end

function print_generation_info(generation, fitness, population, bestindiv, bestfitns, print_level, ς)
  print_with_color(:blue, "Generation ", string(generation)*"\n")
  print_with_color(:cyan, "Fitness (best)...:   "*string(ς*bestfitns)*"\n")
                  println("  mean...........:   ", ς*mean(fitness))
                  println("  variance.......:   ", var(fitness))
                  println("  unique.........:   ", length(unique(fitness)))
  if print_level > 1
    meanindiv = mean(population, 2)
    varindiv = var(population, 2)
    for k in 1:size(population, 1)
      print_with_color(:cyan, XNM[k]*":\n")
      println("  best...........:   ", bestindiv[k])
      println("  mean...........:   ", meanindiv[k])
      println("  var............:   ",  varindiv[k])
    end
    println("")
  end

end

function splits(op::GenoudOperators, sizepop, k)
  ## Calculate operator rate and indexes
  rate = operator_rate(op)
  op_split = round(Int, rate*sizepop)
  ## Oprator that need positive population
  op_split[6] = isodd(op_split[6]) ? (op_split[1] -= 1; op_split[6] + 1) : op_split[6]
  op_split[8] = isodd(op_split[8]) ? (op_split[1] -= 1; op_split[8] + 1) : op_split[8]

  if op_split[5] != 1
    r = k-rem(op_split[5]÷k, k)
    op_split[5] += r
    op_split[1] -= r
  end

  sum_split = sum(op_split)
  if sum_split == sizepop
    op_split[1] -= 1
  elseif sum_split < sizepop
    op_split[1] += sizepop - (sum_split + 1)
  elseif sum_split > sizepop
    op_split[1] -= (sum_split + 1) - sizepop
  end
  cumsum(op_split)[1:end-1]
end

function _describe(x)
  q = quantile(x, [.25, .5, .75])
  [minimum(x); q; maximum(x)]
end


function genoud(fcn, initial_x::Array{Float64, 1};
  sizepop::Int = 1000, sense::Symbol = :Min,
  domains::Domain = Domain(initial_x),
  optimize_best::Bool = true, gr!::Function = identity,
  optimizer::Optim.Optimizer = Optim.BFGS(),
  opts::GenoudOptions = GenoudOptions(),
  op::GenoudOperators = GenoudOperators())

  ## Check
  checkdomain(domains, initial_x)

  ## Number of parameters
  k = length(domains)
  ## Get splits for operator application
  idx = splits(op, sizepop, k)
  ## Options
  f_tol = opts.f_tol
  g_tol = opts.g_tol
  max_generations = opts.max_generations
  hard_generation_limit = opts.hard_generation_limit
  wait_generations = opts.wait_generations
  optim_burnin = opts.optim_burnin
  print_level = opts.print_level
  boundary_enforcement = opts.boundary_enforcement
  initial_selection = opts.initial_selection
  check_gradient = opts.check_gradient
  bmix = opts.bmix::Float64
  pmix = opts.pmix::Float64
  ## Set the solver
  σ = sense == :Min ? 1  : -1
  func(x) = σ*fcn(x)
  function grad!(x, stor)
    gr!(x, stor)
    scale!(stor, σ)
    stor
  end

  # Initialize population
  population  = initialpopulation(domains, sizepop)  ## (k × sizepop)
  offspring   = similar(population)                  ##
  fitness     = zeros(sizepop)                       ## Need to experiment with pmap
  smplidx     = collect(1:sizepop)
  #=
  ## Initialize storages
  =#
  ftols = [0.0]
  gtol  = 0.0
  grx   = Array{Float64}(k)
  #=
  ## Print problem info
  =#
  print_level > 0 && print_problem_info(op, opts, sizepop, domains, sense)
  #=
  ## Set generation
  =#
  generation = 0
  #=
  ## Calculate initial fitness value
  =#
  for i in 1:sizepop
    fitness[i] = func(population[:,i])
  end
  fitidx = sortperm(fitness, rev = false)
  #=
  ## Resample population
  =#
  smplprob = Array{Float64}(sizepop)
  if initial_selection
    smplprob .= pmix.*((1-pmix).^(fitidx-1))
    sample!(1:sizepop, WeightVec(smplprob), smplidx)
    population = population[:, smplidx]
    fitness = fitness[smplidx]
    DEBUG && Base.show(_describe(smplprob))
    copy!(fitidx, 1:sizepop)
  end
  #=
  ## Best individual
  =#
  bestfitns =  fitness[fitidx[1]]
  bestindiv =  population[:, fitidx[1]]
  xvals = [bestindiv]
  fvals = [bestfitns]

  if check_gradient
    gr!(bestindiv, grx)
    gtols = [sumabs(grx)]
  else
    gtols = Array{Float64}(0)
  end

  DEBUG && println("\n Best individual")
  DEBUG && Base.show(bestindiv)
  DEBUG && println("Best fitness")
  DEBUG && show(bestfitns)
  DEBUG && check_gradient && println("Gradient")
  DEBUG && check_gradient && show(bestfitns)

  print_level >0 && print_generation_info(generation, fitness, population, bestindiv, bestfitns, print_level, σ)

  while true
    #=
    Mutate population
    =#
    population = mutation(population, fitness, smplidx, fitidx, idx,
                          domains, generation, max_generations,
                          boundary_enforcement, bmix)
    #=
    Calculate fitness
    =#
    for i in 1:sizepop
      fitness[i] = func(population[:,i])
    end
    #=
    Rank fitness and get best individuals
    =#
    sortperm!(fitidx, fitness, rev = false)
    minidx = fitidx[1]
    maxidx = fitidx[end]
    current_bestindiv =  population[:, minidx]
    current_bestfitns =  fitness[minidx]

    generation += 1
    #=
    Apply solver to best individual of mutated population
    =#
    if optim_burnin < generation + 1 && optimize_best
      try
      DEBUG && println("Running BFGS on")
      DEBUG && show(current_bestindiv)
      if boundary_enforcement
          out = Optim.optimize(DifferentiableFunction(func, grad!),
                               vec(current_bestindiv), domains.m[:,1], domains.m[:,2],
                               Fminbox(), optimizer = BFGS, iterations = 500)
      else
        out = Optim.optimize(func, grad!, vec(current_bestindiv), optimizer)
      end
      DEBUG && println("SOLVER OUTPUT")
      DEBUG && show(out)

      population[:,maxidx-1] = out.minimum
      fitness[maxidx-1] = out.f_minimum
      if out.f_minimum < current_bestfitns
        current_bestindiv = copy(out.minimum)
        current_bestfitns = out.f_minimum
      end
      catch
        DEBUG && println("SOLVER FAILED")
        print_level > 0 && print_with_color(:red, "Solver on best individual failed\n")
      end
    end

    #=
    Carry over best of previous generation
    =#
    population[:, 1] = bestindiv
    fitness[1] = bestfitns
    #=
    Print info
    =#
    print_level > 0 && print_generation_info(generation, fitness, population, bestindiv, bestfitns, print_level, σ)

    #=
    Calculate tolarances
    =#
    ftol = abs(current_bestfitns - bestfitns)/abs(bestfitns)
    if check_gradient
      gr!(current_bestindiv, grx)
      gtol = sumabs(grx)
      push!(gtols, gtol)
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
    if ftol <= f_tol && gtol <= g_tol
      if length(fvals) >= wait_generations && fvals[end-wait_generations+2:end] ≈ fvals[end-wait_generations+1]
          break
      end
    end

    if generation > max_generations
      break
    elseif generation == max_generations
      if hard_generation_limit
        warn("Number of max generation limit reached, but the fitness value is still changing")
        break
      else
        print_level > 1 && warn("Number of max generation limit reached, increasing max number of generation from "*string(max_generations)*" to "*string(max_generations+10))
        generation += 10
      end
    end

    ## Resample population
    sortperm!(fitidx, fitness, rev = false)
    smplprob .= pmix.*((1-pmix).^(fitidx-1))
    # broadcast!(-, fitidx, fitidx, 1)
    # broadcast!(^, smplprob, 1-pmix, fitidx)
    # broadcast!(*, smplprob, smplprob, pmix)
    sample!(1:sizepop, WeightVec(smplprob), smplidx)
    #sample!(1:sizepop,  smplidx)
    population = population[:, smplidx]
    #population[:, 1] = bestindiv
  end

  bestgen = findmin(fvals)[2]

  GenoudOutput(bestindiv,
               σ*bestfitns,
               σ.*fvals,
               xvals,
               ftols,
               gtols,
               grx,
               bestgen,
               sense,
               domains,
               op,
               opts)

end
#
function converged(r::GenoudOutput)
  f_converged(r) || r.opts.check_gradient && last(r.gtols) <= r.opts.g_tol
end

f_converged(r::GenoudOutput) = last(r.ftols) <= r.opts.f_tol
g_converged(r::GenoudOutput) = last(r.gtols) <= r.opts.g_tol

f_tol(r::GenoudOutput) = r.opts.f_tol
g_tol(r::GenoudOutput) = r.opts.f_tol

generations(r::GenoudOutput) = length(r.ftols) - 1
#
function Base.show(io::IO, r::GenoudOutput)
  @printf io "Results of Genoud Optimization Algorithm\n"
  optimum = r.bestindiv
  if r.sense == :Min
    if length(join(optimum, ",")) < 40
          @printf io " * Minimizer: [%s]\n" join(optimum, ",")
      else
          @printf io " * Minimizer: [%s, ...]\n" join(optimum[1:2], ",")
      end
      @printf io " * Minimum: %e\n" r.bestfitns
  else
    if length(join(optimum, ",")) < 40
        @printf io " * Maximizer: [%s]\n" join(optimum, ",")
    else
        @printf io " * Maximizer: [%s, ...]\n" join(optimum[1:2], ",")
    end
    @printf io " * Maximum: %e\n" r.bestfitns
  end

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

end  #module
