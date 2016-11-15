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
            "X₄₁", "X₄₂", "X₄₃", "X₄₄", "X₄₅", "X₄₆", "X₄₇", "X₄₈ ", "X₄₉", "X₅₀"]

immutable Domain{T}
  m::Matrix{T}
  p::Matrix{T}
end

Base.length(d::Domain) = size(d.m, 1)

function Domain{T}(x::Array{T, 1})
  n = length(x)
  Domain([x.-10*ones(T, n) x.+10*ones(T, n)], [√eps(T)*ones(T, n) -√eps(T)*ones(T, n)])
end

function Domain{T}(x::Array{T, 2})
  n = length(x)
  Domain(x, [√eps(T)*ones(T, n) -√eps(T)*ones(T, n)])
end



@with_kw type GenoudOperators
    cloning::Int64 = 50
    uniform_mut::Int64 = 50
    boundary_mut::Int64 = 50
    nonuniform_mut::Int64 = 50
    whole_mut::Int64 = 50
    polytope_cross::Int64 = 50
    simple_cross::Int64 = 50
    heuristic_cross::Int64 = 50
    localmin_cross::Int64 = 0
end

@with_kw type GenoudOptions
    max_generations::Int64 = 100
    wait_generations::Int64 = 10
    hard_generation_limit::Bool = true
    f_tol::FLOAT = 0.001
    boundary_enforcement::Bool = false
    data_type_int::Bool = false
    print_level::Int64 = 2
    optim_burnin::Int64 = 0
    hessian::Bool = false
    pmix::FLOAT = 0.8
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
  rate = Array{FLOAT}(length(fnm))
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


isinbound(x::FLOAT, j, d::Domain) = x <= d.m[j,2] && x >= d.m[j,1]



cloning(x) = x

function uniform_mut!(x, d::Domain)
  k = length(d)
  j = rand(1:k)
  x[j] = _rand(d.m[j,1], d.m[j,2])
  x
end

function boundary_mut!(x, d::Domain)
  k = length(d)
  j = rand(1:k)
  r = rand(1:2)
  x[j] = d.m[j,r] + d.p[j,r]
  x
end

function nonuniform_mut!(x, d::Domain, generation, max_generation, boundary_enforcement)
  B = .5
  k = length(d)
  j = rand(1:k)
  p = rand()*(1-generation/(max_generation+1))^B
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
  k = length(d)
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
  k = length(d)
  p = 0.5
  z = zeros(k)
  for s in 1:k, j in 1:2
    z[s] += 0.5*x[s, j]
  end
  z
end

function whole_mut!(x, d::Domain, generation, max_generation, boundary_enforcement)
  B = .5
  k = length(d)
  ## TODO: Check for boundary condition
  for j in 1:k
    p = rand()*(1-generation/max_generation)^B
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
  k = length(d)
  z = Array{FLOAT}(k)
  attempts = 0
  while true
    p = rand()
    for s = 1:k
      z[s] = p*(x[s,1]-x[s,2]) + x[s,1]
    end
    if z ∈ d
      break
    end
    if attempts > 10
      j = rand(1:2)
      for s = 1:k
        z[s] = x[s, j]
      end
      break
    end
    attempts += 1
  end
  z
end


function Base.Random.rand(d::Domain)
  k = length(d)
  u = Array{FLOAT}(k)
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
    X = Array{FLOAT}(k, n)
    rand!(X, d)
    X
end

function mutation(population, fitness, smplidx, fitidx, idx, domains::Domain, generation, max_generation, boundary_enforcement)
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
    nonuniform_mut!(view(offspring, :, i), domains, generation, max_generation, boundary_enforcement)
  end
  ## Polytope crossover
  for i in idx[4]+1:idx[5]
    offspring[:,i] = polytope_cross(view(population, :, i:i+max(2,k)-1), domains)
  end
  ## Simple Cross
  for i in idx[5]+1:idx[6]-1
    offspring[:,i] = simple_cross(view(population, :, i:i+1), domains)
  end
  ## Whole mutation
  for i in idx[6]:idx[7]
    for j = 1:k
      offspring[j,i] = population[j,i]
    end
    whole_mut!(view(offspring, :, i), domains, generation, max_generation, boundary_enforcement)
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

function print_generation_info(generation, fitness, population, bestindiv, bestfitns)
  print_with_color(:blue, "Generation ", string(generation)*"\n")
  print_with_color(:cyan, "Fitness (best)...:   "*string(bestfitns)*"\n")
                  println("  mean...........:   ", mean(fitness))
                  println("  variance.......:   ", var(fitness))
                  println("  unique.........:   ", length(unique(fitness)))
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

function genoud(fcn, initial_x; sizepop = 1000, sense::Symbol = :Min,
  domains::Domain = Domain(initial_x), optimize_best = true, gr! = identity,
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
  max_generations = opts.max_generations
  hard_generation_limit = opts.hard_generation_limit
  wait_generations = opts.wait_generations
  optim_burnin = opts.optim_burnin
  print_level = opts.print_level
  boundary_enforcement = opts.boundary_enforcement
  pmix = opts.pmix
  ## Set the solver
  σ = sense == :Min ? 1 : -1
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

  print_level > 0 && print_problem_info(op, opts, sizepop, domains, sense)

  ## Calculate initial fitness value
  for i in 1:sizepop
    fitness[i] = func(population[:,i])
  end
  fitidx = sortperm(fitness, rev = false)
  minidx = fitidx[1]
  bestindiv = population[:, minidx]
  bestfitns = fitness[minidx]

  fittols = [0.0]
  fitvals = [bestfitns]
  indvals = Array{Array{FLOAT, 1}, 1}(0)
  push!(indvals, bestindiv)

  generation = 0
  print_generation_info(generation, fitness, population, bestindiv, bestfitns)

  while true
    ## Selection step
    ## Select individuals based on the value of the fitness value
    ## with probability proportional to Q(1-Q)^r, where r is the rank
    #selectionprob = exp(-fitness/2)./Base.sum(exp(-fitness/2),1)
    #sample!(1:sizepop, WeightVec(selectionprob), smplidx)
    #sample!(1:sizepop, smplidx)
    #population = population[:, smplidx]

    population = mutation(population, fitness, smplidx, fitidx, idx,
                          domains, generation, max_generations, boundary_enforcement)
    ## Make sure the best individuals from previus generation survive
    population[:, 1] = bestindiv

    for i in 1:sizepop
      fitness[i] = func(population[:,i])
    end
    fitidx = sortperm(fitness, rev = false)
    minidx = fitidx[1]
    maxidx = fitidx[end]
    current_bestindiv = population[:, minidx]
    current_bestfitns = fitness[minidx]

    generation += 1
    ## Apply BFGS to best individual
    if optim_burnin < generation + 1 && optimize_best
      if boundary_enforcement
          out = Optim.optimize(DifferentiableFunction(func, grad!),
                               vec(current_bestindiv), domains.m[:,1], domains.m[:,2],
                               Fminbox(), optimizer = typeof(optimizer))
      else
        out = Optim.optimize(func, grad!, vec(current_bestindiv), optimizer)
      end
      population[:,maxidx-1] = out.minimum
      fitness[maxidx-1] = out.f_minimum
      if out.f_minimum < current_bestfitns
        current_bestindiv = copy(out.minimum)
        current_bestfitns = out.f_minimum
      end
    end

    print_level > 0 && print_generation_info(generation, fitness, population, bestindiv, bestfitns)

    fittol = abs(current_bestfitns - bestfitns)
    ## Store tolerance level
    push!(fittols, fittol)
    push!(fitvals, bestfitns)
    push!(indvals, bestindiv)

    bestfitns = copy(current_bestfitns)
    bestindiv = copy(current_bestindiv)

    ## Check exit condition
    if fittol >= f_tol
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
    else
      ## Below tolerance
      if length(fitvals) >= wait_generations && fitvals[end-wait_generations+2:end] ≈ fitvals[end-wait_generations+1]
        break
      end
    end
    ## Resample population
    fitidx = sortperm(fitness, rev = false)
    smplprob = pmix.^fitidx
    sample!(1:sizepop, WeightVec(smplprob), smplidx)
    population = population[:, smplidx]
    population[:, 1] = bestindiv
  end
  return bestindiv, bestfitns, fitvals, indvals
end

function Base.isapprox(x::Array{FLOAT, 1}, y::FLOAT)
  for j in eachindex(x)
    isapprox(x[j], y) || return false
  end
  true
end

end  #module
