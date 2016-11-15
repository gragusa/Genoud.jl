module Genud

using MathProgBase
using OptimMPB
using Parameters
using StatsFuns
using StatsBase
# package code goes here

const FLOAT = eltype(1.0)

const XNM = ["x₁", "x₂", "x₃", "x₄", "x₅", "x₆", "x₇", "x₈ ", "x₉", "x₁₀",
            "x₁₁", "x₁₂", "x₁₃", "x₁₄", "x₁₅", "x₁₆", "x₁₇", "x₁₈ ", "x₁₉", "x₂₀",
            "x₂₁", "x₂₂", "x₂₃", "x₂₄", "x₂₅", "x₂₆", "x₂₇", "x₂₈ ", "x₂₉", "x₃₀",
            "x₃₁", "x₃₂", "x₃₃", "x₃₄", "x₃₅", "x₃₆", "x₃₇", "x₃₈ ", "x₃₉", "x₄₀",
            "x₄₁", "x₄₂", "x₄₃", "x₄₄", "x₄₅", "x₄₆", "x₄₇", "x₄₈ ", "x₄₉", "x₅₀"]

immutable Domain{T}
  m::Matrix{T}
end

Base.length(d::Domain) = size(d.m, 1)

function Domain{T}(x::Array{T, 1})
  n = length(x)
  Domain([x.-10*ones(T, n) x.+10*ones(T, n)])
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
    f_tol::Float64 = 0.001
    boundary_enforcement::Int64 = 0
    data_type_int::Bool = false
    print_level::Int64 = 2
    optim_burnin::Int64 = 0
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
  x[j] = d.m[j,r]
  x
end

function nonuniform_mut!(x, d::Domain, generation, max_generation)
  B = .5
  k = length(d)
  j = rand(1:k)
  p = rand()*(1-generation/(max_generation+1))^B
  r = rand(1:2)
  x[j] = (1-p)*x[j]+p*d.m[j,r]
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

function whole_mut!(x, d::Domain, generation, max_generation)
  B = .5
  k = length(d)
  #z = Array{FLOAT}(k)
  for j in 1:k
    p = rand()*(1-generation/max_generation)^B
    r = rand(1:2)
    x[j] = (1-p)*x[j]+p*d.m[j,r]
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
    if z ∈ d || attempts > 10
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
    u = rand(size(X, 2))
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

function claw(xx)
 x = xx[1]
 (0.46*(normpdf(-1.0,2.0/3.0, x) + normpdf(1.0,2.0/3.0, x)) +
 (1.0/300.0)*(normpdf(-0.5,.01, x) + normpdf(-1.0,.01, x) + normpdf(-1.5,.01, x)) +
 (7.0/300.0)*(normpdf(0.5,.07, x) + normpdf(1.0,.07, x) + normpdf(1.5,.07, x)))
end

function mutation(population, fitness, smplidx, fitidx, idx, domains::Domain, generation, max_generation)
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
    nonuniform_mut!(view(offspring, :, i), domains, generation, max_generation)
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
    whole_mut!(view(offspring, :, i), domains, generation, max_generation)
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


# function genoud(fcn, initial_x; gr! = identity, optimize_best = true,
#   method::Optim.Optimizer = Optim.BFGS(),
#   sense::Symbol = :Min, sizepop = 1000,
#   domains::Domain = Domain(initial_x),
#   opts::GenoudOptions = GenoudOptions(),
#   op::GenoudOperators = GenoudOperators())

function genoud(fcn, initial_x, gr! = identity, sizepop::Int64 = 1000,sense::Symbol = :Min,
  optimize_best = true,
  method::Optim.Optimizer = Optim.BFGS(),

  domains::Domain = Domain(initial_x),
  opts::GenoudOptions = GenoudOptions(),
  op::GenoudOperators = GenoudOperators())


  ## Check
  checkdomain(domains, initial_x)

  ## Parameters
  σ = sense == :Min ? 1 : -1

  k = length(domains)

  func(x) = σ*fcn(x)

  ## Calculate operator rate
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
  idx = cumsum(op_split)[1:end-1]

  ## Options
  f_tol = opts.f_tol
  max_generations = opts.max_generations
  hard_generation_limit = opts.hard_generation_limit
  wait_generations = opts.wait_generations
  optim_burnin = opts.optim_burnin
  # Initialize population
  population  = initialpopulation(domains, sizepop)  ## (k × sizepop)
  offspring   = similar(population)                  ##
  fitness     = zeros(sizepop)
  smplidx     = collect(1:sizepop)

  print_problem_info(op, opts, sizepop, domains, sense)

  ## Calculate initial fitness value
  for i in 1:sizepop
    fitness[i] = func(population[i])
  end
  fitidx = sortperm(fitness, rev = false)
  minidx = fitidx[1]
  bestindiv = population[:, minidx]
  bestfitns = fitness[minidx]

  fittols = [0.0]
  fitvals = [bestfitns]
  indvals = Array{Array{Float64, 1}, 1}(0)
  push!(indvals, bestindiv)

  generation = 0
  print_generation_info(generation, fitness, population, bestindiv, bestfitns)

  while true
    ## Selection step
    ## Select individuals based on the value of the fitness value
    ## with probability proportional to Q(1-Q)^r, where r is the rank
    #selectionprob = exp(-fitness/2)./Base.sum(exp(-fitness/2),1)
    #sample!(1:sizepop, WeightVec(selectionprob), smplidx)
    sample!(1:sizepop, smplidx)
    #selectionprob = 1.


    population = population[:, smplidx]
    population = mutation(population, fitness, smplidx, fitidx, idx,
                          domains, generation, max_generations)
    population[:, end] = bestindiv

    for i in 1:sizepop
      fitness[i] = func(population[i])
    end
    fitidx = sortperm(fitness, rev = false)
    minidx = fitidx[1]
    maxidx = fitidx[end]
    current_bestindiv = population[:, minidx]
    current_bestfitns = fitness[minidx]

    generation += 1
    ## Apply BFGS to best individual
    if optim_burnin < generation
      out = Optim.optimize(func, gr!, vec(current_bestindiv), BFGS())
      population[:,maxidx] = out.minimum
      fitness[maxidx]    = out.f_minimum
      if out.f_minimum < current_bestfitns
        bestindiv = copy(out.minimum)
        bestfitns = out.f_minimum
      end
    end

    print_generation_info(generation, fitness, population, bestindiv, bestfitns)

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
        generation += 10
      end
      continue
    else
      ## Below tolerance
      if length(fitvals) >= wait_generations && all(fitvals[end-wait_generations+2:end] .== fitvals[end-wait_generations+1])
        break
      end
    end
  end


  return bestindiv, bestfitns, fitvals, indvals

end


end  #module
