using Genoud
using Ipopt

nruns = 51

fstar = zeros(23, nruns, 4)
solus  = []
times = zeros(23, nruns, 4)
parms = ([30, 5000], [30, 10000], [100, 5000], [100, 10000])

opt    = Genoud.Options(f_tol = 1e-06, max_generations = 50, pmix = .5, print_level = 0)
solver = Ipopt.IpoptSolver(print_level=0)
srand(1)

p, j, i = 1,1,1

#for p = 1:4
  mgens = parms[p][1]
  spop  = parms[p][2]
#  for i in 1:length(funs)
    f      = funs[i]
    fit    = deepcopy(f[:f])
    bounds = copy(f[:domain])
    domain = Genoud.Domain(bounds)
    x0     = bounds[:,1] + rand(size(bounds, 1)).*(bounds[:,2] - bounds[:1])
    
    if f[:differentiable]
      d = OnceDifferentiable(fit, x0)
    else
      d = NonDifferentiable(fit, x0)
    end
  
    print_with_color(:cyan, "Solving problem ", i, " out of 23 -> ")
    print_with_color(:cyan, f[:fname], "\n")
    times[i, j, p] = @elapsed out = Genoud.genoud(d, spop, domain, solver = solver, opt = opt)
    fstar[i, j, p] = Genoud.bestfitns(out)
    push!(solus, Genoud.bestindiv(out))
  #end
#end
