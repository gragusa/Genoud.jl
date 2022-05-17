using Genoud
using Ipopt


nruns = 51

fstar = zeros(23, nruns, 4)
solus  = []
times = zeros(23, nruns, 4)
parms = ([30, 5000], [30, 10000], [100, 5000], [100, 10000])

opt    = Genoud.Options(pmix = .5, print_level = 1)
solver = BFGS()
srand(1)

p, i, j = 1,1,1

for p = 1:4
    mgens = parms[p][1]
    spop  = parms[p][2]
    for i in 1:length(funs)
        f      = funs[i]
        fit    = (x,p) -> f[:f](x)
        bounds = copy(f[:domain])
        domain = Genoud.Domain(bounds)
        x0     = bounds[:,1] .+ rand(size(bounds, 1)).*(bounds[:,2] .- bounds[:,1])
        
        ad = f[:differentiable] ? GalacticOptim.AutoForwardDiff() : SciMLBase.NoAD()
        
        prob = OptimizationProblem(OptimizationFunction(fit, ad), x0; lb = bounds[:,1], ub = bounds[:,2])

        for j in 1:nruns
            printstyled("Solving problem ", i, " out of 23 -> "; color = :cyan)
            printstyled(f[:fname], "\n"; color = :cyan)
            times[i, j, p] = @elapsed out = Genoud.genoud(prob; popsize = spop, solver = solver, opt = opt)
            fstar[i, j, p] = Genoud.bestfitns(out)
            push!(solus, Genoud.bestindiv(out))
        end
    end
end
