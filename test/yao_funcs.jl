# 23 functions as defined in
# Xin Yao, Yong Liu, and Guangming Lin
# Evolutionary Programming Made Faster
# IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 3, NO. 2, JULY 1999


a1 = [-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0,
      16, 32, -32, -16, 0, 16, 32]
a2 = [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0,
     16, 16, 16, 16, 16, 32, 32, 32, 32, 32]

const a = [a1 a2]'

const α = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
const β = 1./[0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]


const ι = [1, 1.2, 3, 3.2]

const γ = [3.0 10 30;
           0.1 10 35;
           3.0 10 30;
           0.1 10 35]

const δ = [0.368900 0.1170 0.2673;
           0.469900 0.4387 0.7470;
           0.109100 0.8732 0.5547;
           0.038150 0.5743 0.8828]


const η = [10.00  3.0 17.00  3.5  1.7  8;
            0.05 10.0 17.00  0.1  8.0 14;
            3.00  3.5  1.70 10.0 17.0  8;
           17.00  8.0  0.05 10.0  0.1 14]

const ϕ = [0.1312 0.1696 0.5569 0.0124 0.8283 0.5886;
           0.2329 0.4135 0.8307 0.3736 0.1004 0.9991;
           0.2348 0.1415 0.3522 0.2883 0.3047 0.6650;
           0.4047 0.8828 0.8732 0.5743 0.1091 0.0381]


const σ = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
const ρ = [4 4.0 4 4.0;
           1 1.0 1 1.0;
           8 8.0 8 8.0;
           6 6.0 6 6.0;
           3 7.0 3 7.0;
           2 9.0 2 9.0;
           5 5.0 3 3.0;
           8 1.0 8 1.0;
           6 2.0 6 2.0;
           7 3.6 7 3.6]'

function shekel(x,m)
  a = map(i -> 1/(norm(x-ρ[:,i])^2 + σ[i]), 1:m)
  -sum(a)
end

## Functions definition

differentiable = (true, false, true, false, true, false, true, true, true, true,
                  [true for j = 1:13]...)

testfuns = (  x -> sumabs2(x),
              x -> sumabs(x) + prod(abs(x)),
              x -> begin
                    k = length(x)
                    sum(map(i -> sumabs2(x[1:i]), 1:k))
                  end,
              x -> maximum(abs(x)),
              x -> begin
                    k = length(x) - 1
                    sum(map(i -> 100*(x[i+1]-x[i]^2)^2 + (x[i]-1)^2, 1:k))
                  end,
              x -> sum(floor(x+0.5).^2),
              x -> begin
                    k = length(x)
                    sum(collect(1:k).*x.^4)
                    #sum(map(i -> i*x[i]^4, 1:k)) + rand()
                   end,
              x -> begin
                    -sum(x.*sin(sqrt(sqrt(x.^2))))
                   end,
              x -> begin
                    k = length(x)
                    -sum(x.^2 - 10.*cos(2*π*x) + 10)
                   end,
              x -> begin
                    k = length(x)
                    -20*exp(-0.2*√(sumabs2(x)/k)) - exp(mean(cos(2*π.*x))) + 20 + e
                   end,
              x -> begin
                    k = length(x)
                    sumabs2(x)/1000 - prod(cos(x./sqrt(collect(1:k)))) + 1.0
                   end,
              x -> begin
                      k = length(x)
                      y = 1 .+ (x .+ 1)./4
                      π/k *(10*sin(π*y[1])^2 +
                      sum(map(i-> (y[i]-1)^2*(1+10*sin(π*y[i+1])^2), 1:k-1)) +
                      (y[k]-1)^2) + sum(map(z -> ifelse(z>10, 100*(z-10)^4,
                                                    ifelse(z < -10, 100*(-z-10)^4, 0)), x))
                  end,
              x -> begin
                    k = length(x)
                    0.1 *(sin(π*3*x[1])^2 +
                        sum(map(i -> (x[i]-1)^2*(1+sin(3*π*x[i+1])^2), 1:k-1)) +
                        (x[k]-1)^2)*(1+sin(3*π*x[k])^2) +
                        sum(map(z -> ifelse(z>5, 100*(z-5)^4,
                                                    ifelse(z>=-5, 0, 100*(-z-5)^4)), x))
                   end,
              x -> begin
                      1/(1/500 + sum(1./(collect(1:25) + sum((x.-a).^6, 1)')))
                   end,
              x -> sum(map(i -> (α[i] - x[1]*(β[i]^2+β[i]*x[2])/(β[i]^2+β[i]*x[3]+x[4]))^2, 1:11)),
              x -> 4*(x[1]^2 - x[2]^2 + x[2]^4) - 2.1*x[1]^4 + x[1]^6/3 + x[1]*x[2],
              x -> (x[2] - 5.1/(4*π^2)*x[1]^2 + 5/π*x[1] - 6)^2 + 10*(1-1/(8*π))*cos(x[1]) + 10,
              x -> (1+(x[1]+x[2]+1)^2*(19-14*x[1]+3*x[1]^2-14*x[2]+6*x[1]*x[2]+3*x[2]^2))*
                   (30+(2*x[1]-3*x[2])^2*(18-32*x[1]+12*x[1]^2+48*x[2]-36*x[1]*x[2]+27*x[2]^2)),
              x -> -sum(map(i -> ι[i]*exp(-sum(map(j -> γ[i,j]*(x[j]-δ[i,j])^2, 1:3))), 1:4)),
              x -> -sum(map(i -> ι[i]*exp(-sum(map(j -> η[i,j]*(x[j]-ϕ[i,j])^2, 1:6))), 1:4)),
              x -> shekel(x, 5),
              x -> shekel(x, 7),
              x -> shekel(x, 10))

mins_testfuns  = (0, 0, 0, 0, 0, 0, 0, -12569.5, 0, 0, 0, 0, 0, 1,  0.0003075,
                  -1.0316285,  0.398, 3, -3.86, -3.32, -10.1532, -10.4029, -10.5364)

xmin_testfuns  = (zeros(30), zeros(30), zeros(30), zeros(30), ones(30),
                  zeros(30), zeros(30), [420.9789 for j = 1:30],
                  zeros(30), zeros(30), zeros(30), -ones(30), ones(30),
                  [-31.98, -31.98], [0.1928, 0.1908, 0.1231, 0.1358],
                  ([0.08983, -0.7126], [-0.8983, 0.7126]),
                  ([-3.142, 2.275], [3.142, 2.275], [9.425, 2.475]),
                  [0, -1], [0.114, 0.556, 0.852],
                  [0.201, 0.150, 0.477, 0.275, 0.311, 0.657],
                  [4,4,4,4.], [4,4,4,4.], [4,4,4,4.])



nargs_testfuns = [repeat([30], inner = 13); 2; 4; 2; 2; 2; 4; 6; 4; 4; 4]

bnds_testfuns = ((-100.,100), (-10.,10), (-100.,100), (-100.,100),
                (-30.,30), (-100., 100), (-1.28,1.28),  (-500.,500),
                (-5.12,5.12), (-32.,32), (-600.,600), (-50.,50),
                (-50.,50), (-65.536,65.536), (-5.,5), (-5.,5),
                (-5,10.,0.,15), (-2.,2), (0.,1),  (0.,1),
                (0.,10), (0.,10), (0.,10))

bnds = map(i -> reshape(repeat([bnds_testfuns[i]...], inner = nargs_testfuns[i]), nargs_testfuns[i] + (i==17)*2, 2), 1:length(nargs_testfuns))
## Fix bound for testfuns 17 (different bounds on different parms)
bnds[17] = [-5 10; 0 15.]

names_testfuns = ("Sphere model",
                  "Schwefel’s Problem 2.22",
                  "Schwefel’s Problem 1.2",
                  "Schwefel’s Problem 2.21",
                  "Generalized Rosenbrock’s Function",
                  "Step Function",
                  "Quartic Function",
                  "Generalized Schwefel’s Problem 2.26",
                  "Generalized Rastrigin’s Function",
                  "Ackley’s Function",
                  "Generalized Griewank Function",
                  "Generalized Penalized Functions (10,100,4)",
                  "Generalized Penalized Functions (5,100,4)",
                  "Shekel’s Foxholes Function",
                  "Kowalik’s Function",
                  "Six-Hump Camel-Back Function",
                  "Branin Function",
                  "Goldstein-Price Function",
                  "Hartman’s Family (n=3)",
                  "Hartman’s Family (n=3)",
                  "Shekel’s Family (m=5)",
                  "Shekel’s Family (m=7)",
                  "Shekel’s Family (m=10)")

for j in 1:23
  print_with_color(:blue, string(j)*" "*names_testfuns[j]*"\n")
  x0 = xmin_testfuns[j]
  f0 = mins_testfuns[j]
  if isa(x0, Tuple)
    println("Function with multiple minima")
    print_with_color(:bold, "Minimum: "*string(f0)*"\n")
    for h in 1:length(x0)
      tx0 = x0[h]
      lx0 = length(tx0)
      if lx0 > 5
        @printf "Minimizer: [%s,...,%s] -> f(x0) = %s \n" join(tx0[1:2], ",") join(tx0[lx0], ",") testfuns[j](tx0)
      else
        @printf "Minimizer: [%s,...,%s] -> f(x0) = %s \n" join(tx0[1:2], ",") join(tx0, ",") testfuns[j](tx0)
      end
      isapprox(1+testfuns[j](tx0), 1+f0, rtol = 0.01) && print_with_color(:bold, "Correct"*"\n")
    end
  else
    print_with_color(:bold, "Minimum: "*string(f0)*"\n")
    lx0 = length(x0)
    if lx0 > 5
      @printf "Minimizer: [%s,...,%s] -> f(x0) = %s\n" join(x0[1:2], ",") join(x0[lx0], ",") testfuns[j](x0)
    else
      @printf "Minimizer: [%s,...,%s] -> f(x0) = %s\n" join(x0[1:2], ",") join(x0, ",") testfuns[j](x0)
    end
    isapprox(1+testfuns[j](x0), 1+f0, rtol = 0.01) && print_with_color(:bold, "Correct"*"\n")
  end
end



using Calculus
using ForwardDiff
using Genoud
sols = []

for i = 1:23
  println("Solveing problem ", i, " out of 23")
  x0 = xmin_testfuns[i]
  isa(x0, Tuple) && (x0 = x0[1])
  k = length(x0)
  cs = min(10, k)
  if differentiable[i]
    out = Genoud.genoud(testfuns[i], zeros(nargs_testfuns[i]),
    sizepop = 5000,
    gr! = (x, store) -> ForwardDiff.gradient!(store, testfuns[i], x, Chunk{cs}()),
    opts = Genoud.GenoudOptions(f_tol = 1e-06,
    max_generations = 100,
    pmix = .5, boundary_enforcement = true, print_level = 1),
    sense = :Min, domains = Genoud.Domain(bnds[i]));
  else
    out = Genoud.genoud(testfuns[i], zeros(nargs_testfuns[i]), sizepop = 5000,
    gr! = (x, store) -> store[:] = Calculus.gradient(testfuns[i], x),
    opts = Genoud.GenoudOptions(f_tol = 1e-06, max_generations = 100,
    pmix = .5, boundary_enforcement = true, print_level = 1),
    sense = :Min,
    domains = Genoud.Domain(bnds[i]));
  end
  push!(sols, out)
end
