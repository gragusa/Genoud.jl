using Genoud
using StatsFuns
using Base.Test

# write your own tests here
function claw(xx)
 x = xx[1]
 (0.46*(normpdf(-1.0,2.0/3.0, x) + normpdf(1.0,2.0/3.0, x)) +
 (1.0/300.0)*(normpdf(-0.5,.01, x) + normpdf(-1.0,.01, x) + normpdf(-1.5,.01, x)) +
 (7.0/300.0)*(normpdf(0.5,.07, x) + normpdf(1.0,.07, x) + normpdf(1.5,.07, x)))
end

opt = Genoud.Options(print_level = 0)
out1 = Genoud.genoud(OnceDifferentiable(claw, [0.0]), 5000, [-10.], [10.], sense = :Max)   
out2 = Genoud.genoud(NonDifferentiable(claw, [0.0]), 5000, [-10.], [10.], sense = :Max)   
out3 = Genoud.genoud(OnceDifferentiable(claw, [0.0]), 5000, [-10.], [10.], sense = :Max, max_generations = 10, wait_generations = 5)   
out4 = Genoud.genoud(NonDifferentiable(claw, [0.0]), 5000, [-10.], [10.], sense = :Max, max_generations = 12, wait_generations = 5)   


@test Genoud.bestindiv(out1) ≈ Genoud.bestindiv(out2)
@test Genoud.bestindiv(out2) ≈ Genoud.bestindiv(out3)
@test Genoud.bestindiv(out3) ≈ Genoud.bestindiv(out4)

@test isa(Genoud.bestgen(out1), Int64)
@test isa(Genoud.bestgen(out2), Int64)

@test Genoud.bestfitns(out1) ≈ Genoud.bestfitns(out2)
@test Genoud.bestfitns(out2) ≈ Genoud.bestfitns(out3)
@test Genoud.bestfitns(out3) ≈ Genoud.bestfitns(out4)
