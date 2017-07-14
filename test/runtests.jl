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

gr!(x, stor) = ForwardDiff.gradient!(stor, claw, x)

opt = Genoud.Options(print_level = 0)

srand(1)

@testset "No optimization" begin
    ## No best individual optimization
    out = Genoud.genoud(claw, [0.0], sizepop = 5000, sense = :Max, optimize_best = false, opt = opt)
    @test Genoud.bestfitns(out) ≈ 0.41131232675083473
    @test Genoud.bestindiv(out) ≈ [0.9995032620966703]
end

@testset "Numeric Derivative - Finite Differences" begin
    ## Best individual optimization - Numerical gradient - No boundary enforcement - BFGS
    out = Genoud.genoud(claw, [0.0], sizepop = 5000, sense = :Max, optimize_best = true, opt = opt)
    @test Genoud.bestfitns(out) ≈ 0.41131232675083473
    @test Genoud.bestindiv(out) ≈ [0.9995032620968612]

    ## Best individual optimization - Numerical gradient - No boundary enforcement - BFGS
    out = Genoud.genoud(claw, [0.0], sizepop = 5000, sense = :Max, optimize_best = true, optimizer = Optim.LBFGS(), opt = opt)
    @test Genoud.bestfitns(out) ≈ 0.41131232675083473
    @test Genoud.bestindiv(out) ≈ [0.9995032620968612]


    ## Best individual optimization - Numerical gradient - boundary enforcement
    out = Genoud.genoud(claw, [0.0], sizepop = 5000, sense = :Max, optimize_best = true,
                        opt = Genoud.Options(boundary_enforcement = true), opt = opt)

    @test Genoud.bestfitns(out) ≈ 0.41131232675083473
    @test Genoud.bestindiv(out) ≈ [0.9995032620966703]
end

@testset "Numeric Derivative - Automatic Differences" begin
    ## Best individual optimization - Numerical gradient - No boundary enforcement - BFGS
    out = Genoud.genoud(claw, [0.0], sizepop = 5000, sense = :Max, optimize_best = true,
                        optimizer_o = Optim.Options(autodiff = true), opt = opt)
    @test Genoud.bestfitns(out) ≈ 0.41131232675083473
    @test Genoud.bestindiv(out) ≈ [0.9995032620968612]

    ## Best individual optimization - Numerical gradient - No boundary enforcement - BFGS
    out = Genoud.genoud(claw, [0.0], sizepop = 5000, sense = :Max, optimize_best = true,
                        optimizer = Optim.LBFGS(), optimizer_o = Optim.Options(autodiff = true), opt = opt)
    @test Genoud.bestfitns(out) ≈ 0.41131232675083473
    @test Genoud.bestindiv(out) ≈ [0.9995032620968612]


    ## Best individual optimization - Numerical gradient - boundary enforcement
    out = Genoud.genoud(claw, [0.0], sizepop = 5000, sense = :Max, optimize_best = true,
                        opt = Genoud.Options(boundary_enforcement = true), opt = opt)

    @test Genoud.bestfitns(out) ≈ 0.41131232675083473
    @test Genoud.bestindiv(out) ≈ [0.9995032620966703]
end


@testset "Analytic Derivative" begin
    ## Best individual optimization - Numerical gradient - No boundary enforcement - BFGS
    out = Genoud.genoud(claw, [0.0], sizepop = 5000, sense = :Max, optimize_best = true, gr! = gr!, opt = opt)
    @test Genoud.bestfitns(out) ≈ 0.41131232675083473
    @test Genoud.bestindiv(out) ≈ [0.9995032620968612]

    ## Best individual optimization - Numerical gradient - No boundary enforcement - BFGS
    out = Genoud.genoud(claw, [0.0], sizepop = 5000, sense = :Max, optimize_best = true, gr! = gr!, optimizer = Optim.LBFGS(), opt = opt)
    @test Genoud.bestfitns(out) ≈ 0.41131232675083473
    @test Genoud.bestindiv(out) ≈ [0.9995032620968612]


    ## Best individual optimization - Numerical gradient - boundary enforcement
    out = Genoud.genoud(claw, [0.0], sizepop = 5000, sense = :Max, optimize_best = true, gr! = gr!,
                        opt = Genoud.Options(boundary_enforcement = true), opt = opt)

    @test Genoud.bestfitns(out) ≈ 0.41131232675083473
    @test Genoud.bestindiv(out) ≈ [0.9995032620966703]
end
