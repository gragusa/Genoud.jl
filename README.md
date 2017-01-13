# Genoud.jl
[![Build Status](https://travis-ci.org/gragusa/Genoud.jl.svg?branch=master)](https://travis-ci.org/gragusa/Genoud.jl)
[![Coverage Status](https://coveralls.io/repos/gragusa/Genoud.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/gragusa/Genoud.jl?branch=master)
[![codecov.io](http://codecov.io/github/gragusa/Genoud.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/Genoud.jl?branch=master)

GENetic Optimization Using Derivative.

```julia
using Genoud

function claw(xx)
 x = xx[1]
 (0.46*(normpdf(-1.0,2.0/3.0, x) + normpdf(1.0,2.0/3.0, x)) +
 (1.0/300.0)*(normpdf(-0.5,.01, x) + normpdf(-1.0,.01, x) + normpdf(-1.5,.01, x)) +
 (7.0/300.0)*(normpdf(0.5,.07, x) + normpdf(1.0,.07, x) + normpdf(1.5,.07, x)))
end

## Not using derivatives
out = Genoud.genoud(claw, [0.0], sizepop = 5000, sense = :Max, optimize_best = false)
## Using derivatives (in this case finite diff derivatives)
out = Genoud.genoud(claw, [0.0], sizepop = 5000, sense = :Max)
## Using derivatives (in this case automatic diff derivatives)
out = Genoud.genoud(claw, [0.0], sizepop = 5000, sense = :Max, optimizer_o = Optim.Options(autodiff = true))

```
