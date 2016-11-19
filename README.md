# Genoud.jl
[![Build Status](https://travis-ci.org/gragusa/Genud.jl.svg?branch=master)](https://travis-ci.org/gragusa/Genud.jl)
[![Coverage Status](https://coveralls.io/repos/gragusa/Genud.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/gragusa/Genud.jl?branch=master)
[![codecov.io](http://codecov.io/github/gragusa/Genud.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/Genud.jl?branch=master)

GENetic Optimization Using Derivative.

```julia
using Genoud
using ForwardDiff

function claw(xx)
 x = xx[1]
 (0.46*(normpdf(-1.0,2.0/3.0, x) + normpdf(1.0,2.0/3.0, x)) +
 (1.0/300.0)*(normpdf(-0.5,.01, x) + normpdf(-1.0,.01, x) + normpdf(-1.5,.01, x)) +
 (7.0/300.0)*(normpdf(0.5,.07, x) + normpdf(1.0,.07, x) + normpdf(1.5,.07, x)))
end

gr!(x, stor) = ForwardDiff.gradient!(stor, claw, x)

out = Genoud.genoud(claw, [0.0], sizepop = 5000, sense = :Max)
```
