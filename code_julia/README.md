# Julia Code

This code can be used to reproduce the numerical experiments
performed with [Julia](https://julialang.org).

## Usage

The code in this repository is developed for Julia 1.9.3.
To reproduce the numerical results, install Julia, start
Julia in this directory, and execute

```julia
julia> include("code.jl")

julia> stability_function_experiments()

julia> error_growth_bbm()

julia> taylor_green_experiments()
```

in the Julia REPL. You can improve the performance of the
Taylor-Green experiments by starting Julia with multiple
threads (e.g., `julia --threads=auto`).
