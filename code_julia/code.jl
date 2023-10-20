# Install dependencies
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Load dependencies
using LinearAlgebra: I, dot, ldiv!, mul!, norm

using DelimitedFiles: writedlm

using NLsolve: nlsolve

using SimpleNonlinearSolve

using ForwardDiff: ForwardDiff
using StructArrays: StructArrays, StructArray

using SummationByPartsOperators
using Trixi

using LaTeXStrings
using Plots: Plots, plot, plot!, scatter, scatter!, savefig

using Printf: @sprintf


#####################################################################
# An advanced interface using AD to compute u''

# Some helper functions to get the second derivative via
# automatic/algorithmic differentiation (AD)
# https://github.com/JuliaDiff/ForwardDiff.jl/issues/319#issuecomment-685006123
function StructDual(x::AbstractArray{T, N}, w::AbstractArray{T, N}) where {T, N}
  @assert length(x) == length(w)
  # This was the original suggestion. However, it is currently not stable
  # under broadcasting. Thus, we use a slightly different version.
  # partials = StructArray{ForwardDiff.Partials{1, T}}(
  #     (StructArray{Tuple{T}}(
  #         (w,)
  #     ),)
  # )
  partials = reinterpret(reshape, ForwardDiff.Partials{1, T}, w)
  duals = StructArray{ForwardDiff.Dual{Nothing, T, 1}}((x, partials))
  return duals
end

function ForwardDiff.value(dx::StructArray{D}) where {D <: ForwardDiff.Dual}
  return dx.value
end

function ForwardDiff.partials(dx::StructArray{<: ForwardDiff.Dual{Tag, T, 1}}, i) where {Tag, T}
  # This was the original suggestion. We need to update it (see above).
  # return getproperty(dx.partials.values, i)
  @assert i == 1
  return reinterpret(reshape, T, dx.partials)
end

"""
    solve_explicit(f!, dt, tspan, u_analytical, parameters;
                   relaxation = true, nonconservative = false,
                   dot_entropy = dot,
                   entropy = nothing, entropy_rate = nothing,
                   norm_error = norm)

Solve the ODE `u' = f(u)`, `u'' = g(u) = f'f(u)` with the
explicit, fourth-order, two-derivative Runge-Kutta method of

> Chan, R.P.K., Tsai, A.Y.J.
> On explicit two-derivative Runge-Kutta methods.
> Numerical Algorithms 53, 171-194 (2010).
> https://doi.org/10.1007/s11075-009-9349-1

with time step size `dt` in a given time span `tspan`.
The analytical solution `u_analytical` is used to compute
errors and the initial condition.

`f!` is assumed to be an in-place update of the form
`f!(du, u, parameters)`. `g` is computed via AD from `f!`.

The function also measures the error (using `norm_error`) of the
numerical solution compared to the exact solution given by `u_analytical`.

If `relaxation` is set to `true`, the relaxation approach is used to
preserve the correct evolution of an entropy functional. If `entropy` is
`nothing`, it is assumed that the entropy is a quadratic functional
associated to the inner product `dot_entropy`. Otherwise, both the functional
`entropy(u, parameters)` and the corresponding `entropy_rate(u, du)` must
be provided.
If `nonconservative = true`, the entropy does not need to be conserved.
"""
function solve_explicit(f!, dt, tspan, u_analytical, parameters;
                        relaxation = true, relax_last_step = true,
                        nonconservative = false,
                        dot_entropy = dot,
                        entropy = nothing,
                        entropy_rate = (u, du) -> 2 * dot_entropy(u, du),
                        entropy_solver = ITP(),
                        norm_error = norm)
  t = first(tspan)
  # Values that will be used at the beginning of a step
  u = u_analytical(t, parameters)
  du = similar(u)
  u_du = StructDual(u, du)
  du_ddu = similar(u_du)
  ddu = ForwardDiff.partials(du_ddu, 1)
  # Temporary values used at the intermediate stage
  y = similar(u)
  dy = similar(u)
  y_dy = StructDual(y, dy)
  dy_ddy = similar(y_dy)
  ddy = ForwardDiff.partials(dy_ddy, 1)
  # Values used at the end of a step
  unew = similar(u)
  du_new = similar(u)
  u_du_new = StructDual(unew, du_new)
  du_ddu_new = similar(u_du_new)
  ddu_new = ForwardDiff.partials(du_ddu_new, 1)

  times = Vector{typeof(t)}()
  errors = Vector{eltype(u)}()
  entropies = Vector{eltype(u)}()

  while t < last(tspan)
    # Avoid stepping over the final time
    if t + dt > last(tspan)
      dt = last(tspan) - t
    end

    # Fourth-order method from section 3.2
    f!(du, u, parameters)
    # Instead of g!(ddu, u, parameters)
    f!(du_ddu, u_du, parameters)
    @. y = u + 0.5 * dt * du + 0.125 * dt^2 * ddu
    # Instead of g!(ddy, y, parameters)
    f!(dy, y, parameters)
    f!(dy_ddy, y_dy, parameters)
    @. unew = u + dt * du + (dt^2 / 6) * ddu + (dt^2 / 3) * ddy
    tnew = t + dt

    # # Third-order method from page 184
    # f!(du, u, parameters)
    # # Instead of g!(ddu, u, parameters)
    # f!(du_ddu, u_du, parameters)
    # @. y = u + dt * du + 0.5 * dt^2 * ddu
    # f!(dy, y, parameters)
    # @. unew = u + (2 * dt / 3) * du + (dt / 3) * dy + (dt^2 / 6) * ddu
    # tnew = t + dt

    if relaxation
      if nonconservative
        # Compute the relaxation parameter for a general (not necessarily
        # entropy-conservative) problem

        # We compute the estimate of the new entropy via a four-point
        # Gauss-Lobatto-Legendre quadrature (exact for polynomials of degree 5).
        # The corresponding values are
        # nodes = (0.0, 0.5 - sqrt(5) / 10, 0.5 + sqrt(5) / 10, 1.0)
        weights = (1 / 12, 5 / 12, 5 / 12, 1 / 12)

        # The values u, u' = du, u'' = ddu are available at the left node.
        # We compute the values u = unew, u' = du_new, u'' = ddu_new at the
        # right node.
        f!(du_new, unew, parameters)
        f!(du_ddu_new, u_du_new, parameters)

        # First Gauss-Lobatto-Legendre node
        entropy_diff = weights[1] * entropy_rate(u, du)
        # Second Gauss-Lobatto-Legendre node
        c_u0   = (250 + 82 * sqrt(5)) / 500
        c_up0  = ( 60 + 16 * sqrt(5)) / 500 * dt
        c_upp0 = (  5 +      sqrt(5)) / 500 * dt^2
        c_u1   = (250 - 82 * sqrt(5)) / 500
        c_up1  = (-60 + 16 * sqrt(5)) / 500 * dt
        c_upp1 = (  5 -      sqrt(5)) / 500 * dt^2
        @. y = (c_u0 * u    + c_up0 * du     + c_upp0 * ddu +
                c_u1 * unew + c_up1 * du_new + c_upp1 * ddu_new)
        f!(dy, y, parameters)
        entropy_diff += weights[2] * entropy_rate(y, dy)
        # Third Gauss-Lobatto-Legendre node
        c_u0   = (250 - 82 * sqrt(5)) / 500
        c_up0  = ( 60 - 16 * sqrt(5)) / 500 * dt
        c_upp0 = (  5 -      sqrt(5)) / 500 * dt^2
        c_u1   = (250 + 82 * sqrt(5)) / 500
        c_up1  = (-60 - 16 * sqrt(5)) / 500 * dt
        c_upp1 = (  5 +      sqrt(5)) / 500 * dt^2
        @. y = (c_u0 * u    + c_up0 * du     + c_upp0 * ddu +
                c_u1 * unew + c_up1 * du_new + c_upp1 * ddu_new)
        f!(dy, y, parameters)
        entropy_diff += weights[3] * entropy_rate(y, dy)
        # Fourth Gauss-Lobatto-Legendre node
        entropy_diff += weights[4] * entropy_rate(unew, du_new)
        entropy_diff *= dt

        if entropy === nothing
          # In this case, we have a quadratic functional.
          # For inner product norms, we have
          #   gamma = (entropy_new - entropy_old - 2 * <u, unew - u>) / |unew - u|^2
          @. y = unew - u
          gamma = (entropy_diff - 2 * dot_entropy(u, y)) / dot_entropy(y, y)
        else
          # Here, we have to deal with a general entropy functional
          initial_entropy = entropy(u, parameters)
          function residual_nonconservative(gamma, _)
            @. y = u + gamma * (unew - u)
            return entropy(y, parameters) - initial_entropy - gamma * entropy_diff
          end
          bounds = (0.9, 1.1)
          prob = IntervalNonlinearProblem(residual_nonconservative,
                                          bounds)
          sol = solve(prob, entropy_solver)
          gamma = sol.u
        end
      else
        # Compute the relaxation parameter for an entropy-conservative problem
        if entropy === nothing
          # In this case, we have a quadratic functional.
          # For inner product norms, we have
          #   gamma = -2 * <u, unew - u> / |unew - u|^2
          @. y = unew - u
          gamma = -2 * dot_entropy(u, y) / dot_entropy(y, y)
        else
          # Here, we have to deal with a general entropy functional
          initial_entropy = entropy(u, parameters)
          function residual_conservative(gamma, _)
            @. y = u + gamma * (unew - u)
            return entropy(y, parameters) - initial_entropy
          end
          bounds = (0.9, 1.1)
          prob = IntervalNonlinearProblem(residual_conservative,
                                          bounds)
          sol = solve(prob, entropy_solver)
          gamma = sol.u
        end
      end
    else
      gamma = one(t)
    end

    if tnew != last(tspan)
      @. u = u + gamma * (unew - u)
      t = t + gamma * (tnew - t)
    else
      if relax_last_step
        # Use an IDT step in the last step to hit the final time
        @. u = u + gamma * (unew - u)
      else
        @. u = unew
      end
      t = tnew
    end

    # compute functionals
    push!(times, t)
    push!(errors, norm_error(u - u_analytical(t, parameters)))
    if entropy === nothing
      # Quadratic entropy functional
      push!(entropies, dot_entropy(u, u))
    else
      push!(entropies, entropy(u, parameters))
    end
  end

  return (; t, u, times, errors, entropies, parameters)
end


#####################################################################
# Discretization of the BBM equation

function bbm_f!(du, u, parameters)
  (; D1, invImD2, tmp1, tmp2) = parameters
  one_third = one(eltype(D1)) / 3

  # This semidiscretization conserves the linear and quadratic invariants.
  tmp1 = @. -one_third * u^2
  tmp2 = D1 * tmp1
  mul!(tmp1, D1, u)
  @. tmp2 += -one_third * u * tmp1 - tmp1
  mul!(du, invImD2, tmp2)

  return nothing
end

function bbm_solution(t, x, xmin, xmax, c)
  # Physical setup of a traveling wave solution with speed `c`
  A = 3 * (c - 1)
  K = 0.5 * sqrt(1 - 1 / c)
  x_t = mod(x - c * t - xmin, xmax - xmin) + xmin

  return A / cosh(K * x_t)^2
end
function bbm_u_analytical(t, parameters)
  (; D1, xmin, xmax, c) = parameters
  x = grid(D1)
  u = bbm_solution.(t, x, xmin, xmax, c)
  return u
end

function bbm_main(; dt = 0.5, domain_traversals = 50,
                    ode_solve = solve_explicit, kwargs...)
  nnodes = 2^8
  xmin = -90.0
  xmax =  90.0
  c = 1.2

  D1 = fourier_derivative_operator(; xmin, xmax, N = nnodes)
  D2 = D1^2
  ImD2 = I - D2
  invImD2 = inv(I - D2)

  tspan = (0.0, (xmax - xmin) / (3 * c) +
                domain_traversals * (xmax - xmin) / c)
  u0 = bbm_u_analytical(first(tspan), (; D1, xmin, xmax, c))
  tmp1 = similar(u0)
  tmp2 = similar(u0)
  parameters = (; D1, D2, ImD2, invImD2, tmp1, tmp2, xmin, xmax, c)

  dot_entropy = let D = D1, ImD2 = ImD2, tmp = tmp1
    function dot(u, v)
      mul!(tmp, ImD2, v)
      @. tmp *= u
      return integrate(tmp, D)
    end
  end

  norm_error = let D = D1
    function norm(u)
      return integrate(abs2, u, D) |> sqrt
    end
  end

  res = ode_solve(bbm_f!, #= use AD =# dt, tspan,
                  bbm_u_analytical, parameters;
                  relaxation = false, nonconservative = false,
                  dot_entropy, norm_error,
                  kwargs...)
  @info "Baseline" extrema(res.entropies)
  fig_errors    = plot(res.times, res.errors, label = "Baseline",
                       xguide = "Time", yguide = "Error",
                       xscale = :log10, yscale = :log10,
                       legend = :topleft)
  fig_entropies = plot(res.times, res.entropies, label = "Baseline",
                       xguide = "Time", yguide = "Entropy")
  fig_solutions = plot(grid(res.parameters.D1), res.u, label = "Baseline",
                       xguide = L"x", yguide = L"u")

  res = ode_solve(bbm_f!, #= use AD =# dt, tspan,
                  bbm_u_analytical, parameters;
                  relaxation = true, nonconservative = false,
                  dot_entropy, norm_error,
                  kwargs...)
  @info "Relaxation" extrema(res.entropies)
  plot!(fig_errors,    res.times, res.errors, label = "Relaxation")
  plot!(fig_entropies, res.times, res.entropies, label = "Relaxation")
  plot!(fig_solutions, grid(res.parameters.D1), res.u, label = "Relaxation")
  plot!(fig_solutions, grid(res.parameters.D1),
        bbm_u_analytical(last(tspan), parameters), label = "Analytical",
        linestyle = :dash)

  plot(fig_errors, fig_entropies, fig_solutions)
end

function plot_kwargs()
  fontsizes = (
    xtickfontsize = 14, ytickfontsize = 14,
    xguidefontsize = 16, yguidefontsize = 16,
    legendfontsize = 14)
  (; linewidth = 3, gridlinewidth = 2,
     markersize = 8, markerstrokewidth = 4,
     fontsizes...)
end

function error_growth_bbm()
  figdir = @__DIR__

  fig_errors    = plot(xguide = L"Time $t$", yguide = "Error",
                       xscale = :log10, yscale = :log10,
                       legend = :topleft)
  fig_entropies = plot(xguide = L"Time $t$", yguide = L"Entropy $\eta(u(t))$",
                       legend = :topright)

  nnodes = 2^8
  xmin = -90.0
  xmax =  90.0
  c = 1.2

  D1 = fourier_derivative_operator(; xmin, xmax, N = nnodes)
  D2 = D1^2
  ImD2 = I - D2
  invImD2 = inv(I - D2)

  dt = 0.5
  domain_traversals = 60
  tspan = (0.0, (xmax - xmin) / (3 * c) +
                domain_traversals * (xmax - xmin) / c)
  u0 = bbm_u_analytical(first(tspan), (; D1, xmin, xmax, c))
  tmp1 = similar(u0)
  tmp2 = similar(u0)
  parameters = (; D1, D2, ImD2, invImD2, tmp1, tmp2, xmin, xmax, c)

  dot_entropy = let D = D1, ImD2 = ImD2, tmp = tmp1
    function dot(u, v)
      mul!(tmp, ImD2, v)
      @. tmp *= u
      return integrate(tmp, D)
    end
  end

  norm_error = let D = D1
    function norm(u)
      return integrate(abs2, u, D) |> sqrt
    end
  end

  res = solve_explicit(bbm_f!, dt, tspan,
                       bbm_u_analytical, parameters;
                       relaxation = false, dot_entropy, norm_error)
  plot!(fig_errors, res.times[2:end], res.errors[2:end];
        label = "Baseline", plot_kwargs()...)
  plot!(fig_entropies, res.times, res.entropies;
        label = "Baseline", plot_kwargs()...)
  let data = hcat(res.times, res.errors, res.entropies)
    open(joinpath(figdir, "bbm__CT42__baseline.csv"), "w") do io
      println(io, "# time\terror\tentropy")
      writedlm(io, data)
    end
  end

  res = solve_explicit(bbm_f!, dt, tspan,
                       bbm_u_analytical, parameters;
                       relaxation = true, dot_entropy, norm_error)
  plot!(fig_errors, res.times[2:end], res.errors[2:end];
        label = "Relaxation", linestyle = :dash, plot_kwargs()...)
  plot!(fig_entropies, res.times, res.entropies;
        label = "Relaxation", linestyle = :dash, plot_kwargs()...)
  let data = hcat(res.times, res.errors, res.entropies)
    open(joinpath(figdir, "bbm__CT42__relaxation.csv"), "w") do io
      println(io, "# time\terror\tentropy")
      writedlm(io, data)
    end
  end

  savefig(fig_errors, joinpath(figdir, "bbm__CT42__errors.pdf"))
  savefig(fig_entropies, joinpath(figdir, "bbm__CT42__entropies.pdf"))

  @info "Results saved in the directory" figdir
end


#####################################################################
# DG with Trixi.jl

# These are some hacks to make it work...
function Trixi.wrap_array(u_ode::StructArray,
                          mesh::Trixi.AbstractMesh, equations, dg::DGSEM, cache)
  u_value = Trixi.wrap_array(ForwardDiff.value(u_ode),
                             mesh, equations, dg, cache)
  u_duals = Trixi.wrap_array(ForwardDiff.partials(u_ode, 1),
                             mesh, equations, dg, cache)
  u = StructDual(u_value, u_duals)
  return u
end
function Trixi.reset_du!(du::StructArray, dg::DGSEM, cache)
  du .= zero(eltype(du))
  return nothing
end

function trixi_linadv(; dt = 0.05, ode_solve = solve_explicit, kwargs...)
  equations = LinearScalarAdvectionEquation1D(1.0)
  solver = DGSEM(polydeg = 3, surface_flux = flux_godunov)
  coordinates_min = (-1.0,)
  coordinates_max = (+1.0,)
  mesh = TreeMesh(coordinates_min, coordinates_max;
                  initial_refinement_level = 3,
                  n_cells_max = 10^5,
                  periodicity = true)
  semi_base = SemidiscretizationHyperbolic(mesh, equations,
                                          initial_condition_convergence_test,
                                          solver)
  semi_dual = SemidiscretizationHyperbolic(mesh, equations,
                                          initial_condition_convergence_test,
                                          solver;
                                          uEltype = ForwardDiff.Dual{Nothing, Float64, 1})
  tspan = (0.0, 100.0)

  function rhs!(du_ode, u_ode, parameters)
    if eltype(u_ode) <: ForwardDiff.Dual
      Trixi.rhs!(du_ode, u_ode, parameters.semi_dual, nothing)
    else
      Trixi.rhs!(du_ode, u_ode, parameters.semi_base, nothing)
    end
    return nothing
  end

  function dot_entropy(u, v)
    return integrate(first, u .* v, semi_base; normalize = false)
  end

  function norm_error(u)
    return integrate(abs2 ∘ first, u, semi_base; normalize = false) |> sqrt
  end

  function u_analytical(t, parameters)
    u_ode = compute_coefficients(t, parameters.semi_base)
    return u_ode
  end

  res = ode_solve(rhs!, #= use AD =# dt, tspan,
                  u_analytical, (; semi_base, semi_dual);
                  relaxation = false, nonconservative = true,
                  dot_entropy, norm_error,
                  kwargs...)
  @info "Baseline" extrema(res.entropies)
  fig_errors    = plot(res.times[2:end], res.errors[2:end], label = "Baseline",
                       xguide = "Time", yguide = "Error",
                       xscale = :log10, yscale = :log10)
  fig_entropies = plot(res.times, res.entropies, label = "Baseline",
                       xguide = "Time", yguide = "Entropy")

  res = ode_solve(rhs!, #= use AD =# dt, tspan,
                  u_analytical, (; semi_base, semi_dual);
                  relaxation = true, nonconservative = true,
                  dot_entropy, norm_error,
                  kwargs...)
  @info "Relaxation" extrema(res.entropies)
  plot!(fig_errors,    res.times[2:end], res.errors[2:end], label = "Relaxation",
        linestyle = :dash)
  plot!(fig_entropies, res.times, res.entropies, label = "Relaxation",
        linestyle = :dash)

  plot(fig_errors, fig_entropies)
end

function trixi_densitywave(; dt = 0.002, ode_solve = solve_explicit, kwargs...)
  equations = CompressibleEulerEquations1D(1.4)
  solver = DGSEM(polydeg = 3,
                 volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
                 surface_flux = flux_ranocha)
  coordinates_min = (-1.0,)
  coordinates_max = (+1.0,)
  mesh = TreeMesh(coordinates_min, coordinates_max;
                  initial_refinement_level = 3,
                  n_cells_max = 10^5,
                  periodicity = true)
  semi_base = SemidiscretizationHyperbolic(mesh, equations,
                                          initial_condition_density_wave,
                                          solver)
  semi_dual = SemidiscretizationHyperbolic(mesh, equations,
                                          initial_condition_density_wave,
                                          solver;
                                          uEltype = ForwardDiff.Dual{Nothing, Float64, 1})
  tspan = (0.0, 100.0)

  function rhs!(du_ode, u_ode, parameters)
    if eltype(u_ode) <: ForwardDiff.Dual
      Trixi.rhs!(du_ode, u_ode, parameters.semi_dual, nothing)
    else
      Trixi.rhs!(du_ode, u_ode, parameters.semi_base, nothing)
    end
    return nothing
  end

  function entropy(u_ode, parameters)
    return integrate(Trixi.entropy, u_ode, parameters.semi_base; normalize = false)
  end

  function norm_error(u)
    return integrate(abs2 ∘ first, u, semi_base; normalize = false) |> sqrt
  end

  function u_analytical(t, parameters)
    u_ode = compute_coefficients(t, parameters.semi_base)
    return u_ode
  end

  res = ode_solve(rhs!, #= use AD =# dt, tspan,
                  u_analytical, (; semi_base, semi_dual);
                  relaxation = false, nonconservative = false,
                  entropy, norm_error,
                  kwargs...)
  @info "Baseline" extrema(res.entropies) -(extrema(res.entropies)...)
  fig_errors    = plot(res.times[2:end], res.errors[2:end], label = "Baseline",
                       xguide = "Time", yguide = "Error",
                       xscale = :log10, yscale = :log10)
  fig_entropies = plot(res.times, res.entropies, label = "Baseline",
                       xguide = "Time", yguide = "Entropy")

  res = ode_solve(rhs!, #= use AD =# dt, tspan,
                  u_analytical, (; semi_base, semi_dual);
                  relaxation = true, nonconservative = false,
                  entropy, norm_error,
                  kwargs...)
  @info "Relaxation" extrema(res.entropies) -(extrema(res.entropies)...)
  plot!(fig_errors,    res.times[2:end], res.errors[2:end], label = "Relaxation",
        linestyle = :dash)
  plot!(fig_entropies, res.times, res.entropies, label = "Relaxation",
        linestyle = :dash)

  plot(fig_errors, fig_entropies)
end

function relative_change(v)
  return (v[(begin+1):end] .- v[begin]) ./ abs(v[begin])
end

function trixi_taylorgreen(; dt = 0.005, t_end = 20.0,
                             surface_flux = flux_ranocha,
                             ode_solve = solve_explicit, kwargs...)
  equations = CompressibleEulerEquations3D(1.4)
  function initial_condition_taylor_green_vortex(x, t,  equations)
    A  = 1.0 # magnitude of speed
    Ms = 0.1 # maximum Mach number

    rho = 1.0
    v1  =  A * sin(x[1]) * cos(x[2]) * cos(x[3])
    v2  = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
    v3  = 0.0
    p   = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
    p   = p + 1.0/16.0 * A^2 * rho * (cos(2*x[1])*cos(2*x[3]) + 2*cos(2*x[2]) + 2*cos(2*x[1]) + cos(2*x[2])*cos(2*x[3]))

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
  end
  solver = DGSEM(polydeg = 3,
                 volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha_turbo),
                 surface_flux = surface_flux)
  coordinates_min = (-1.0, -1.0, -1.0,) .* π
  coordinates_max = (+1.0, +1.0, +1.0,) .* π
  mesh = TreeMesh(coordinates_min, coordinates_max;
                  initial_refinement_level = 3,
                  n_cells_max = 10^5,
                  periodicity = true)
  semi_base = SemidiscretizationHyperbolic(mesh, equations,
                                          initial_condition_taylor_green_vortex,
                                          solver)
  semi_dual = SemidiscretizationHyperbolic(mesh, equations,
                                          initial_condition_taylor_green_vortex,
                                          solver;
                                          uEltype = ForwardDiff.Dual{Nothing, Float64, 1})
  tspan = (0.0, t_end)

  function rhs!(du_ode, u_ode, parameters)
    if eltype(u_ode) <: ForwardDiff.Dual
      Trixi.rhs!(du_ode, u_ode, parameters.semi_dual, nothing)
    else
      Trixi.rhs!(du_ode, u_ode, parameters.semi_base, nothing)
    end
    return nothing
  end

  function entropy(u_ode, parameters)
    return integrate(Trixi.entropy, u_ode, parameters.semi_base; normalize = false)
  end

  function entropy_rate(u_ode, du_ode)
    u = Trixi.wrap_array(u_ode, semi_base)
    du = Trixi.wrap_array(du_ode, semi_base)
    Trixi.analyze(Trixi.entropy_timederivative, du, u, nothing, semi_base)
  end

  function norm_error(u)
    # Not used since we don't have an analytical solution
    # return integrate(abs2 ∘ first, u, semi_base; normalize = false) |> sqrt
    return 0.0
  end

  u0_ode = compute_coefficients(0.0, semi_base)
  function u_analytical(t, parameters)
    return u0_ode
  end

  res = ode_solve(rhs!, #= use AD =# dt, tspan,
                  u_analytical, (; semi_base, semi_dual);
                  relaxation = false,
                  nonconservative = surface_flux != flux_ranocha,
                  entropy, entropy_rate, norm_error,
                  kwargs...)
  @info "Baseline" extrema(res.entropies) -(extrema(res.entropies)...)
  baseline_time = res.times[2:end]
  baseline_relative_entropy = relative_change(res.entropies)
  fig_entropies = plot(baseline_time, baseline_relative_entropy;
                       label = "Baseline", plot_kwargs()...,
                       xguide = "Time", yguide = "Rel. entropy change")

  res = ode_solve(rhs!, #= use AD =# dt, tspan,
                  u_analytical, (; semi_base, semi_dual);
                  relaxation = true,
                  nonconservative = surface_flux != flux_ranocha,
                  entropy, entropy_rate, norm_error,
                  kwargs...)
  @info "Relaxation" extrema(res.entropies) -(extrema(res.entropies)...)
  relaxation_time = res.times[2:end]
  relaxation_relative_entropy = relative_change(res.entropies)
  plot!(fig_entropies, relaxation_time, relaxation_relative_entropy;
        label = "Relaxation", plot_kwargs()...,
        linestyle = :dot)

  return (; fig_entropies,
            baseline_time, baseline_relative_entropy,
            relaxation_time, relaxation_relative_entropy)
end

function taylor_green_experiments()
  figdir = @__DIR__

  @info "Running the entropy-conservative setup"
  res = trixi_taylorgreen(surface_flux = flux_ranocha)
  let data = hcat(res.baseline_time, res.baseline_relative_entropy)
    open(joinpath(figdir, "taylor_green_vortex__entropy__conservative_baseline.csv"), "w") do io
      println(io, "# time\trelative entropy change")
      writedlm(io, data)
    end
  end
  let data = hcat(res.relaxation_time, res.relaxation_relative_entropy)
    open(joinpath(figdir, "taylor_green_vortex__entropy__conservative_relaxation.csv"), "w") do io
      println(io, "# time\trelative entropy change")
      writedlm(io, data)
    end
  end
  savefig(res.fig_entropies,
          joinpath(figdir, "taylor_green_vortex__entropy__conservative.pdf"))

  @info "Running the entropy-dissipative setup"
  res = trixi_taylorgreen(surface_flux = flux_lax_friedrichs)
  let data = hcat(res.baseline_time, res.baseline_relative_entropy)
    open(joinpath(figdir, "taylor_green_vortex__entropy__dissipative_baseline.csv"), "w") do io
      println(io, "# time\trelative entropy change")
      writedlm(io, data)
    end
  end
  let data = hcat(res.relaxation_time, res.relaxation_relative_entropy)
    open(joinpath(figdir, "taylor_green_vortex__entropy__dissipative_relaxation.csv"), "w") do io
      println(io, "# time\trelative entropy change")
      writedlm(io, data)
    end
  end
  savefig(res.fig_entropies,
          joinpath(figdir, "taylor_green_vortex__entropy__dissipative.pdf"))

  @info "Results saved in the directory" figdir
  return nothing
end


#####################################################################
# stability functions

"""
    JaustSchuetzSeal2016ThirdOrderLstable()

Implicit third-order L-stable method;
see equation (7) of
- Jaust, Schütz, Seal (2016)
  https://doi.org/10.1007/s10915-016-0221-x
"""
struct JaustSchuetzSeal2016ThirdOrderLstable end

function stability_function(::JaustSchuetzSeal2016ThirdOrderLstable)
  r(z) = (1 + z / 3) / (1 - 2 * z / 3 + z^2 / 6)
end


"""
    SchuetzDealZeifang2022FourthOrderAstable()

Implicit, fourth-order, two-derivative Runge-Kutta method of

- Jochen Schütz, David C. Seal, Jonas Zeifang
  Parallel-in-Time High-Order Multiderivative IMEX Solvers.
  Journal of Scientific Computing (2022).
  https://doi.org/10.1007/s10915-021-01733-3
  Example 1
"""
struct SchuetzDealZeifang2022FourthOrderAstable end

function stability_function(::SchuetzDealZeifang2022FourthOrderAstable)
  r(z) = (12 + 6 * z + z^2) / (12 - 6 * z + z^2)
end

"""
    ImplicitEuler()

The classical implicit Euler method.
"""
struct ImplicitEuler end

function stability_function(::ImplicitEuler)
  r(z) = 1 / (1 - z)
end

"""
    HB_I2DRK6_3s()

See
```bibtex
@article{schutz2017implicit,
  title={Implicit multiderivative collocation solvers for linear partial
         differential equations with discontinuous {G}alerkin spatial
         discretizations},
  author={Sch{\"u}tz, Jochen and Seal, David C and Jaust, Alexander},
  journal={Journal of Scientific Computing},
  volume={73},
  pages={1145--1163},
  year={2017},
  publisher={Springer},
  doi={10.1007/s10915-017-0485-9}
}
```
"""
struct HB_I2DRK6_3s end

function stability_function(::HB_I2DRK6_3s)
  r(z) = (z^4 + 18*z^3 + 156*z^2 + 720*z + 1440) / (z^4 - 18*z^3 + 156*z^2 - 720*z + 1440)
end

"""
    HB_I2DRK8_4s()

See
```bibtex
@article{schutz2022parallel,
  title={Parallel-in-time high-order multiderivative {IMEX} solvers},
  author={Sch{\"u}tz, Jochen and Seal, David C and Zeifang, Jonas},
  journal={Journal of Scientific Computing},
  volume={90},
  number={1},
  pages={54},
  year={2022},
  publisher={Springer},
  doi={10.1007/s10915-021-01733-3}
}
```
"""
struct HB_I2DRK8_4s end

function stability_function(::HB_I2DRK8_4s)
  r(z) = (80704505322479284649.0*z^6 + 2663248675641816580638.0*z^5 + 46727908581715508333748.0*z^4 + 522965194489665792773865.0*z^3 + 3791497660050076965273600.0*z^2 + 16473403626424472331878400.0*z + 32946807252848944663756800.0) / (3*(26901501774159762693.0*z^6 - 887749558547272170074.0*z^5 + 15575969527238502654436.0*z^4 - 174321731496555262673595.0*z^3 + 1263832553350025655091200.0*z^2 - 5491134542141490777292800.0*z + 10982269084282981554585600.0))
end

"""
    HB_I3DRK9_3s()

A three-derivative method.
"""
struct HB_I3DRK9_3s end

function stability_function(::HB_I3DRK9_3s)
  r(z) = (4*(945755921747804107.0*z^6 + 34047213182920949760.0*z^5 + 624198908353550740512.0*z^4 + 7149914768413399449600.0*z^3 + 52432708301698262630400.0*z^2 + 228797272589228782387200.0*z + 457594545178457564774400.0))/(3783023686991216747.0*z^6 - 136188852731683799040.0*z^5 + 2496795633414202992672.0*z^4 - 28599659073653597798400.0*z^3 + 209730833206793050521600.0*z^2 - 915189090356915129548800.0*z + 1830378180713830259097600.0)
end

"""
    HB_I3DRK12_4s()

A three-derivative method with four stages.
"""
struct HB_I3DRK12_4s end

function stability_function(::HB_I3DRK12_4s)
  r(z) = -(1232*(646956598837339685073421289369419259547.0*z^9 + 42699135523264428835020185538864621345303.0*z^8 + 1523582790261935378982186783153439048300987.0*z^7 + 37022091368466771880253451655797327261722912.0*z^6 + 656615660857981364819085252046979660217723904.0*z^5 + 8642382664700580894340420913102718342319308800.0*z^4 + 83195772001447021439694962678037482637172408320.0*z^3 + 558600183438287113715139441902131812224477429760.0*z^2 + 2353251836612358376413439562560789225682868633600.0*z + 4706503673224716752826879125121578451365737267200.0))/(797050529767602622373092416186785266398531.0*z^9 - 52605334964661773133267308575337508588408609.0*z^8 + 1877053997602704170508801248273792640772332515.0*z^7 - 45611216565951061219392197709485710297207642336.0*z^6 + 808950494177032942684360907271759806357690728448.0*z^5 - 10647415442911115244578956903396023807213083557888.0*z^4 + 102497191105782721381911594828142028479639018536960.0*z^3 - 688195425995969696006924736413323912922691608248320.0*z^2 + 2899206262706425519741357541074892326041294156595200.0*z - 5798412525412851039482715082149784652082588313190400.0)
end

"""
    HB_I4DRK8_2s()

A four-derivative method.
"""
struct HB_I4DRK8_2s end

function stability_function(::HB_I4DRK8_2s)
  r(z) = (z^4 + 20*z^3 + 180*z^2 + 840*z + 1680)/(z^4 - 20*z^3 + 180*z^2 - 840*z + 1680)
end

"""
    HB_I4DRK12_3s()

A four-derivative method.
"""
struct HB_I4DRK12_3s end

function stability_function(::HB_I4DRK12_3s)
  r(z) = (4*(1387108685230112768.0*z^8 + 83226521113806764755.0*z^7 + 2580022154528009821568.0*z^6 + 52432708301698264744680.0*z^5 + 748039971770895222543360.0*z^4 + 7550309995444550091571200.0*z^3 + 52013246635284676558848000.0*z^2 + 221475759866373461350809600.0*z + 442951519732746922701619200.0))/(5548434740920451072.0*z^8 - 332906084455227062613.0*z^7 + 10320088618112039071168.0*z^6 - 209730833206793050316760.0*z^5 + 2992159887083580862394880.0*z^4 - 30201239981778199275110400.0*z^3 + 208052986541138705999462400.0*z^2 - 885903039465493845403238400.0*z + 1771806078930987690806476800.0)
end

"""
    SSP_I2DRK3_2s()

A two-derivative method. See
```bibtex
@article{gottlieb2022high,
  title={High Order Strong Stability Preserving Multiderivative Implicit
         and {IMEX} {R}unge-{K}utta Methods with Asymptotic Preserving
         Properties},
  author={Gottlieb, Sigal and Grant, Zachary J and Hu, Jingwei and
          Shu, Ruiwen},
  journal={SIAM Journal on Numerical Analysis},
  volume={60},
  number={1},
  pages={423--449},
  year={2022},
  publisher={SIAM},
  doi={10.1137/21M1403175}
}
```
"""
struct SSP_I2DRK3_2s end

function stability_function(::SSP_I2DRK3_2s)
  r(z) = 18/(z^4 - 3*z^3 + 9*z^2 - 18*z + 18)
end

"""
    SSP_I2DRK4_5s()

A two-derivative method. See
```bibtex
@article{gottlieb2022high,
  title={High Order Strong Stability Preserving Multiderivative Implicit
         and {IMEX} {R}unge-{K}utta Methods with Asymptotic Preserving
         Properties},
  author={Gottlieb, Sigal and Grant, Zachary J and Hu, Jingwei and
          Shu, Ruiwen},
  journal={SIAM Journal on Numerical Analysis},
  volume={60},
  number={1},
  pages={423--449},
  year={2022},
  publisher={SIAM},
  doi={10.1137/21M1403175}
}
```
"""
struct SSP_I2DRK4_5s end

function stability_function(::SSP_I2DRK4_5s)
  r(z) = (40564819207303340847894502572032.0*(1081939488881458343015503484642835956869907222016.0*z^6 - 7808497762921559648667084742326295271271280361172.0*z^5 + 27981150261373043470567863556070258393585091609613.0*z^4 - 60784119921691636072491432311861136394607441150298.0*z^3 + 90414541222105875759971277490358917431002717487104.0*z^2 - 86864597453170205566863291079415274617564657876992.0*z + 46768052394588893382517914646921056628989841375232.0))/(1709372719277737994366648126268973308061798134554742886558838151741848274024960.0*z^10 - 19244454758042949898562287887624699494882445211351576172318898889757455788832476.0*z^9 + 118214649179994202757089504386771839508379527868822528239790645344589281037321119.0*z^8 - 488454924637658829103743429921437344663783499107882868953477772012221144316470694.0*z^7 + 1488356736654113538251674243516297487176965934761218892040594528735685789983004416.0*z^6 - 3438967001633728708751702768959161048999641385085079029609885336175838358996252000.0*z^5 + 6100893744176292366777546834323952414667422478994959479901736655473897135912517632.0*z^4 - 8211359297630006672480219995928815840541396949390038792956431940017528479087067136.0*z^3 + 8139865004621120567825725988710842531352666350167459873141670562610545347730079744.0*z^2 - 5420784281267218360809085845894218465958584511406981374243019488709373958620708864.0*z + 1897137590064188545819787018382342682267975428761855001222473056385648716020711424.0)
end


"""
    plot_stability_region(method, γ = 1.0)

Plot the stability region for a given `method` with fixed relaxation
parameter γ.
"""
function plot_stability_region(method, γ = 1.0)
  r = stability_function(method)

  f(z) = abs2(1 + γ * (r(z) - 1))

  n = 1_000
  x = range(-10.0, 5.0, length = n)
  y = range(-5.0, 5.0, length = n)
  z = x' .+ im .* y
  values = f.(z) .<= 1
  fig = Plots.heatmap(x, y, values) #; aspect_ratio = :equal)

  # imaginary axis
  v = f.(im .* y)
  # v = f.(im .* big.(y))
  @show extrema(v)

  return fig
end

"""
    stability_angle(method, γ)

Given a `method`, approximate the angle α (in degree) of
A(α)-stability of the relaxed update with fixed relaxation parameter γ.
"""
function stability_angle(r, γ)
  # Squared absolute value of the relaxed stability function
  f(z) = abs2(1 + γ * (r(z) - 1))

  # Sample the line with fixed maximal radius at a given angle φ
  # to check for A(α)-stability
  function objective(α)
    n = 10_000
    radii = range(0.0, 10.0, length = n)
    maximum(radii) do r
      z = r * cis(α)
      return f(z)
    end
  end

  res = nlsolve([π / 2]; ftol = 1.0e-12) do residual, α
    residual[1] = objective(α[1]) - 1
    return nothing
  end

  α = res.zero[1]
  return rad2deg(π - α)
end

"""
    stability_angle(method)

Given a `method`, plot the angle α of A(α)-stability of the relaxed
update with fixed relaxation parameter γ = 1.1.
"""
function stability_angle(method)
  # TODO: obsolete?
  r = stability_function(method)

  γ = 1.1
  α = stability_angle(r, γ)

  # Squared absolute value of the relaxed stability function
  f(z) = abs2(1 + γ * (r(z) - 1))

  n = 1_000
  x = range(-10.0, 5.0, length = n)
  y = range(-5.0, 5.0, length = n)
  z = x' .+ im .* y
  values = f.(z) .<= 1
  fig = Plots.heatmap(x, y, values) #; aspect_ratio = :equal)

  label = @sprintf("\$\\alpha \\approx %.2f\\degree\$", α)
  plot!(fig, [0.0, 5.0 * cosd(180 - α)], [0.0, 5.0 * sind(180 - α)];
        color = "gray", label, linewidth = 5, linestyle = :dot)

  return fig
end

"""
    stability_angle(method)

Given a `method`, plot the angle α of A(α)-stability of the relaxed
update for different values of the relaxation parameter γ.
"""
function plot_stability_angle(method)
  r = stability_function(method)

  dγ = 10.0 .^ range(-15.0, -1.0, length = 1_000)
  α = @. stability_angle.(r, 1 + dγ)
  idx = findfirst(<(90.0 - 1.0e-12), α)
  if idx !== nothing
    γ = 1 + dγ[idx]
    @info "first non-A-stable value"  γ γ-1
  end
  label = L"90\degree - \alpha"
  quarter_minus_α = nextfloat(90.0) .- α
  fig = plot(dγ, quarter_minus_α; plot_kwargs()...,
       label = label, legend = :left,
       xguide = L"\gamma - 1", yguide = label,
       xscale = :log10, yscale = :log10,
       color = :black)
  label = L"\alpha"
  plot!(Plots.twinx(), dγ, α; plot_kwargs()...,
        label = label, legend = :bottomright,
        yguide = label,
        xscale = :log10,
        linestyle = :dot, color = :gray)
  Plots.xticks!([1.0e-15, 1.0e-10, 1.0e-5, 1.0e-1])

  return (; fig, dγ, α, quarter_minus_α)
end


function stability_function_experiments()
  figdir = @__DIR__

  methods = [
    JaustSchuetzSeal2016ThirdOrderLstable(),
    SchuetzDealZeifang2022FourthOrderAstable(),
    HB_I2DRK6_3s(),
    HB_I2DRK8_4s(),
    HB_I3DRK9_3s(),
    HB_I3DRK12_4s(),
    HB_I4DRK8_2s(),
    HB_I4DRK12_3s(),
    SSP_I2DRK3_2s(),
    SSP_I2DRK4_5s()
  ]

  for method in methods
    @show method
    res = plot_stability_angle(method)
    let data = hcat(res.dγ, res.α, res.quarter_minus_α)
      open(joinpath(figdir, "stability_function_$(nameof(typeof(method))).csv"), "w") do io
        println(io, "# gamma - 1\talpha\t90 degree - alpha")
        writedlm(io, data)
      end
    end
    savefig(res.fig,
            joinpath(figdir, "stability_function_$(nameof(typeof(method))).pdf"))
  end

  @info "Results saved in the directory" figdir
end
