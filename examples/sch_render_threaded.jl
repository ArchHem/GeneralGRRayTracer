#examp usage



include("../backend/abstract_metrics.jl")
using DifferentialEquations, .metric_backend, Symbolics, StaticArrays



@variables x1::Real, x2::Real, x3::Real, x4::Real

coords = @SVector [x1, x2, x3, x4]

c = 1.0
M = 1.0
G = 1.0
const r_s = 2.0 * G * M / c^2

sch_g_00 = -(1.0 - r_s / x2) * c^2 
sch_g_11 =  (1.0 - r_s / x2)^(-1.0)
sch_g_22 = x2^2 * sin(x4)^2
sch_g_33 = x2^2 

#i.e. x3 = ϕ, x4 = θ

sch_metric_representation = @SMatrix [
    sch_g_00 0.0 0.0 0.0;
    0.0 sch_g_11 0.0 0.0;
    0.0 0.0 sch_g_22 0.0;
    0.0 0.0 0.0 sch_g_33
]

const epsilon = 0.0
schwarschild = AnalyticMetric(sch_metric_representation, coords)
accel, u0 = SymbolicIntegratorBuilder(schwarschild,   epsilon, do_simplify = false)

camera_pos = @SVector [0.0, 10.00, pi/2, pi/2]
camera_veloc = @SVector [1.0, 0.0, 0.0, 0.0]
camera_front = @SVector [0.0, -1.0, 0.0, 0.0]
camera_up = @SVector [0.0,0.0, 0.0, 1.0]

const N_x = 800
const N_y = 400
integrand, t, u0s = camera_rays_generator(schwarschild,camera_pos,camera_veloc,camera_front,camera_up,0.0027,N_x,N_y, epsilon = epsilon)

du(x,p,t) = accel(x,t)

function sch_flatness(integrand)
    r = integrand[1]

    return r > 25.0*r_s

end

redshift = approx_redshift_terminator(u0; critical_ratio =  5000, initial_u0 = 1.0)

condition(u, t, integrator) = sch_flatness(u) || redshift(u, t, integrator)
affect!(integrator) = terminate!(integrator)
cb = DiscreteCallback(condition, affect!)

tspan = (0.0,-100.0)
x0 = copy(integrand[1])
prob = ODEProblem{false}(du, integrand[1], tspan)
function prob_func(prob, i, repeat)
    remake(prob, u0=integrand[i])
end
ensembleprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
sol = solve(ensembleprob, Tsit5(), EnsembleThreads(), progress = true, 
trajectories = length(integrand), callback = cb, save_everystep=false, dtmax = 0.2)

#extract results 

end_vectors = Vector{Vector{Float64}}(undef,length(integrand))
end_times = Vector{Float64}(undef,length(integrand))

for i in 1:length(integrand)
    end_vectors[i] = sol[i].u[end]
    end_times[i] = sol[i].t[end]
end

function sch_to_mink(x0)
    t, r, phi, theta = x0

    x = cos(phi)*sin(theta)*r
    y = sin(phi)*sin(theta)*r
    z = cos(theta)*r

    x1 = [t, x, y, z]

    return x1

end


data = sphere_caster.(Ref(schwarschild), end_vectors, end_times, epsilon = epsilon, coordinate_transform = sch_to_mink)



using Images, ColorTypes

CS = load("./cspheres/tracker.png")

θ = [data[i][1] for i in 1:N_x*N_y]
ϕ = [data[i][2] for i in 1:N_x*N_y]

newim = spherical_to_rgb(θ, ϕ, CS)

is_redshifted = redshift.(end_vectors, end_times, 0.0)
newim[is_redshifted] .= RGBA(0.0,0.0,0.0,1.0)

newim = reshape(newim, (N_y, N_x))