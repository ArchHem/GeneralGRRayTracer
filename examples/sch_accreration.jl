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

const camera_pos = @SVector [0.0, 7*r_s, pi/2, pi/2 + pi/60]
const camera_veloc = @SVector [1.0, 0.0, 0.0, 0.0]
const camera_front = @SVector [0.0, -1.0, 0.0, 0.0]
const camera_up = @SVector [0.0,0.0, 0.0, 1.0]

const sup_res = 4
const N_x = 200*sup_res
const N_y = 100*sup_res

const N_trajectories = N_x * N_y

integrand, t, u0s = camera_rays_generator(schwarschild,camera_pos,camera_veloc,camera_front,camera_up,0.012/sup_res,N_x,N_y, epsilon = epsilon)

du(x,p,t) = accel(x,t)

function sch_flatness(integrand)
    r = integrand[1]

    return r > 25.0*r_s

end

redshift = approx_redshift_terminator(u0; critical_ratio =  5000, initial_u0 = 1.0)

mutable struct all_cont{T<:Real}
    freq::T
    iter::Int64
end

all_cont(w, index) = all_cont{typeof(w)}(w, index)

final_freqs = Vector{Float64}(undef, N_trajectories)

condition1(u, t, integrator) = sch_flatness(u) || redshift(u, t, integrator)
affect1!(integrator) = begin
    lindex = integrator.p.iter
    final_freqs[lindex] = integrator.p.freq
    terminate!(integrator) 
end
cb1 = DiscreteCallback(condition1, affect1!)

function is_in_ring(r::T,r_min::T, r_max::T) where T
    if r>r_min && r<r_max
        return r
    else
        return zero(T)
    end
end

const mind = 3.0
const maxd = 6.0
l_is_in(r) = is_in_ring(r,mind*r_s,maxd*r_s)


condition2(u, t, integrator) = begin cos(u[3]) end

const ring_freq = 0.

affect2!(integrator) = begin 
    l_r = integrator.u[1]
    l_r = l_is_in(l_r)

    inv_lr = l_r == 0.0 ? l_r : 1/l_r
    

    weight = (inv_lr*r_s)^2 * abs(cos(ring_freq*l_r/r_s))
    integrator.p.freq += weight
    return 
end

cb2 = ContinuousCallback(condition2,affect2!)

cb = CallbackSet(cb1, cb2)

tspan = (0.0,-250.0)
x0 = copy(integrand[1])
prob = ODEProblem{false}(du, integrand[1], tspan)
function prob_func(prob, i, repeat)
    remake(prob, u0=integrand[i], p = all_cont(0.0,i))
end
ensembleprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
sol = solve(ensembleprob, Tsit5(), EnsembleThreads(), progress = true, 
trajectories = N_trajectories, callback = cb, save_everystep=false, dtmax = 0.2)

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


black = RGBA(0.0, 0.0, 0.0, 1.0)


width, height = 10, 10


black_image = fill(black, height, width)

#CS = load("./cspheres/QUASI_CS.png")

CS = black_image

θ = [data[i][1] for i in 1:N_x*N_y]
ϕ = [data[i][2] for i in 1:N_x*N_y]

newim = spherical_to_rgb(θ, ϕ, CS)

final_freqs = reshape(final_freqs, (N_y, N_x))

is_redshifted = redshift.(end_vectors, end_times, 0.0)
newim[is_redshifted] .= RGBA(0.0,0.0,0.0,1.0)

function freq_to_RGB(w, wmax)
    w = w/wmax
    return RGBA(0.95*w, 0.05*w, 0.08*w, 1.0)
end

to_add = freq_to_RGB.(final_freqs, maximum(final_freqs))

newim = reshape(newim, (N_y, N_x))