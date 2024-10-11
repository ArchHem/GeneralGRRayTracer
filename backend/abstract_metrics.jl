module metric_backend

include("./numeric_src.jl")
include("./symbolic_src.jl")

using .numeric_backend, .symbolic_backend

using LinearAlgebra, Symbolics, StaticArrays, ForwardDiff, Roots, ColorTypes, Interpolations
#for help construct the following papers:
#on general geodesic tracing: https://iopscience.iop.org/article/10.3847/1538-4365/aac9ca/pdf?fbclid=IwAR0pORzJb6EvCVdTIWo32F6wxhdd3_eQE_-x8afe94Y8dY_2IH_NuNcPiD0
#on how to generate the camera rays and how to work in an arbitrary metric: https://arxiv.org/pdf/1410.7775
#for the automatic (planned) scattering at timelike infinity: https://physics.stackexchange.com/questions/240898/coordinates-vs-geometries-how-can-we-know-two-coordinate-systems-describe-the

struct AnalyticMetric{T1,T2}
    metric::SMatrix{4,4,Num,16}
    inverse_metric::SMatrix{4,4,Num,16}
    coordinates::SVector{4,Num}
    numeric_metric::T1
    numeric_inverse_metric::T2
end

#convinence constructors
AnalyticMetric(met,coords; do_simplify::Bool = false) = begin
    inv_met = inv(met)
    if do_simplify
        inv_met = simplify.(inv_met)
    end
    numeric_metric = numeric_array_generator(met,coords)
    numeric_inverse_metric = numeric_array_generator(inv_met,coords)
    AnalyticMetric{typeof(numeric_metric),typeof(numeric_inverse_metric)}(
        met,inv_met,coords, numeric_metric, numeric_inverse_metric)
end

#the integration of the geodesics requires separating α, γ, β and the derivatives of each wrt. to each spatial coordinate

#for the stopping condition, we shall use the Ricci scalar; heavy TODO 

#the next few elements will build the associated integrators.

function SymbolicIntegratorBuilder(metric::AnalyticMetric, epsilon::T; do_simplify::Bool = false) where T<:Real
    #
    b_index = StaticArrays.SUnitRange(2,4)

    gamma = metric.metric[b_index,b_index]
    beta = metric.metric[1,b_index]

    gamma_up = inv(gamma)
    beta_up = gamma_up * beta
    alpha = sqrt(beta'beta_up - metric.metric[1,1])

    base_coords = metric.coordinates

    acceleration_vector =  zeros(Num,6)
    
    @variables u0::Real, u1::Real, u2::Real, u3::Real
    #these are lowered indeces
    lower_spatial_u = [u1, u2, u3]
    
    implicit_u0 = sqrt(lower_spatial_u'gamma_up*lower_spatial_u + epsilon)/alpha

    if do_simplify
        implicit_u0 = simplify(implicit_u0)
    end

    dx_spatial = (gamma_up * lower_spatial_u) ./ implicit_u0 .- beta_up

    if do_simplify
        dx_spatial = simplify.(dx_spatial)
    end

    acceleration_vector[1:3] = dx_spatial

    differential_operators = [Differential(base_coords[i]) for i in 2:4]

    du_spatial = zeros(Num,(3,))

    for i in 1:3
        alpha_deriv = expand_derivatives(differential_operators[i](alpha))
        beta_up_deriv = zeros(Num,(3,))
        gamma_up_deriv = zeros(Num,(3,3))

        for k in eachindex(beta_up_deriv)
            beta_up_deriv[k] = expand_derivatives(differential_operators[i](beta_up[k]))
        end
        for k in eachindex(gamma_up_deriv)
            gamma_up_deriv[k] = expand_derivatives(differential_operators[i](gamma_up[k]))
        end
        
        du_spatial[i] = -alpha * implicit_u0 * alpha_deriv + lower_spatial_u'beta_up_deriv - 1/(2 * implicit_u0) * (lower_spatial_u'gamma_up_deriv*lower_spatial_u)
    end

    if do_simplify
        du_spatial .= simplify.(du_spatial)
    end

    
    acceleration_vector[4:6] = du_spatial

    #for odeproblems, its customary that the time is the last input
    spatial_inputs = [base_coords[2:4]..., u1, u2, u3]

    #wrap for constructor
    spatial_inputs = SVector{6,Num}(spatial_inputs)
    acceleration_vector = SVector{6,Num}(acceleration_vector)

    acceleration_quotes = build_function(acceleration_vector,spatial_inputs, base_coords[1])

    acceleration_function = eval(acceleration_quotes[1])

    u0_quotes = build_function(implicit_u0, spatial_inputs, base_coords[1])

    u0_function = eval(u0_quotes)

    return acceleration_function, u0_function
end

function camera_rays_generator(metric_repr, 
    camera_fourpos::SVector{4,T}, camera_velocity::SVector{4,T}, camera_front_vector::SVector{4,T}, camera_up_vector::SVector{4,T}, 
    angular_pixellation::T, N_x::Z, N_y::Z; epsilon::T = zero(T)) where {T<:Real, Z<:Integer}

    #initializes the camera rays in datatype T for solvers 

    local_metric = metric_repr.numeric_metric(camera_fourpos)
    local_inverse_metric = metric_repr.numeric_inverse_metric(camera_fourpos)

    levi = levi_civita_generator(T)

    metric_determinant = det(local_metric)

    initial_norm = camera_velocity'local_metric*camera_velocity

    if initial_norm > one(T)
        throw(ArgumentError("Spacelike four-velocity given for the camera."))
    end
    if (camera_front_vector[1] == camera_up_vector[1] == zero(T)) != true
        throw((ArgumentError("Camera alignment vectors must have a zero temporal part!")))
    end

    normalizing_quant = sqrt(-one(T)/initial_norm)

    camera_velocity= normalizing_quant .* camera_velocity

    e0 = copy(camera_velocity)

    #since v1, v2 is going to be a spacelike vector....
    
    v1 = camera_front_vector + (e0'local_metric*camera_front_vector) .* e0

    e1 = v1 ./ sqrt(v1'local_metric*v1)

    #whener we are projecting unto e0, use that its norm is -1, not +1, hence the addition.

    v2 = camera_up_vector + (camera_up_vector'local_metric*e0) .*e0 - (camera_up_vector'local_metric*e1) .*e1

    e2 = v2 ./ sqrt(v2'local_metric*v2)

    e3 = sqrt(-metric_determinant) .* levi

    e3_lower = zeros(T,4)

    for l in 1:4
        for u in 1:4
            for v in 1:4
                for p in 1:4
                    e3_lower[p] = e3_lower[p] + levi[l,u,v,p] .* e0[l] .* e1[u] .* e2[v]
                end
            end
        end
    end

    e3 = local_inverse_metric * e3_lower

    e3 = e3 ./ sqrt(e3'local_metric*e3)

    a_array = collect(LinRange(0,1,N_x))
    b_array = collect(LinRange(0,1,N_y))

    alpha_h = angular_pixellation * N_x 
    #the vertical angle must be chosen st. tan(a_h)/tan(a_v) = N_x/N_y
    alpha_v = 2 * acot(N_x * cot(alpha_h/2)/N_y)
    
    meshgrid_a = ones(T,N_y) .* a_array'
    meshgrid_b = b_array .* ones(T,N_x)'

    preliminary_lower_momenta = Vector{MVector{4,T}}(undef,N_x*N_y)
    u0_raised = Vector{T}(undef,N_x*N_y)

    function local_root(α, fourmemonta)
        lmomenta = [fourmemonta[1], α*fourmemonta[2],α*fourmemonta[3],α*fourmemonta[4]]
        return -epsilon-(lmomenta'*local_metric*lmomenta)
    end

    for k in eachindex(meshgrid_a)
        a = meshgrid_a[k]
        b = meshgrid_b[k]
        C = sqrt(1 + (2b-1)^2 * tan(alpha_v/2)^2 + (2a-1)^2 * tan(alpha_h/2)^2 + epsilon)
        
        temp = C .* e0 - e1 - (2b-1) * tan(alpha_v/2) .* e2 - (2a-1) * tan(alpha_h/2) .* e3
        #horrible spaghetti
        temp = MVector{4,T}(temp)
        u0_raised[k] = one(T)
        
        
        temp[1] = one(T)

        f(x) = local_root(x,temp)

        α = fzero(f,one(T))
        
        temp = α*temp
        temp[1] = one(T)
        
        
        lowered_momenta = local_metric * temp
        
        
        preliminary_lower_momenta[k] = lowered_momenta
        
        
    end

    spatial_output = Vector{SVector{6,T}}(undef,N_x*N_y)

    #for our output, we treat the timelike momenta and positions separately (see EOM above)
    for k in eachindex(spatial_output)
        
        spatial_output[k] = SVector{6,T}([camera_fourpos[2:4]..., preliminary_lower_momenta[k][2:4]...])

    end

    #return ray initalizers and the rest
    return spatial_output, camera_fourpos[1], u0_raised

end

function approx_redshift_terminator(u0_function; critical_ratio, initial_u0)
    #a large raised u0 implies a large energy

    
    function condition(u, t, integrator)

        current_u0 = u0_function(u,t)

        ratio = current_u0/initial_u0

        return ratio > critical_ratio

    end

    return condition

end

function sphere_caster(metric_repr, spatial_integra, timelike_coordinate; epsilon, coordinate_transform)
    #coordinate transform should define the mapping from the assymptically flat metric coordinates to the cannonian minkowski coordinatess
    T = eltype(spatial_integra)
    local_fourpos = @SVector [timelike_coordinate,spatial_integra[1],spatial_integra[2],spatial_integra[3]]
    
    local_inverse_metric = metric_repr.numeric_inverse_metric(local_fourpos)

    f(x) = begin
        ex = SVector{4,T}([x, spatial_integra[4:6]...])
        innerprod = -epsilon-ex'*local_inverse_metric*ex
        return innerprod
    end

    u0_lower = fzero(f,one(T))
    u_lower = SVector{4,T}([u0_lower, spatial_integra[4:6]...])
    u_raised = local_inverse_metric*u_lower

    jac(x) = ForwardDiff.jacobian(coordinate_transform,x)

    l_jac = jac(local_fourpos)

    

    minkowski_fourveloc = l_jac*u_raised

    vt, vx, vy, vz = minkowski_fourveloc
    vr = sqrt(vx^2 + vy^2 + vz^2)

    θ = acos(vz/vr)
    ϕ = atan(vy, vx)

    return θ, ϕ

end

function spherical_to_rgb(θ, ϕ, img)
    
    θ = mod.(θ, π)
    ϕ = mod.(ϕ, 2π)

    
    img_height, img_width = size(img)

    
    x = @. ϕ / (2π) * img_width
    y = @. (θ - π) / π * img_height
    
    interp = extrapolate(interpolate(img, BSpline(Linear()), OnGrid()), Periodic())
    
    
    color = interp.(y, x)

    return color
end

export AnalyticMetric, NumericMetric, SymbolicIntegratorBuilder, camera_rays_generator, approx_redshift_terminator, sphere_caster, spherical_to_rgb
end
