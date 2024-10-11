module numeric_backend

using LinearAlgebra

function permutation_sign(perm::AbstractVector{T}) where T<:Real
    #only works for unique permutations
    L = length(perm)
    crosses = 0
    for i = 1:L
        for j = i+1 : L
            crosses += perm[j] < perm[i]
        end
    end
    return iseven(crosses) ? one(T) : -one(T)
end

#this will be run once: performance is irrelevant.
function levi_civita_generator(dtype::Type{<:Real})
    outp = zeros(dtype,(4,4,4,4))
    for d in 1:4
        
        for c in 1:4
            
            for b in 1:4
                
                for a in 1:4

                    number_uniq = length(unique([a,b,c,d]))
                    
                    if number_uniq == 4
                        
                        outp[a,b,c,d] = permutation_sign([a,b,c,d])
                    end

                end
            end
        end
    end
    return outp
end

export levi_civita_generator
end
