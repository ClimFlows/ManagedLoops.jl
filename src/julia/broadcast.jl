# support for syntax : @. mgr[a] = b + c
# which is semantically equivalent to:
#     @. a = b + c
# if a is an array, otherwise to:
#     @. b + c

#====================== array lhs ======================#

struct ManagedArray{A,M,T,N} <: AbstractArray{T,N}
    mgr::M
    a::A
    ManagedArray(mgr::M, a::A) where {M,A}= new{A, M, eltype(a), ndims(a)}(mgr, a)
end
Base.getindex(mgr::LoopManager, a::AbstractArray) = ManagedArray(mgr, a)
Base.ndims(::Type{ManagedArray{A}}) where A = ndims(A)
Base.size(ma::ManagedArray) = size(ma.a)

@inline function Broadcast.materialize!(lhs::ManagedArray, bc::Broadcast.Broadcasted)
    bc = Broadcast.instantiate(bc)
    T = Broadcast.combine_eltypes(bc.f, bc.args)
    managed_copyto!(lhs.mgr, lhs.a, bc, axes(bc)...)
    return lhs.a
end

function Base.copyto!(ma::ManagedArray, bc::Broadcast.Broadcasted)
    managed_copyto!(ma.mgr, ma.a, bc, axes(bc)...)
    return ma.a
end

#====================== void lhs ======================#

struct ManagedOther{Other,M}
    mgr::M
    other::Other
end
Base.getindex(mgr::LoopManager, other) = ManagedOther(mgr, other)

@inline function Broadcast.materialize!(lhs::ManagedOther, bc::Broadcast.Broadcasted)
    bc = Broadcast.instantiate(bc)
    T = Broadcast.combine_eltypes(bc.f, bc.args)
    a = similar(bc, T)
    managed_copyto!(lhs.mgr, a, bc, axes(bc)...)
    return a
end

@loops function managed_copyto!(_, a, bc, ax1)
    let irange = ax1
        @vec for i in irange
            @inbounds a[i] = bc[i]
        end
    end
end

#====================== managed_copyto! ======================#

@loops function managed_copyto!(_, a, bc, ax1, ax2)
    let (irange, jrange) = (ax1, ax2)
        for j in jrange
            @vec for i in irange
                @inbounds a[i,j] = bc[i,j]
            end
        end
    end
end

@loops function managed_copyto!(_, a, bc, ax1, ax2, ax3)
    let (irange, jrange, krange) = (ax1, ax2, ax3)
        for j in jrange, k in krange
            @vec for i in irange
                @inbounds a[i,j,k] = bc[i,j,k]
            end
        end
    end
end

@loops function managed_copyto!(_, a, bc, ax1, ax2, ax3, ax4)
    let (irange, jrange, krange, lrange) = (ax1, ax2, ax3, ax4)
        for j in jrange, k in krange, l in lrange
            @vec for i in irange
                @inbounds a[i,j,k,l] = bc[i,j,k,l]
            end
        end
    end
end
