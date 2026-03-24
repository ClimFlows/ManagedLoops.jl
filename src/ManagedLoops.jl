module ManagedLoops

export @with, @unroll, offload, configure, no_simd
export LoopManager, HostManager, DeviceManager

"""
    offload(fun1, mgr::LoopManager, range, args...)
    offload(fun2, mgr::LoopManager, (irange, jrange), args...)

Given a function performing a loop / loop nest:
```
    function fun1(range, args...)
        # do some computation shared by all indices i
        for i irange
            # do some work at index i
        end
        return Nothing
    end

    function fun2((irange, jrange), args...)
        # do some computation shared by all indices i,j
        for j in jrange
            # do some computation shared by all indices i
            for i irange
                # do some work at index i
            end
        end
        return Nothing
    end
```
Executes the loop nest with the provided manager. Depending on the manager, `fun` may be called just once on the full range,
several times on sub-ranges, or many times on 'ranges' consisting of a single index.

!!! warning
    Each call to fun() must be independent, and several calls may occur concurrently.
    The only guarantee is that the whole iteration space is covered without overlap.

!!! note
    `offload` *may* amortize the cost of a pre-computation
    whose results are reused across iterations, as in the above example.
    This depends on how `offload` is implemented by the manager.

"""
@inline offload(fun, ::Nothing, args...) = fun(args...)


"""
    mgr_conf = configure(mgr, config)
Returns a manager similar to `mgr` but with some extra information provided by `config`.
The default behavior is to return `mgr` unchanged.
This function is meant to be specialized by packages providing loop managers.
"""
configure(mgr, config) = mgr

# default manager
"""
    mgr = default_manager()
    mgr = default_manager(ManagerType) :: ManagerType
    # examples
    host = default_manager(HostManager)
    device = default_manager(DeviceManager)

Returns a manager of the desired type, if provided. ManagedLoops implements only :

    default_manager() = default_manager(HostManager)

`default_manager(::Type{HostManager}) is defined if `LoopManagers` is loaded.
`default_manager` is also meant to be specialized by the user, for instance :
    ManagedLoops.default_manager(::Type{HostManager}) = LoopManagers.Vectorized_CPU()
"""
default_manager() = default_manager(HostManager)

# abstract managers
"""
    abstract type LoopManager end

Ancestor of types describing a loop manager.
"""
abstract type LoopManager end

"""
    abstract type DeviceManager <: LoopManager end

Ancestor of types describing a loop manager running on a device (GPU).
"""
abstract type DeviceManager <: LoopManager end

"""
    abstract type HostManager <: LoopManager end

Ancestor of types describing a loop manager running on the host.
"""
abstract type HostManager <: LoopManager end


# Sometimes we need to deactivate SIMD on loops that would not work with it
"""
    mgr_nosimd = no_simd(mgr::LoopManager)
Returns a manager similar to `mgr` but with SIMD disabled.

!!! tip
    Due to implementation details, not all loops support SIMD.
    If errors are thrown when offloading a loop
    on an SIMD-enabled manager, use this function.
"""
no_simd(mgr::LoopManager) = mgr

"""
    synchronize(mgr)
Waits until ongoing computations on `mgr` work complete. Should be specialized for GPU managers.
"""
synchronize(::LoopManager) = nothing

# Fallback implementation for flag::Bool, ensures that (@vec flag ? a : b) == (flag ? a : b) 
# The added-value method is defined in LoopManagers, for flag::SIMD.Vec{Bool}
Base.@propagate_inbounds choose(flag::Bool, iftrue, iffalse) = flag ? iftrue() : iffalse()

# parallel, barrier, master, share
include("julia/parallel.jl")

# API for wrapped managers and arrays
include("julia/wrapped.jl")

module _internals_

using ..ManagedLoops: LoopManager, offload, choose
using MacroTools

include("julia/at_unroll.jl")
include("julia/at_loops.jl")
include("julia/at_vec.jl")
include("julia/broadcast.jl")

end # internals

using ._internals_: @vec, @unroll, @loops, @with, bulk, tail

using PackageExtensionCompat
function __init__()
    @require_extensions
end

end # module ManagedLoops
