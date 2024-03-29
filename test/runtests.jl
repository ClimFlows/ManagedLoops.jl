using ManagedLoops: @loops, @vec, @unroll
using Test

@loops function test1!(_, a, b, c)
    let irange = eachindex(a, b, c)
        @vec for i in irange
            a[i] = @unroll sum( c[i]^n for n in 1:4)
            b[i] = c[i] + c[i]^2 + c[i]^3 + c[i]^4
        end
    end
end

@loops function test2!(_, a, b, c)
    let (irange, jrange) = axes(c)
        @unroll @vec for i in irange, j in jrange
            tp = ( c[i,j]^n for n in 1:4)
            a[i,j] = sum(tp)
            b[i,j] = c[i,j] + c[i,j]^2 + c[i,j]^3 + c[i,j]^4
        end
    end
end

function check(fun!, c)
    a, b = similar(c), similar(c)
    fun!(nothing, a, b, c)
    return a ≈ b
end

@testset "Macros" begin
    @test check(test1!, randn(100))
    @test check(test2!, randn(100, 100))
end
