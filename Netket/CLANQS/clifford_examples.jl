include("clifford_optimize.jl")


# -- simple example 1 -----------------------------------------------------------------------------

weights = [0.3, -0.3, 0.2, 0.1]
paulis = [P"ZZ_", P"_ZZ", P"-X_Z", P"__X"]

Ws, Ps, C = find_Clifford_for_plus_state(weights, paulis)

println(" -- example 1 --")
@show Ws;
@show Ps;
@show C;

##

N = 10
weights = Float64[]
paulis = PauliOperator[]

for n in 1:N-1
    push!(weights, 1.0)
    push!(paulis, PauliOperator(P"ZZ", N, [n,n+1]))
end

for n in 1:N
    push!(weights, 0.5)
    push!(paulis, PauliOperator(P"X", N, [n]))
end

Ws, Ps, C = find_Clifford_for_plus_state(weights, paulis)
println(" -- example 2 --")
C
