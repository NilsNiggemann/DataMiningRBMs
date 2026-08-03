using QuantumClifford
using JuMP, HiGHS

function Base.:*(op::CliffordOperator, p::PauliOperator)
    s = op * Stabilizer([p])
    return s[1]
end

function QuantumClifford.comm(p::PauliOperator, s::Stabilizer, N_stabilizers::Int)
    for i in 1:N_stabilizers
        if comm(p,s[i]) != 0 
            return false
        end
    end
    return true
end

function max_weight_clique(adj_matrix, weights)
    n = length(weights)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, x[1:n], Bin)

    # Non-edge constraints
    for i in 1:n, j in i+1:n
        if adj_matrix[i, j] == 0
            @constraint(model, x[i] + x[j] <= 1)
        end
    end

    @objective(model, Max, sum(weights[i] * x[i] for i in 1:n))

    op = optimize!(model)
    # display(model)
    # display(termination_status(model))
    clique = [i for i in 1:n if value(x[i]) > 0.5]
    return clique
end


function find_Clifford_for_plus_state(weights::Vector{<:Real}, paulis::Vector{<:PauliOperator})
    Ws = copy(weights)
    Ps = copy(paulis)

    # make all weights non-negative
    for i in eachindex(Ps)
        if Ws[i]<0.0
            Ws[i] = -Ws[i]
            Ps[i] = -Ps[i]
        end
    end

    # sort in decending order
    perm = sortperm(Ws; rev=true)
    Ws = Ws[perm]
    Ps = Ps[perm]

    return _find_clifford_sorted_weights(Ws, Ps; clique_objective_weights = Ws, target_basis = :plus)
end



"""
    _find_clifford_sorted_weights(Ws, Ps; clique_objective_weights=Ws, target_basis=:plus)

`target_basis = :plus`          -> clique terms become X-strings; C|+...+> = |psi>
`target_basis = :computational` -> clique terms become Z-strings; C|0...0> = |psi>
"""
function _find_clifford_sorted_weights(Ws, Ps; clique_objective_weights = Ws,
                                        target_basis::Symbol = :plus)
    N_qubits = nqubits(Ps[1])

    adj_matrix = fill(false, length(Ps), length(Ps))
    for i in eachindex(Ps)
        adj_matrix[i, i] = true
        for j in i+1:length(Ps)
            c = comm(Ps[i], Ps[j]) == 0
            adj_matrix[i, j] = c
            adj_matrix[j, i] = c
        end
    end

    clique = max_weight_clique(adj_matrix, clique_objective_weights)

    psi_long = canonicalize!(Stabilizer(-Ps[clique]))
    psi = psi_long[1:min(length(psi_long), N_qubits)]

    psi = Stabilizer(tab(MixedDestabilizer(psi))[N_qubits+1:2N_qubits])
    canonicalize!(psi)

    full_tab = tab(canonicalize!(MixedDestabilizer(psi)))

    if target_basis == :plus
        # swap stabilizers/destabilizers -> C|+...+> = |psi>, clique -> X-strings
        perm = vcat(N_qubits+1:2N_qubits, 1:N_qubits)
        tableau = full_tab[perm]
    elseif target_basis == :computational
        # standard order -> C|0...0> = |psi>, clique -> Z-strings (diagonal!)
        tableau = full_tab
    else
        error("Unknown target_basis: $target_basis")
    end

    C = CliffordOperator(tableau)
    Cinv = inv(C)
    Ps = [Cinv * P for P in Ps]

    return Ws, Ps, C
end

function is_offdiag(P::PauliOperator)
    for i in 1:nqubits(P)
        if P[i] == X[1] || P[i] == Y[1]
            return true
        end
    end
    return false
end

function sign(P::PauliOperator)
    phase_Int = Int(P.phase[])
    if phase_Int == 0
        return 1.0
    elseif phase_Int == 2
        return -1.0
    else
        error("Pauli operator has non-real phase, which is not supported.")
    end
end

function find_Clifford_for_sign_problem(weights::Vector{<:Real}, paulis::Vector{<:PauliOperator})
    Ws = copy(weights)
    Ps = copy(paulis)

    # make all weights non-negative
    for i in eachindex(Ps)
        if Ws[i]<0.0
            Ws[i] = -Ws[i]
            Ps[i] = -Ps[i]
        end
    end
    heuristic_weights = [(is_offdiag(P)&& sign(P) == 1.0) * w for (P,w) in zip(Ps,Ws)]
    # sort in decending order
    perm = sortperm(heuristic_weights; rev=true)
    Ws = Ws[perm]
    Ps = Ps[perm]

    return _find_clifford_sorted_weights(Ws, Ps)
end