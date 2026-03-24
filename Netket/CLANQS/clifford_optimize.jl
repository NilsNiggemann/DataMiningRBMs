using QuantumClifford

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

function find_Clifford_for_plus_state(weights::Vector{<:Real}, paulis::Vector{<:PauliOperator})
    N_qubits = nqubits(paulis[1])
    Ws = copy(weights)
    Ps = copy(paulis)

    # make all weights non-negative
    for i in eachindex(Ws)
        if Ws[i]<0.0
            Ws[i] = -Ws[i]
            Ps[i] = -Ps[i]
        end
    end

    # sort in decending order
    perm = sortperm(Ws; rev=true)
    Ws = Ws[perm]
    Ps = Ps[perm]

    # building up the stabilizer state
    N_paulis = length(paulis)
    N_stabilizers = 0 
    i = 0
    psi = zero(Stabilizer, N_qubits)
    while i < N_paulis && N_stabilizers < N_qubits
        i += 1
        P = -Ps[i]
        if !comm(P, psi, N_stabilizers)
            continue
        end
        psi[N_stabilizers+1] = P
        canonicalize!(psi)
        if psi[N_stabilizers+1].xz != [0, 0]
            N_stabilizers += 1
        end
    end

    # "fill-in" orthogonal stabilizers
    psi = Stabilizer(tab(MixedDestabilizer(psi))[N_qubits+1:2N_qubits])
    canonicalize!(psi)

    # create Clifford operator that maps |+++> to |psi>
    perm = vcat(N_qubits+1:2N_qubits,1:N_qubits)
    tableau = tab(canonicalize!(MixedDestabilizer(psi)))[perm]
    C = CliffordOperator(tableau)
    Ps = [C*P for P in Ps]
    return Ws, Ps, C
end

function QuantumClifford.PauliOperator(p::PauliOperator, N::Int, sites::AbstractArray{Int,1})
    x = fill(false,N)
    z = fill(false,N)
    phase = p.phase[]
    for i in eachindex(sites)
        x[sites[i]] = p[i][1]
        z[sites[i]] = p[i][2]
    end
    return PauliOperator(phase, x, z)
end
