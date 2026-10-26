# Clifford preconditioning for gauge theories in Rydberg arrays

Status: idea / discussion note (2026-10-06). Nothing here has been tested yet.
Literature references are from memory; check them before citing.

---

## 1. The physics in plain words

**Rydberg arrays.** Atoms are held in a pattern of laser traps. Each atom is a
qubit: ground state or highly excited ("Rydberg") state. The key effect is the
**blockade**: two atoms that are close together cannot both be excited, because
the interaction energy between them is huge.

**Blockade = a local rule.** On suitable lattices, the blockade enforces a rule
at every small group of atoms, for example "around each vertex, exactly one atom
is excited". Draw each excited atom as a short bar (a *dimer*) on the link where
it sits, and every low-energy configuration becomes a dimer covering of a
lattice. The best-known case is the ruby lattice, where atoms sit on the links
of a kagome lattice (Verresen, Lukin, Vishwanath, PRX 2021; experiment:
Semeghini et al., Science 2021).

**Why "gauge theory".** A model in which a local rule holds at *every* site,
and the dynamics never breaks it, is called a lattice gauge theory. The rule
plays the role of Gauss's law in electromagnetism ("charge here = field lines
leaving here"). In the simplest version, the $\mathbb{Z}_2$ gauge theory, only
the **parity** matters: the number of dimers touching each vertex must be odd
(or even). In qubit language, that parity is a single Pauli string:

$$G_v = \prod_{i \text{ around } v} \sigma^z_i , \qquad G_v = \pm 1 .$$

The Rydberg experiments use this to prepare a quantum spin liquid. That is a
superposition of a huge number of dimer coverings, with a definite pattern of
amplitudes and signs.

## 2. Why this is hard for a neural network

1. **Most configurations are forbidden.** Only a tiny fraction of all bit
   strings obey the rule at every vertex. The network has to learn to put
   (almost) zero amplitude on all the others, and Monte Carlo sampling wastes
   time proposing configurations that break the rule.
2. **The allowed part is a delicate superposition.** A spin liquid spreads its
   weight over very many dimer coverings that are connected by collective
   moves of several atoms at once. This is the kind of structure NQS find hard
   to represent from a uniform starting point.
3. **The rules are known, but they are not used.** We know exactly which
   constraints the state must satisfy. A standard NQS still has to rediscover
   them from the energy.

## 3. What the Clifford transformation can do

**Key idea: turn each rule into a frozen qubit.**

Suppose each rule is a Pauli string $G_v$ that commutes with the Hamiltonian
(it is conserved). All the $G_v$ also commute with each other. That is exactly
the situation the clique construction is built for:

1. Put the $G_v$ into the list of candidate Pauli strings, with a large weight.
   This forces them into the chosen group. Adding $-\lambda \sum_v G_v$ to the
   Hamiltonian is harmless here: because $G_v$ is conserved, the term only picks
   which rule sector has the lowest energy.
2. The Clifford $U_C$ maps each independent $G_v$ onto a **single-site**
   $\sigma^x_j$ (step 4 of the appendix).
3. Every term of the rotated Hamiltonian still commutes with that $\sigma^x_j$.
   So on qubit $j$ it contains only $1$ or $\sigma^x_j$, and we can replace
   $\sigma^x_j$ by its fixed value $\pm 1$.
4. **Result: qubit $j$ drops out of the problem.** The neural network only
   sees the remaining qubits, and every configuration it samples automatically
   obeys all the rules.

So the Clifford does **automatic gauge fixing**:

- the number of variables shrinks by the number of independent rules (for a
  $\mathbb{Z}_2$ gauge theory on a square lattice, roughly by half);
- no sampling effort is wasted on forbidden configurations;
- the network starts from the best "rule-respecting" simple state, instead of
  from a uniform state that mostly violates the rules.

For the simplest models this is known by hand. For example, the pure
$\mathbb{Z}_2$ gauge theory maps onto a transverse-field Ising model (Wegner
duality). The point is that the clique/Clifford pipeline finds such a mapping
**automatically**, including for messy models where nobody has worked it out:
gauge theories with matter, extra longer-range terms, or effective models
derived from real Rydberg interactions.

## 4. Where it applies, and where it does not

| Model | Rule is a single Pauli string? | Rule exactly conserved? | Expected gain |
|---|---|---|---|
| $\mathbb{Z}_2$ gauge theory (toric code + field), with or without matter | yes | yes | exact elimination of constrained qubits |
| Effective dimer / $\mathbb{Z}_2$ model of the ruby Rydberg array | yes (vertex parity) | yes, within the effective model | same as above |
| Full Rydberg Hamiltonian ($\Omega$, $\Delta$, $V_{ij}$) | yes (vertex parity) | **no**, only approximately at low energy | constrained qubits become *nearly* frozen, not exactly |
| U(1) quantum link models (PXP in 1D, square-ice-type models) | **no**: the rule is a *sum* of $\sigma^z$, not a product | yes | only the parity part of the rule can be eliminated |

**Two caveats about the full Rydberg Hamiltonian:**

- **The rules have to be added by hand.** In the raw Rydberg Hamiltonian, the
  vertex parities $G_v$ are *not* terms of $\mathcal{H}$, so the clique search
  cannot find them on its own. My guess (not sure): run as is, the search mostly
  picks the large diagonal $\sigma^z\sigma^z$ terms from $V_{ij}$, which only
  selects a classical configuration. That is a trivial change of basis and gains
  little. This is quick to check numerically (see below).
- **Approximate rules are still useful.** The full Rydberg dynamics can break
  the vertex rule; the rule is only enforced energetically. The Clifford is
  still an exact change of basis, so nothing is lost: the eliminated qubits
  simply become "almost always $+$" instead of exactly $+$. That is still much
  simpler for the network than the original structure.

## 5. Suggested first tests

1. **Check the clique on the raw Rydberg Hamiltonian.** Run
   `find_Clifford_for_plus_state` on a small ruby or square Rydberg cluster and
   look at which terms are chosen. This settles the guess in Section 4.
2. **$\mathbb{Z}_2$ gauge theory with matter on a small square lattice**
   (the Fradkin–Shenker model).
   - Add the $G_v$ by hand and compare the number of frozen qubits with the
     number of independent rules.
   - Compare NQS energy errors with and without the Clifford against exact
     diagonalization.
   - The phase diagram is known, so this is a clean benchmark.
3. **Effective ruby model, then the full Rydberg Hamiltonian.**
   - Measure how close to frozen the eliminated qubits are, i.e.
     $\langle \sigma^x_j \rangle$ in the rotated frame.
   - Compare with the DMRG results of Verresen et al.
4. **Observables.** The loop operators measured in the experiment are products
   of Pauli operators along a closed path. I think the off-diagonal loop is only
   approximately a product of single-site $X$'s (not sure). Under the Clifford,
   a Pauli string stays a single Pauli string, so these loops remain cheap to
   evaluate.

## 6. Side case: pyrochlore quantum spin ice

**Question.** Can the ice rule ("2-in-2-out" on every tetrahedron) be absorbed
into a Clifford circuit acting on $\ket{+}$, as in Section 3?

**Short answer.** Only partly. The full ice rule cannot be absorbed by any
Clifford. Its parity part can, but exactly only in the effective ring-exchange
model, not in the microscopic XXZ Hamiltonian.

### Why the full ice rule cannot be absorbed

- **Dimension argument (exact).** A set of frozen qubits in $\ket{+}$ plus a
  set of free qubits spans a subspace of dimension $2^k$. A Clifford preserves
  dimension, so it can only map the rule-respecting states onto this form if
  there are exactly $2^k$ of them (i.e. they form a stabilizer code space).
  A single tetrahedron has $\binom{4}{2} = 6$ ice states, which is not a power
  of 2. So no Clifford can do it.
- **Root cause.** The ice rule is a *sum*, $\sum_{i\in t} S^z_i = 0$ (a U(1)
  rule). Only rules that are a *product* of Pauli operators can be frozen out.
  This is the "U(1) quantum link model" row in the table of Section 4.
- **Larger clusters.** The number of ice states on finite clusters (the
  Pauling count) is generally not a power of 2 either (expected, not checked
  for every cluster).

### What can be absorbed: the parity

The tetrahedron parity $G_t = \prod_{i\in t}\sigma^z_i$ is a single Pauli
string. Even parity allows 0, 2 or 4 up spins: 8 states per tetrahedron, a
power of 2.

- **Effective ring-exchange model**
  $-J_\text{ring}\sum_\text{hex}(S^+S^-S^+S^-S^+S^- + \text{h.c.})$:
  every $G_t$ is exactly conserved. One qubit per tetrahedron can be frozen.
  With $N/2$ tetrahedra and one global relation between them, the problem
  shrinks from roughly $N$ to $N/2$ qubits.
- **Microscopic XXZ Hamiltonian:** the parities are *not* conserved.
  - $J_\pm S^+_i S^-_j$ keeps the parity of the tetrahedron containing the bond
    $ij$.
  - But $i$ and $j$ each also belong to a second, different tetrahedron, and
    both of those flip parity. This is exactly how a pair of monopoles is
    created.
  - So the parities are only approximately conserved for
    $J_\pm \ll J_{zz}$. The frozen qubits become *nearly* frozen, as for the
    full Rydberg Hamiltonian.
- **What parity leaves for the network.** Fixing parity removes the
  3-in-1-out tetrahedra (monopoles), which are the cheapest defects. All-in /
  all-out tetrahedra (0 or 4 up) still have even parity, so the network has to
  suppress those itself. That is mild, because their $J_{zz}$ cost is about 4
  times that of a monopole.

### What the current code would do on bare XXZ

- All $Z_iZ_j$ terms commute, so they form a large clique. But on a
  tetrahedron, all six $Z_iZ_j = -1$ is impossible.
- This is the sign-consistency caveat of the appendix (step 4): **geometric
  frustration shows up as an inconsistent clique.**
- Guess (not sure): the search ends up on a classical ice configuration. The
  Clifford is then just a relabelling of bits and absorbs nothing.
- So, as for Rydberg arrays, the parities $G_t$ must be added by hand, or the
  method must be applied to the effective ring-exchange model.

### The remaining 6-out-of-8 constraint

Excluding all-in / all-out is not a stabilizer rule. It has to come from the
network or the sampler. Two options:

- **Constrained sampling:** only propose moves that preserve the ice rule
  (loop or hexagon moves).
- **Gauge-equivariant networks:** for example Luo et al., PRL 2021 (fairly
  confident about this reference).

Both combine naturally with Clifford parity-fixing.

### Side issue: π-flux spin ice ($J_\pm < 0$)

The sign of $J_\text{ring}$ corresponds to a physical background flux (0 or
π through each hexagon). It is not a gauge choice, so I do not expect any
Clifford to remove it (not sure). If that is right, the sign problem in the
π-flux case survives the parity-fixing.

## 7. Open questions

- **Symmetry vs. simplicity.** The Clifford can make the remaining terms
  non-local and break translation symmetry. A good choice of *which* qubit
  absorbs each rule may keep things local (e.g. one qubit per vertex, chosen in
  a translation-invariant way). The current code makes this choice
  automatically, not symmetrically.
- **Dynamics.** In the experiment the spin liquid is prepared by a slow ramp,
  not as a ground state. Time-dependent VMC in a gauge-fixed frame could be more
  stable, because the network can no longer leak into rule-breaking
  configurations.
- **Matter and defects.** If the rules are broken on purpose (monomers =
  charges), the frozen qubits become dynamical again. How gracefully the method
  degrades as the cost of a defect decreases is an interesting question in its
  own right.
