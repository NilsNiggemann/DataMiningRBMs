# Plan: quenches that are hard for MPS-TDVP but tractable for NQS (Rydberg & related platforms)

**Goal.** Find quenches where the entanglement grows fast (bad for MPS) while the RBM+TDVP
infidelity stays small, using the cumulant proxies μ, μ_ph as the screening tool. Two targets:

1. **Near-classical target.** H_f ≈ diagonal in the σ^z basis plus a small off-diagonal term
   (Rydberg-like: strong Ising interactions, weak drive Ω). Entanglement grows through the phase,
   which is (nearly) two-body.
2. **Gold standard.** μ and μ_ph stay low, S grows, *and* H_f is not close to classical.
   The report must contain this case explicitly, even if the answer is negative.

**Prioritisation principle (user, 2026-10-26).** Prefer quenches that matter outside this study:
- realised in an experiment;
- an established benchmark for classical simulation (tensor networks, NQS);
- proposed in prominent theory work.
Every family carries a priority tier (§3). Tier 1 runs first and gets the densest grids. The report
motivates each studied family in a short paragraph, with citations, explaining why it is
interesting.

Builds on `PLAN_quench_family.md` / `report/quench2d_report.pdf`. Results carried over:
- μ orders RBM failure times well (Spearman 0.91 / 0.99) and carries information beyond S
  (partial ρ(μ|S) = 0.81 vs ρ(S|μ) = 0.32).
- μ saturates at ≈ −5, so it cannot rank already-hard quenches; μ_ph and w_hi resolve that regime.
- The relative threshold μ₀ + 5 depends on the initial state, so **use an absolute threshold**
  (μ > −10, μ_ph > −10) for t*.
- α=2 helps easy quenches by 1.2–4×, but not hard ones.

Decisions (2026-10-26, user):
- **MPS hardness** is judged from entanglement proxies at ED sizes plus scaling arguments, not from
  MPS runs.
- **Lattices:** geometry-specific clusters with ≤ 16 sites: chain, square, triangular, kagome,
  honeycomb, and a ruby patch. The 4×3 cylinder stays the baseline.
- **Initial states:** |+x⟩, tilted product |θ⟩, ground states, and tilted Z2/Néel.
- **Headline proxies:** μ (from log|ψ|) and μ_ph (from arg ψ), shown side by side.

---

## 0. Premises checked before planning (affect the design)

- **At Ω = 0, μ is degenerate.** A diagonal H on a product state only changes phases, so
  log|ψ(s,t)| = log|ψ₀(s)|, which is at most one-body for product states. Then μ = −∞ (exactly),
  and all the difficulty is in the phase. For Ω = 0, arg ψ = −t·E_cl(s) + const, which is exactly
  ≤ 2-body for Ising (≤ k-body for k-spin terms), so μ_ph is also trivially bounded. The
  interesting physics is the **Ω → 0⁺ limit**: how fast μ, μ_ph switch on with Ω.
- **"Two-body phase ⇒ easy for RBM" is only partly true.** An exact RBM representation of
  exp(−iφ_ij Z_iZ_j) needs about one hidden unit per bond with non-trivial φ_ij = V_ij t mod π.
  - For NN couplings this is about z/2 hidden units per site (α ≈ 2 on the square lattice).
  - For 1/r⁶ the bond at distance 2a matters once t·V_nn ≳ 64·π/4 ≈ 50.
  - For 1/r³ distant bonds matter by t·J_nn ≳ 8.
  - A **Jastrow** factor log ψ = Σ W_ij s_i s_j is exact at Ω = 0. It is the natural baseline,
    and RBM×Jastrow the natural ansatz.
- **Zero-field Ising entanglement is bounded** by the crossing bonds: S ≤ Σ_{ij ∈ cut} min(1, ·)·log 2.
  - With NN couplings S(t) is periodic and area-law, so MPS-easy in 1D and exp(L_y) in 2D.
  - Long-range tails make S grow slowly, roughly t^{d/6} for vdW in 1D (not sure; to be checked).
  - So "MPS-hard at Ω = 0" requires 2D and/or long times. The plan measures it rather than
    assuming it.
- **Exact zeros.** Rydberg-native starts (|g…g⟩, Z2 crystal) and blockade-projected dynamics
  have ψ(s) = 0 on many s, where log|ψ| is singular. Use tilted product states (θ ∈ (0, π), so
  log|ψ₀| is finite and one-body) and finite V instead of hard projection. Report `minlog` always.
- **Stiffness.** With V_nn/Ω up to 100, phases wind at a rate of about V. Choose dt·V_max ≤ 0.2
  for μ_ph unwrapping (check by halving dt). Also compute an **interaction-picture phase**
  ψ̃ = e^{+iH_cl t}ψ, which isolates the non-classical part and removes trivial winding.
- **Time reversal.** For real H and real ψ₀, evolving with −H gives ψ*, so |ψ| and S are identical.
  The sign of the interaction (attractive or repulsive) is redundant for μ and S. Do not scan it.

## 1. Conventions

- Rydberg occupation: n_i = (1 − Z_i)/2, with |g⟩ ↔ Z = +1.
  - n_i n_j = (1 − Z_i − Z_j + Z_iZ_j)/4, so a 1/r^α coupling also generates site-dependent
    longitudinal fields, which matter at open boundaries. Keep them; do not drop them.
- Energy unit: the largest nearest-neighbour coupling, V_nn = C₆/a⁶ = 1 or J_nn = 1.
  - Time grid t ∈ [0, t_max] with t_max = max(10, 3/Ω).
  - Screening uses ≤ 201 time points (non-uniform if needed), with dt chosen for unwrapping.
- Lattices and site positions (minimum image for periodic directions):

  | name | N | BC | use |
  |---|---|---|---|
  | `chain12`, `chain16` | 12, 16 | OBC / PBC | 1D reference, kicked Ising |
  | `sq4x3` | 12 | PBC-x, OBC-y | baseline (previous study) |
  | `sq4x4` | 16 | PBC | size check |
  | `tri4x3`, `tri4x4` | 12, 16 | PBC | frustrated Ising / XY |
  | `kag12` | 12 | PBC 2×2 cells | spin-liquid-relevant XY, blockade |
  | `hc12` | 12 | PBC 2×3 cells | honeycomb (Rydberg honeycomb arrays) |
  | `ruby12` | 12 | OBC patch | Semeghini-type geometry (tiny; qualitative only) |
  | `tc12` | 12 | 2×3 torus links | toric-code family S1 |

## 2. Hamiltonian catalogue

Notation:
- Ω is the transverse drive (Ω/2 Σ X), Δ the detuning (−Δ Σ n), and V(r) the pair potential.
- g is the control parameter of the μ(g, t) map.
- Init codes: `+x`, `θ` (tilted product, θ = π/4 default), `Z2θ` (Néel/checkerboard tilted by θ),
  `gs(...)` (ground state).
- "Expected" is my prior. It is a hypothesis, not a result.

### R — Rydberg van der Waals Ising, H = Ω/2 ΣX − Δ Σn + Σ_{i<j} V(r_ij) n_i n_j, V = 1/r⁶

| ID | control g (range) | fixed | init | lattices | why / expected |
|---|---|---|---|---|---|
| R0 | tilt θ ∈ [0.1, π/2] | Ω = 0, Δ = 0 | θ | chain16, sq4x4, tri4x4, kag12 | Exactly classical reference. μ = −∞ exactly, μ_ph two-body. Measures S(t), which is bounded, and the hidden units an RBM needs vs Jastrow |
| R1 | Ω ∈ [0, 0.5] (12 pts, log-spaced near 0) | Δ = 0 | +x | sq4x3, sq4x4, tri4x4, kag12 | **Central near-classical family.** Expected: t*_μ ∝ 1/Ω² (perturbative), while S grows from the Ising phase at all Ω |
| R2 | Ω ∈ [0, 0.5] | Δ = 0 | θ = π/4, Z2θ | sq4x3, chain12 | As R1, with non-uniform amplitudes. Tests whether μ₀ ≠ −∞ changes the onset |
| R3 | Δ ∈ [−1, 3] (31 pts) | Ω = 0.1 | +x, θ | sq4x3, chain12 | **Facilitation / anti-blockade resonances** at Δ = V_nn, 2V_nn, V(√2a), … Expected: μ, S peaks at the resonances and near-classical off resonance. A sharp g-structure like family B |
| R4 | R_b/a ∈ [0.8, 2.6] (C₆ scan) | Ω = 1, Δ = 0 | θ small (0.2), Z2θ | sq4x4, kag12, ruby12 | Blockade crossover, not near-classical. Includes the Semeghini regime R_b/a ≈ 2.4 (ruby) |
| R5 | Δ/Ω ∈ [−2, 2] | V_nn = 30 Ω (soft blockade ≈ PXP) | Z2θ, θ small | chain16, sq4x4 | PXP scars from Z2 (slow S, revivals) vs fast thermalisation from |g⟩. MPS-easy reference; asks whether μ sees the scars |
| R6 | dressing R_c/a ∈ [0.5, 3] | U(r) = U₀/(1+(r/R_c)⁶), Ω = 0.05 | +x | sq4x4, tri4x4 | Soft-core dressing. R_c → ∞ gives one-axis twisting, where S ≤ log(N/2+1) (MPS-easy). Intermediate R_c gives a plateau-shaped long-range phase |
| R7 | position disorder σ/a ∈ [0, 0.2] | Ω = 0.1 | +x | sq4x4, chain16 | Tweezer disorder dephases the Ising phase. Expected: log-like S growth (MBL-like) and RBM-easy |
| R8 | Ω ∈ [0, 0.5] | random Δ_i ∈ [−W, W], W = 2 | +x | chain16, sq4x4 | Near-classical with random fields (l-bit regime). S ~ log t. Tests μ when the phase is quasi-local |
| R9 | inter/intra-species ratio ∈ [0, 1] | dual-species array, Ω = 0.1 | +x | sq4x4 (bipartite) | Dual-species (e.g. Rb/Cs) arrays: interspecies-only Ising is bipartite. Frustration switched by g |

### I — generic long-range Ising (ions, polar molecules, magnetic atoms), H = Σ J/r^α Z_iZ_j + h_x ΣX + h_z ΣZ

| ID | control g | fixed | init | lattices | why / expected |
|---|---|---|---|---|---|
| I1 | α ∈ [0.5, 6] (10 pts) | h_x = 0.1, h_z = 0 | +x | chain16, sq4x4 | Interpolates ions (α ≲ 3), dipolar (3) and vdW (6). Small α behaves like OAT (MPS-easy); intermediate α gives the fastest long-range entanglement |
| I2 | h_x ∈ [0, 1] | α = 3 (dipolar Ising) | +x, θ | sq4x4, tri4x4 | Near-classical dipolar Ising. Distant bonds matter at t ≳ 8, which tests RBM capacity vs Jastrow |
| I3 | h_z ∈ [0, 1] | NN, h_x = 0.25 | gs(h_x = 4) | chain16 | Confinement (mesons, slow S). Control case where both MPS and RBM should be easy |
| I4 | h_x ∈ [0, 1] | NN | θ ∈ {π/4, π/8} | sq4x3, tri4x3 | Near-classical NN Ising with non-trivial amplitudes. Frustrated vs bipartite |
| I5 | h_x ∈ [0, 0.5] | 3-body Z_iZ_jZ_k (triangles) / 4-body plaquette | +x | tri4x3, sq4x4 | Classical k-body phase (Rydberg LGT / Xu–Moore). Tests whether RBMs need more units for 3- and 4-body phases, and whether μ_ph sees cluster order |

### D — dipolar exchange (resonant nS–nP Rydberg pairs, molecules), H = Σ J_ij (σ⁺σ⁻ + h.c.) + …, J_ij = J(1 − 3cos²Θ_ij)/r³

| ID | control g | fixed | init | lattices | why / expected |
|---|---|---|---|---|---|
| D1 | quantisation-axis angle Θ ∈ [0, π/2] | none | +x | sq4x4, tri4x4 | Dipolar XY (spin squeezing / continuous symmetry breaking experiments: Bornet et al. 2023, Chen et al. 2023, not sure of exact refs). Not classical. Early times behave like OAT. Gold-standard candidate if S grows while μ stays low |
| D2 | h_x ∈ [0, 1] | Θ = 0 | +x, θ | sq4x4 | U(1)-breaking field on dipolar XY |
| D3 | J_z/J_xy ∈ [−2, 2] (21 pts) | Floquet-engineered dipolar XXZ | +x, Z2θ | sq4x4, chain16 | **Connects classical (J_xy → 0, dipolar Ising) to XY.** Contains the Heisenberg point (|+x⟩ is an eigenstate there: trivial) and near-Ising points (near-classical target). Microwave-engineered XXZ (Scholl et al. 2022; Geier et al. 2021, not sure) |
| D4 | h_x or J_z ∈ [0, 1] | dipolar XY | +x, Z2θ, 120° product | kag12, tri4x4 | Frustrated dipolar XY, spin-liquid relevant: Bintz et al., "Dirac spin liquid in quantum dipole arrays" (paper confirmed by user; verify the lattice and parameters it uses and run that lattice first) |
| D5 | ratio λ = V_zz/J_xy ∈ [0, 3] | XY 1/r³ + Ising 1/r⁶ | +x | sq4x4 | Mixed exchange and vdW, as for two Rydberg states with both couplings present |
| D6 | filling f ∈ [0.5, 1] | dipolar XY on randomly diluted sites | +x | sq4x4 | Molecule/tweezer filling disorder. Glassy, slow dynamics |

### C — constrained / kinetic dynamics

| ID | control g | fixed | init | lattices | why / expected |
|---|---|---|---|---|---|
| C1 | Δ = V_nn + δ, δ ∈ [−0.3, 0.3] | Ω = 0.1, V = 1/r⁶ | single seed excitation (tilted) and +x | chain16, sq4x4 | Facilitation: a spin flips only next to an excited neighbour. Avalanche-like spreading. Non-classical on resonance, near-classical off resonance |
| C2 | Δ/Ω | PXP + Floquet modulation | Z2θ | chain16 | Driven scars (Bluvstein et al. 2021, not sure) |
| C3 | s ∈ [−1, 1] (tilt of the East constraint) | quantum East model | θ | chain16 | Kinetically constrained, slow S. Reference |

### S — commuting / Clifford-like / basis-rotated (**gold-standard candidates**)

| ID | control g | fixed | init | lattices | why / expected |
|---|---|---|---|---|---|
| S1 | ε ∈ [0, 0.5] (perturbation h_x ΣX or h_z ΣZ) | toric code −ΣA_v − ΣB_p | θ, +x | tc12 | Commuting but not classical in σ^z. log ψ is a sum of local (≤ 4-body plaquette) terms, so μ, μ_ph should be bounded while S grows to the area-law value. Ruby Rydberg spin liquid is toric-code-like. ε turns on genericity |
| S2 | ε ∈ [0, 0.5] | cluster H = Σ Z X Z (1D) / Σ X Π Z (2D) + ε Σ X | θ | chain12, sq4x3 | H = U_CZ (ΣX) U_CZ† is locally equivalent to a trivial evolution. Fixed-depth-circuit states give bounded but non-zero S (exp(L_y) for 2D MPS). μ expected low. Clean test of "non-classical H, low μ" |
| S3 | longitudinal kick h ∈ [0, π/4] | kicked Ising at J = b = π/4 (dual-unitary) | +x, θ, \|g⟩ | chain12, chain16, sq4x3 | **Prime gold candidate.** At h = 0 the states are stabilizer states (uniform \|ψ\| on an affine support, quadratic phases), with **maximal entanglement velocity**. For h ≠ 0 the dynamics are dual-unitary chaotic. μ(h, t) maps "how far from Clifford". Check zeros on the support (\|g⟩ start may have them; use θ) |
| S4 | detuning ε of J = π/4 + ε | KIM, h = 0 | +x | chain16 | Second axis: leave the Clifford point in a different direction |
| S5 | none (reference) | NN XY chain (free fermions) | +x, Néel θ | chain16 | Gaussian states, volume-law S. Measures how much μ grows for an integrable, non-classical quench |
| S6 | Ω ∈ [0, 0.5] | Ising in the X basis, Σ X_iX_j + Ω Σ Z | Z-tilted θ | sq4x3 | Classical in a rotated basis, non-classical for the σ^z RBM. Tests the basis dependence of both μ and RBM hardness |
| S7 | Δ_zz ∈ [0, 1] | NN XXZ (integrable 1D) | Néel θ | chain16 | Bethe-integrable reference |
| S8 | measurement-free "magic" doping p ∈ [0, 1] | brickwork Clifford circuit with T gates at rate p | +x | chain12 | Optional. Direct Clifford-to-generic interpolation; compare μ with the stabilizer Rényi entropy (magic) |

### G — ground-state starts (Rydberg-native analogues of family A/B)

| ID | control g | fixed | init | lattices | why / expected |
|---|---|---|---|---|---|
| G1 | Δ_f ∈ [−1, 5] | Ω = 1, R_b/a = 1.2 | gs(Δ_i = −2) (disordered) | sq4x4, kag12 | Quench across the checkerboard / Z2 transition. Rydberg analogue of family A |
| G2 | Ω_f ∈ [0, 0.5] | Δ = 3 | gs in checkerboard phase (cat) | sq4x4 | Near-classical evolution from an ordered state |
| G3 | J_z/J_xy ∈ [−1, 1] | dipolar | gs of dipolar XY ferromagnet | sq4x4 | Quench out of the continuous-symmetry-broken state |
| G4 | h_f ∈ [0, 4] | Ising 1/r⁶ (Rydberg TFI-like) | gs(h_i = 4) | sq4x3 | Direct bridge to family A with Rydberg tails |

### Q — alternating near-classical evolution (QAOA-type)

| ID | control g | fixed | init | lattices | why / expected |
|---|---|---|---|---|---|
| Q1 | mixer angle β ∈ [0, π/4] (p = 1–4 layers) | phase separator γ·H_cl with H_cl = Rydberg/MIS unit-disk Ising | +x | sq4x4, kag12 | Alternating classical phase and X mixer. Small β is the near-classical limit. NQS have simulated QAOA beyond exact sizes; Rydberg arrays run MIS optimisation |

That is 39 families. With 2–4 inits and 1–3 lattices each, this gives **≈ 150 (family, init, lattice)
combinations** for the screen.

## 3. Relevance, priority tiers and references

**Tiers.**
- **Tier 1.** The same or a very close quench (H, initial state, protocol) was realised
  experimentally, or is a published classical-simulation benchmark. These families also get the
  user's explicitly named interests.
  - Run first, with all listed lattices and initial states and the densest g grid.
  - Mark the experimental parameter values on the g axis of the figures.
- **Tier 2.** The ingredients are realised experimentally or proposed in prominent theory, but this
  particular quench has not been done.
  - Coarse grid, main lattice only, 1–2 initial states.
- **Tier 3.** Methodological controls with no specific outside motivation.
  - One lattice, one initial state, coarse grid. They appear in the report only as controls.

| ID | tier | relevance (why anyone cares) | references (keys, §8) |
|---|---|---|---|
| S3 | 1 | Kicked Ising near Clifford angles. This was the "utility" experiment on 127 qubits, which tensor-network and other classical methods then reproduced. Dual-unitary point: maximal entanglement growth, exactly solvable | Kim23, Tindall24, Begusic24?, BKP18, BKP19 |
| R0, R6 | 1 | Near-classical Ising from \|+x⟩ (Ramsey / many-body interferometry) in Rydberg-dressed lattices and chains. This is exactly target 1 | Zeiher16, Zeiher17, Borish20 |
| R1, G4 | 1 | Quench dynamics of 2D Rydberg Ising arrays (growth of AF correlations), plus the NQS benchmark of 2D TFI quenches that our family A mirrors | Lienhard18, GuardadoSanchez18, SchmittHeyl20 |
| R4, G1 | 1 | Phase diagram and sweeps of square, kagome and ruby Rydberg arrays. The ruby geometry is the toric-code spin-liquid experiment | Ebadi21, Scholl21, Samajdar20, Keesling19, Semeghini21, Verresen21, Samajdar21 |
| R5, C2 | 1 | Many-body scars after Z2 quenches and their Floquet stabilisation | Bernien17, Turner18, Bluvstein21 |
| D1, G3 | 1 | Dipolar XY Rydberg arrays: scalable spin squeezing from \|+x⟩ and continuous symmetry breaking | Bornet23, Chen23, deLeseleuc19 |
| D3 | 1 | Floquet- or microwave-engineered dipolar XXZ, a tunable bridge from Ising to XY | Scholl22, Geier21 |
| D4 | 1 (user) | Frustrated dipolar XY with a proposed Dirac spin liquid in Rydberg/dipole arrays; also kagome dipolar spin liquid | Bintz24, Yao18 |
| I1, I3 | 1 | Trapped-ion long-range Ising: light cones, dynamical quantum phase transitions (the 1D mechanism behind the μ jump) and confinement | Richerme14, Jurcevic14, Schachenmayer13, Jurcevic17, Zhang17, Heyl13, Tan21, Kormos17, Monroe21 |
| D6, R7 | 1 | Positional and filling disorder in dipolar Rydberg systems (glassy dynamics) | Signoles21, Marcuzzi17 |
| Q1 | 1 | QAOA on Rydberg unit-disk graphs, and its classical NQS simulation | Ebadi22, MedvidovicCarleo21 |
| R3, C1 | 2 | Facilitation / anti-blockade and kinetic constraints in Rydberg gases | LesanovskyGarrahan13, Marcuzzi17 |
| S1 | 2 | Toric-code dynamics: realised on processors and the target phase of the ruby Rydberg experiment. Exact RBM forms exist | Satzinger21, Verresen21, Deng17 |
| S2 | 2 | Cluster states: the resource state of measurement-based computation, prepared in Rydberg arrays | Raussendorf01, Bluvstein22 |
| S4, S8 | 2 | Leaving the Clifford point. Magic (stabilizer Rényi entropy) vs μ. Compact RBM forms for stabilizer and Jastrow states | Leone22, PeiClark21 |
| I5 | 2 | Multi-spin interactions in Rydberg lattice-gauge proposals | Celi20? |
| R9 | 2 | Dual-species tweezer arrays | Singh22 |
| R8, I2, D2, D5, G2, R2 | 2 | Natural variations of Tier-1 setups (random fields, dipolar Ising, U(1) breaking, mixed couplings, ordered starts, tilted starts) | inherit from parent family |
| I4, S5, S6, S7, C3 | 3 | Controls: frustration, free fermions, basis rotation, integrability, East constraint | textbook; cite only if needed |

Background references for the report introduction: CarleoTroyer17, SchmittHeyl20, Haegeman16,
Paeckel19, Browaeys20, PeiClark21, Deng17.

## 4. Diagnostics computed for every ED run

Reuse `quench2d.analyze`, with these additions:

- **Fit proxies.** μ, T, R² (A₀ = N_real), and the same for the phase: μ_ph and μ̃_ph (the
  interaction-picture phase).
  - Use the absolute thresholds t*_μ: μ > −10 and t*_ph: μ_ph > −10. Keep the relative one only
    for comparison.
  - If fewer than 5 coefficients survive (an exactly few-body log ψ), report μ = −∞ and the
    order spectrum instead.
- **Order spectrum** W_k(t) = Σ_{|A|=k} c_A² for amplitude and phase, and w_hi (k > 4).
  This is fit-free and works when the FD fit is degenerate.
- **Entanglement / MPS-hardness proxies.**
  - S₁ and S₂ (Rényi-2) at the half cut and the maximum over all bipartitions along the MPS
    ordering.
  - S_rel = S/S_Page.
  - Early growth rate v_E = dS/dt.
  - χ_ε(t): the minimal bond dimension for truncation error ε = 1e-3 at the worst cut. At N ≤ 16
    it caps at 2^{N/2}, so report the fraction of the cap.
  - **Scaling proxy:** at the same t, compare S per cut-boundary bond between N = 12 and 16 on the
    same geometry. Extrapolate χ ~ exp(s_∂ · L_∂) to e.g. 10×10 (clearly labelled as
    extrapolation).
- **Non-classicality.**
  - η = ‖H − diag_σz(H)‖_F / ‖H − tr H/D‖_F. This is basis-dependent on purpose: the RBM lives in
    the σ^z basis.
  - For Floquet families, use η of the effective Floquet Hamiltonian.
- **Representability at Ω = 0.** The effective bond count n_eff(t) = #{i<j : |sin(2φ_ij(t))| > 0.1}
  versus the hidden-unit count αN.
- `minlog`, as before. Optionally the stabilizer Rényi entropy M₂ (4^N Paulis via FWHT; N ≤ 12
  only) for S3/S4/S8.

## 5. Execution plan

Resources: ≤ 30 CPUs and ≤ 40 GB in total (CLAUDE.md). Run ≤ 28 single-thread processes pinned with
`taskset`, and set `OMP_NUM_THREADS=1` and the XLA single-thread flags per process.

0. **References and experimental parameters (main agent, before any compute).**
   - Verify every reference key of §3 by web search: authors, title, journal, volume, pages, year,
     DOI or arXiv number. Write the verified entries to `report/refs.bib`. Drop or flag any entry
     that cannot be verified. Keys marked `?` are the least certain.
   - For each Tier-1 family, extract the experimental parameters from the paper and record them
     with their source in `data/exp_params.csv`: lattice, R_b/a or C₆, Ω/V_nn, Δ,
     initial state, evolution times, α of the ion couplings, and the kick angles of the IBM
     experiment.
   - Make sure each g grid contains the experimental value.
   - Where the experimental geometry does not fit ≤ 16 sites (e.g. the 127-qubit heavy-hex lattice
     of the IBM experiment), use the closest small cluster and state the mismatch.
   - Adjust tiers if a reference turns out weaker or stronger than assumed, and note the change.
1. **Code (main agent, small).**
   - Write a new module `rydberg2d.py` that imports the analysis from `quench2d.py`. It contains:
     - lattice positions;
     - a generic builder from pair lists (ZZ 1/r^α, exchange, k-body, fields) to dense
       (N = 12), sparse (N = 16) and NetKet operators;
     - Floquet propagators for S3/S4/S8;
     - `expm_multiply` (Krylov) evolution for N = 16;
     - the extra diagnostics of §4.
   - Self-checks (`python rydberg2d.py`):
     - Ω = 0 gives |ψ| constant and phase order ≤ 2;
     - Krylov and dense evolution agree at N = 12;
     - KIM at h = 0 gives a stabilizer state (|ψ| uniform on its support);
     - NetKet and dense H agree;
     - n_i n_j expands correctly into ZZ + fields.
2. **Stage 1: cheap ED screen, run in tier order.**
   - Tier 1: all listed lattices and initial states, 12–16 g points including the experimental
     values.
   - Tier 2: main lattice, 1–2 initial states, 8 g points.
   - Tier 3: one lattice, one initial state, 6 g points.
   - N = 12 everywhere. N = 16 for Tier 1 families on 16-site lattices; for Tier 2 only if the
     12-site screen is interesting.
   - Cost: about 10 s per N = 12 run and 1–2 min per N = 16 Krylov run, so ~1 h wall time in total.
   - Output: `data/screen/<ID>_<lattice>_<init>.npz`, plus one quick-look sheet per
     combination at `figs/screen/<ID>_….png`. Each sheet shows μ | μ_ph | S | χ_ε-fraction
     over (g, t).
3. **Triage (main agent).**
   - Build one table with one row per (combination, g). Columns: S_rel(t_max), v_E, max_t μ,
     max_t μ_ph, t*_μ, t*_ph, η, n_eff/αN.
   - Classify each row:
     - (i) TFI-like onset: plateau, then a jump at a g-dependent t*, possibly with "never" windows;
     - (ii) near-classical target: η ≤ 0.1, S_rel ≥ 0.5, μ_ph below threshold;
     - (iii) **gold**: η ≥ 0.3, S_rel ≥ 0.5, μ and μ_ph below threshold up to t_max;
     - (iv) boring: everything easy or everything hard.
   - Produce one overview figure: every screened point in the (S_rel, max μ_ph) plane, coloured by
     η. This is "the map".
   - Select at most 8 families for Stage 2, with at least 2 from (i), the best (ii), and the best
     (iii) candidates. If (iii) is empty, keep the closest one and report it as such.
   - **Prefer relevance.** Among comparably interesting candidates, pick the higher tier. At
     least 5 of the 8 Stage-2 families and at least 4 of the 6 Stage-3 families must be Tier 1.
     A Tier-2/3 family enters only if it is the sole representative of a class, in particular
     gold.
4. **Stage 2: cheap RBM probes (≤ 30 runs).**
   - Settings: α = 1, full sum, N = 12, 3 g values per selected family (easy / onset / hard),
     t_max as screened, RK45 adaptive with fixed tolerances.
   - Cost: ~10–20 min per run on 1 core, so < 30 min wall time.
   - Add a Jastrow and an RBM×Jastrow run at the near-classical points.
   - Drop families where the infidelity does not follow any proxy *and* nothing else is
     interesting.
5. **Stage 3: full maps for interesting families (≤ 6 families).**
   - Full g grid (15–21 points), α = 1.
   - α ∈ {1, 2, 4} plus Jastrow at 3 points. Also check whether α ≳ n_eff/N is what the
     near-classical cases need.
   - N = 16 repeats for at most 2 families at 3 g values, to show the trend in S and in infidelity
     with size (≈ 1 h per run).
   - Initial RBM: SR infidelity minimisation, cached per (α, ψ₀), as in `rbm_run.py`.
     - Exactly representable starts get exact parameters instead.
     - Tilted product states are exact with α = 0 plus visible biases.
6. **Analysis.**
   - As before: Spearman and partial correlations ρ(μ|S), ρ(μ_ph|S), ρ(S|μ), ρ(·|S,t), and t* vs
     t*_inf ordering.
   - New: χ_ε(t) against log I_RBM(t). The **"MPS-hard / NQS-easy" quadrant** is
     χ_ε-fraction → 1 while I_RBM < 1e-3.
7. **Figures and report.**
   - Main comparison figure: one row per interesting family, four panels:
     **μ(g,t) | μ_ph(g,t) | S(g,t) | log₁₀ I_RBM(g,t)**.
     - Shared t axis.
     - t*_μ (white solid), t*_ph (white dotted) and t*_inf (cyan dashed) overlaid.
     - A χ_ε = cap contour (grey) on the S panel.
     - The experimental parameter value(s) marked on the g axis by a labelled tick or arrow
       ("exp."), with the reference in the caption.
     - Rows ordered by tier, then by interest.
   - The overview map (Stage 1).
   - Representability at Ω = 0 (infidelity vs α and Jastrow; n_eff(t)).
   - Scaling proxy (N = 12 vs 16).
   - **Plot style.** Every figure follows the `/python-plotting` skill:
     - the `STYLE` preamble, `simple_ticks` and LaTeX labels;
     - categorical colours in the order black, red, blue, then Wong;
     - `viridis` heatmaps with labelled colorbars;
     - explicit legend placement and (a)/(b) panel labels;
     - a label or legend entry for every guide line.
   - **Readability.** Each figure has to be understandable on its own:
     - axis labels in words plus symbols (e.g. "time $t\,V_{nn}$");
     - the family's Hamiltonian and its fixed parameters stated in the panel title or caption;
     - shared colour limits across rows wherever the rows are compared;
     - captions that say what to look at, not only what is plotted.
     Check every figure by eye (the PNG preview) before it goes into the report.
8. **Update the report and deliver it (main agent).**
   - Extend the existing `report/quench2d_report.tex` with new sections, not a new file:
     - "Rydberg and related quenches", with the overview map, the comparison rows, the Ω = 0
       representability study and the scaling proxy;
     - a dedicated **"Gold-standard case"** section. It shows the best candidate (or the negative
       result and why) with its η, S, μ, μ_ph and I_RBM;
     - updated abstract, verdict and limitations;
     - a **Motivation** paragraph for every family that appears in a figure. It says what was
       realised or proposed, by whom (citation), and why that quench is interesting for the
       MPS-vs-NQS question;
     - a short "relevance" column in the summary table of studied families (tier and references);
     - a bibliography from `refs.bib`, containing only the verified entries.
   - Add a **Definitions** section (or table) covering every quantity used anywhere in the
     report, each with its formula:
     - c_A, μ, T, A₀, μ₂, μ_ph, μ̃_ph, W_k, w_hi, minlog;
     - S₁, S₂, S_rel, S_Page, v_E, χ_ε, η, n_eff;
     - t*_μ, t*_ph, t*_w, t_dev, t*_inf, I_RBM = 1 − F;
     - α and the Jastrow ansatz;
     - every Hamiltonian parameter (Ω, Δ, V_nn, R_b, R_c, α of 1/r^α, J_z/J_xy, Θ, θ, h, ε).
   - Compile to PDF with `pdflatex → bibtex → pdflatex → pdflatex`, with no undefined references or
     citations. Check the rendered pages visually.
   - Send the compiled PDF to the user in the conversation (`SendUserFile`) once the results are
     in.

## 6. Success criteria

- **Near-classical target.** At least one Rydberg family (R/I/D) with S_rel ≥ 0.5 (χ_ε at its cap)
  where I_RBM(α ≤ 2 or RBM×Jastrow) < 1e-3 over the same time window.
- **Gold.** The same, with η ≥ 0.3, and μ and μ_ph staying below −10 throughout.
- **Relevance.** The headline result (near-classical or gold) comes from a Tier-1 family if any
  Tier-1 family qualifies; otherwise the report says so explicitly.
- **Proxy validity.** In at least 2 new families, μ or μ_ph orders t*_inf with Spearman ≥ 0.8, and
  the partial correlation with S removed stays ≥ 0.5.

## 7. Known risks

- **Small clusters.** At 12–16 sites, 2D entanglement saturates quickly (S_Page is reached within
  t ≈ 1–2 for generic quenches). "MPS-hard" rests on the S-per-boundary scaling proxy, which is
  an extrapolation.
- **Exact zeros.** Stabilizer supports, blockade, and Néel starts make log|ψ| singular. Mitigations:
  tilted starts, finite V, and reporting `minlog` and the fraction of near-zero amplitudes.
- **Degenerate FD fits** when log ψ is exactly few-body. Mitigation: fall back to the order
  spectrum W_k.
- **Stiff phases** at V/Ω ≫ 1. Mitigations: dt·V_max ≤ 0.2, the interaction-picture phase, and
  dt-halving checks on 3 points per family. TDVP will also be slower there.
- **Basis dependence.** "Near-classical" means diagonal in σ^z. S6 deliberately breaks this
  assumption.
- **Uncertain references.** The reference list below was written from memory. Keys marked `?` and
  every detail (journal, volume, year) must be verified in step 0 before they are cited. Nothing
  unverified goes into the bibliography.
- **Experiment ≠ cluster.** Tier-1 relevance refers to the experiment's Hamiltonian and protocol,
  not to its system size. The report must state, for each family, how far the 12–16-site cluster is
  from the experiment.
- **Scope.** 39 families is a screen, not a study. Only Stage 1 touches all of them. Stages 2–3 are
  capped at 8 and 6 families.

## 8. Reference candidates (from memory; **verify all in step 0**)

Confidence: ✓ = confident in authors, title and venue; ~ = fairly confident; ? = unsure.

| key | reference | conf. |
|---|---|---|
| Kim23 | Y. Kim et al., "Evidence for the utility of quantum computing before fault tolerance", Nature 618, 500 (2023) | ✓ |
| Tindall24 | J. Tindall et al., "Efficient tensor network simulation of IBM's Eagle kicked Ising experiment", PRX Quantum 5, 010308 (2024) | ~ |
| Begusic24? | T. Begušić, G. K.-L. Chan, classical simulation of the IBM utility experiment (Pauli propagation), Sci. Adv. (2024) | ? |
| BKP18 | B. Bertini, P. Kos, T. Prosen, "Exact spectral form factor in a minimal model of many-body quantum chaos", PRL 121, 264101 (2018) | ✓ |
| BKP19 | B. Bertini, P. Kos, T. Prosen, "Entanglement spreading in a minimal model of maximal many-body quantum chaos", PRX 9, 021033 (2019) | ✓ |
| Zeiher16 | J. Zeiher et al., "Many-body interferometry of a Rydberg-dressed spin lattice", Nat. Phys. 12, 1095 (2016) | ✓ |
| Zeiher17 | J. Zeiher et al., "Coherent many-body spin dynamics in a long-range interacting Ising chain", PRX 7, 041063 (2017) | ✓ |
| Borish20 | V. Borish et al., "Transverse-field Ising dynamics in a Rydberg-dressed atomic gas", PRL 124, 063601 (2020) | ~ |
| Lienhard18 | V. Lienhard et al., "Observing the space- and time-dependent growth of correlations in dynamically tuned synthetic Ising models with antiferromagnetic interactions", PRX 8, 021070 (2018) | ~ |
| GuardadoSanchez18 | E. Guardado-Sanchez et al., "Probing the quench dynamics of antiferromagnetic correlations in a 2D quantum Ising spin system", PRX 8, 021069 (2018) | ~ |
| SchmittHeyl20 | M. Schmitt, M. Heyl, "Quantum many-body dynamics in two dimensions with artificial neural networks", PRL 125, 100503 (2020) | ✓ |
| Ebadi21 | S. Ebadi et al., "Quantum phases of matter on a 256-atom programmable quantum simulator", Nature 595, 227 (2021) | ✓ |
| Scholl21 | P. Scholl et al., "Quantum simulation of 2D antiferromagnets with hundreds of Rydberg atoms", Nature 595, 233 (2021) | ✓ |
| Samajdar20 | R. Samajdar et al., "Complex density wave orders and quantum phase transitions in a model of square-lattice Rydberg atom arrays", PRL 124, 103601 (2020) | ~ |
| Samajdar21 | R. Samajdar et al., "Quantum phases of Rydberg atoms on a kagome lattice", PNAS 118, e2015785118 (2021) | ~ |
| Keesling19 | A. Keesling et al., "Quantum Kibble–Zurek mechanism and critical dynamics on a programmable Rydberg simulator", Nature 568, 207 (2019) | ✓ |
| Semeghini21 | G. Semeghini et al., "Probing topological spin liquids on a programmable quantum simulator", Science 374, 1242 (2021) | ✓ |
| Verresen21 | R. Verresen, M. D. Lukin, A. Vishwanath, "Prediction of toric code topological order from Rydberg blockade", PRX 11, 031005 (2021) | ✓ |
| Bernien17 | H. Bernien et al., "Probing many-body dynamics on a 51-atom quantum simulator", Nature 551, 579 (2017) | ✓ |
| Turner18 | C. J. Turner et al., "Weak ergodicity breaking from quantum many-body scars", Nat. Phys. 14, 745 (2018) | ✓ |
| Bluvstein21 | D. Bluvstein et al., "Controlling quantum many-body dynamics in driven Rydberg atom arrays", Science 371, 1355 (2021) | ~ |
| Bornet23 | G. Bornet et al., "Scalable spin squeezing in a dipolar Rydberg atom array", Nature 621, 728 (2023) | ~ |
| Chen23 | C. Chen et al., "Continuous symmetry breaking in a two-dimensional Rydberg array", Nature 616, 691 (2023) | ~ |
| deLeseleuc19 | S. de Léséleuc et al., "Observation of a symmetry-protected topological phase of interacting bosons with Rydberg atoms", Science 365, 775 (2019) | ~ |
| Scholl22 | P. Scholl et al., "Microwave engineering of programmable XXZ Hamiltonians in arrays of Rydberg atoms", PRX Quantum 3, 020303 (2022) | ~ |
| Geier21 | S. Geier et al., "Floquet Hamiltonian engineering of an isolated many-body spin system", Science 374, 1149 (2021) | ~ |
| Bintz24 | J. Bintz et al., "Dirac spin liquid in quantum dipole arrays" (title confirmed by user; venue/year to verify) | ~ |
| Yao18 | N. Y. Yao, M. P. Zaletel, D. M. Stamper-Kurn, A. Vishwanath, "A quantum dipolar spin liquid", Nat. Phys. 14, 405 (2018) | ~ |
| Richerme14 | P. Richerme et al., "Non-local propagation of correlations in quantum systems with long-range interactions", Nature 511, 198 (2014) | ✓ |
| Jurcevic14 | P. Jurcevic et al., "Quasiparticle engineering and entanglement propagation in a quantum many-body system", Nature 511, 202 (2014) | ✓ |
| Schachenmayer13 | J. Schachenmayer et al., "Entanglement growth in quench dynamics with variable range interactions", PRX 3, 031015 (2013) | ✓ |
| Jurcevic17 | P. Jurcevic et al., "Direct observation of dynamical quantum phase transitions in an interacting many-body system", PRL 119, 080501 (2017) | ✓ |
| Zhang17 | J. Zhang et al., "Observation of a many-body dynamical phase transition with a 53-qubit quantum simulator", Nature 551, 601 (2017) | ✓ |
| Heyl13 | M. Heyl, A. Polkovnikov, S. Kehrein, "Dynamical quantum phase transitions in the transverse-field Ising model", PRL 110, 135704 (2013) | ✓ |
| Tan21 | W. L. Tan et al., "Domain-wall confinement and dynamics in a quantum simulator", Nat. Phys. 17, 742 (2021) | ~ |
| Kormos17 | M. Kormos et al., "Real-time confinement following a quantum quench to a non-integrable model", Nat. Phys. 13, 246 (2017) | ✓ |
| Monroe21 | C. Monroe et al., "Programmable quantum simulations of spin systems with trapped ions", RMP 93, 025001 (2021) | ✓ |
| Signoles21 | A. Signoles et al., "Glassy dynamics in a disordered Heisenberg quantum spin system", PRX 11, 011011 (2021) | ~ |
| Marcuzzi17 | M. Marcuzzi et al., "Facilitation dynamics and localization phenomena in Rydberg lattice gases with position disorder", PRL 118, 063606 (2017) | ~ |
| LesanovskyGarrahan13 | I. Lesanovsky, J. P. Garrahan, "Kinetic constraints, hierarchical relaxation, and onset of glassiness in strongly interacting and dissipative Rydberg gases", PRL 111, 215305 (2013) | ~ |
| Ebadi22 | S. Ebadi et al., "Quantum optimization of maximum independent set using Rydberg atom arrays", Science 376, 1209 (2022) | ✓ |
| MedvidovicCarleo21 | M. Medvidović, G. Carleo, "Classical variational simulation of the Quantum Approximate Optimization Algorithm", npj Quantum Inf. 7, 101 (2021) | ~ |
| Satzinger21 | K. J. Satzinger et al., "Realizing topologically ordered states on a quantum processor", Science 374, 1237 (2021) | ✓ |
| Deng17 | D.-L. Deng, X. Li, S. Das Sarma, "Quantum entanglement in neural network states", PRX 7, 021021 (2017) | ✓ |
| Raussendorf01 | R. Raussendorf, H. J. Briegel, "A one-way quantum computer", PRL 86, 5188 (2001) | ✓ |
| Bluvstein22 | D. Bluvstein et al., "A quantum processor based on coherent transport of entangled atom arrays", Nature 604, 451 (2022) | ~ |
| Leone22 | L. Leone, S. F. E. Oliviero, A. Hamma, "Stabilizer Rényi entropy", PRL 128, 050402 (2022) | ✓ |
| PeiClark21 | M. Y. Pei, S. R. Clark, "Compact neural-network quantum state representations of Jastrow and stabilizer states", J. Phys. A 54, 405304 (2021) | ~ |
| Celi20? | A. Celi et al., "Emerging two-dimensional gauge theories in Rydberg configurable arrays", PRX 10, 021057 (2020) | ? |
| Singh22 | K. Singh et al., "Dual-element, two-dimensional atom array with continuous-mode operation", PRX 12, 011040 (2022) | ~ |
| CarleoTroyer17 | G. Carleo, M. Troyer, "Solving the quantum many-body problem with artificial neural networks", Science 355, 602 (2017) | ✓ |
| Haegeman16 | J. Haegeman et al., "Unifying time evolution and optimization with matrix product states", PRB 94, 165116 (2016) | ✓ |
| Paeckel19 | S. Paeckel et al., "Time-evolution methods for matrix-product states", Ann. Phys. 411, 167998 (2019) | ✓ |
| Browaeys20 | A. Browaeys, T. Lahaye, "Many-body physics with individually controlled Rydberg atoms", Nat. Phys. 16, 132 (2020) | ✓ |
