# Agent task: does minimizing Walshness help RBMs on the Kitaev model? (proof of concept)

## 0. Goal and deliverable

Test whether a basis change $U$ chosen to minimize the Walshness of the ground state $\psi$ improves RBM energy convergence, compared with the naive (computational) basis.

**Model.** Kitaev honeycomb via `kh.get_KitaevHoneycomb_Hamiltonian` (`Netket/Hamiltonians/KitaevHoneycomb.py`), `Lx=Ly=3, pbc=True` (18 sites), ground state from `nk.exact.lanczos_ed`. Two parameter sets, both already present (commented out) in `Kitaev.ipynb`:
- `kitaev_honeycomb`: pure Kitaev, `K=1, J=Gamma=GammaPrime=0, h=0` (stabilizer-like, Pauli-sum with only XX/YY/ZZ bonds; the clean case for term counting);
- `kitaev_honeycomb_gapped_3`: `h=(0.05,)*3, J=0.3, K=(0.5,0.8,1), Gamma=0.3, GammaPrime=0.1` (generic, many Pauli terms).

Build `op_kitaev = H.to_pauli_strings()` as in `Kitaev.ipynb`. (The notebook currently has `spin_glass` active; Kitaev is selected by uncommenting.) The Kagome J1-J2 model is an alternative only if Kitaev is inconclusive.

**Environment.** Python: `check_time_evolution/.venv` (NetKet 3.19.2, stim 1.16.0 installed; `stim.Tableau.iter_all` exists). The NetKet 3.22.4 remarks below were checked on another version; re-assert ordering on 3.19.2.

**Deliverable.** A jupytext-paired notebook `Netket/CLANQS/Walshness_Kitaev.{py,ipynb}` (py:percent), structured like `Kitaev.ipynb`. A small module `Netket/CLANQS/walshness.py` holds the numerics (see §5). Per the `jupytext-sync` skill: after every edit run `uv run jupytext --sync Walshness_Kitaev.py` and re-execute the .ipynb in place, so code and outputs stay current.

**Four bases compared** (each gives a transformed Hamiltonian $H_U = U H U^\dagger$ and target $\psi_U = U\psi$):

| name | $U$ | meaning |
|---|---|---|
| `naive` | $I$ | computational basis |
| `local` | $V$ | single-qubit rotations only (the paper's method) |
| `cliff→local` | $V\,C$ | Clifford first, then local rotations (extension, §4.1) |
| `local→cliff` | $C\,V$ | local rotations first, then Clifford (converse, §4.2) |

**Reported for every basis** (table + plots in the notebook):
1. Walshness $W^Z_\alpha[\psi_U]$ at $\alpha=1$ (and 2), plus the lower bound LB.
2. RBM energy convergence vs. iteration, with the exact ED energy as reference (MCState, one seed, settings below).
3. **Hamiltonian term growth**: number of Pauli terms and max Pauli weight of $H_U$ vs. $H$ (§6). Answer explicitly: is there an approach that does not grow the term count?

**Background.** Paul, Chen, Jiang, Zhao, Luo, *Walshness: an intrinsic neural-network representability metric for quantum states*, arXiv:2610.00505 (Prop. 1, End Matter App. A, Supplement II.3). The paper minimizes Walshness only over single-site SU(2) rotations. Putting a Clifford before or after is an extension that the paper lists as future work; none of it is validated there, so the tests in §7 are mandatory.

---

## 1. Conventions (fix first, then test)

- **Basis.** $|\mathbf s\rangle$, $\mathbf s\in\{0,1\}^n$, $Z_j|\mathbf s\rangle=(-1)^{s_j}|\mathbf s\rangle$.
- **Vector index.** $\mathrm{idx}(\mathbf s)=\sum_j s_j 2^{n-1-j}$ (site 0 most significant). Matches NetKet `Spin(0.5,n).all_states()` with $\sigma_j=1-2s_j$ (checked for 3.22.4; re-assert on the installed 3.19.2); assert in tests.
- **Walsh transform.** $\widehat\psi(\mathbf r)=2^{-n}\sum_{\mathbf s}(-1)^{\mathbf r\cdot\mathbf s}\psi(\mathbf s)$, via in-place FWHT, $O(n2^n)$. Spectrum $p(\mathbf r)=|\widehat\psi(\mathbf r)|^2/\sum|\widehat\psi|^2$.
- **Fixed-basis Walshness.** $W^Z_\alpha[\psi]=\frac1\alpha\log\sum_{\mathbf r}p(\mathbf r)e^{\alpha|\mathbf r|_1}$ (`scipy.special.logsumexp`), range $[0,n]$.
- **Intrinsic (local-optimized) Walshness.** $W_\alpha[\varphi]=\min_V W^Z_\alpha[V\varphi]$.
- **Transformation.** $U$ acts on $\psi$; $H_U=UHU^\dagger$; the RBM is trained on $H_U$ in the computational basis and its target is $\psi_U=U\psi$.
- **Defaults.** Optimize at $\alpha=1$; also report $\alpha=2$ with $V$ re-optimized.

---

## 2. Structural facts

### 2.1 Inner problem for fixed $\varphi$: classical spin model

With $\varphi$ normalized,

$$W_\alpha[\varphi]=\frac n2+\frac1\alpha\log\min_{\{\hat{\mathbf n}_j\}}E,\quad E=\Big\langle\varphi\Big|\bigotimes_jA_\alpha(\hat{\mathbf n}_j)\Big|\varphi\Big\rangle,\quad A_\alpha(\hat{\mathbf n})=\cosh\tfrac\alpha2 I-\sinh\tfrac\alpha2\,\hat{\mathbf n}\cdot\boldsymbol\sigma,$$

with $V_j^\dagger XV_j=\hat{\mathbf n}_j\cdot\boldsymbol\sigma$. $E$ is linear in each $\hat{\mathbf n}_j$, so the exact single-site update is

$$\mathbf v_j=\Big\langle\varphi\Big|\boldsymbol\sigma_j\bigotimes_{k\ne j}A_\alpha(\hat{\mathbf n}_k)\Big|\varphi\Big\rangle,\quad\hat{\mathbf n}_j\leftarrow\mathbf v_j/\|\mathbf v_j\|$$

(skip if $\mathbf v_j=0$). $E$ never increases; the joint problem is nonconvex, so use restarts.

### 2.2 Invariances

$W^Z_\alpha$ is unchanged by global phase, qubit permutations, and any single-qubit gate applied *after* $C$ in $VC$ (absorbed by $V$). Hence for `cliff→local` only the coset of $C$ modulo left-multiplication by local Cliffords matters, and every coset contains a product of generalized CZ gates

$$G^{ab}_{ij}=I-2\Pi^a_i\Pi^b_j,\quad\Pi^a=\tfrac12(I-\sigma^a),\quad a,b\in\{x,y,z\},\ i<j,$$

(proof sketch: push local Cliffords left through CZ, which only relabels axes/signs; signs are Paulis, also absorbed). Search space: words $w=g_k\cdots g_1$ over $9\binom n2$ gates, each Hermitian and self-inverse; never repeat a gate twice in a row.

Standard-gate form: $G^{ab}_{ij}=(L_a\otimes L_b)^\dagger\mathrm{CZ}_{ij}(L_a\otimes L_b)$ with $L_z=I$, $L_x=H$, $L_y=HS^\dagger$ ($S^\dagger$ first). E.g. $G^{zx}_{ij}$ = CNOT(control $i$, target $j$).

**This argument does NOT apply to `local→cliff`** ($U=CV$): the local layer now sits before $C$ and the final basis is fixed, so local Cliffords after $C$ are not free. See §4.2.

### 2.3 Rigorous lower bound

$$W_\alpha[\varphi]\ge \mathrm{LB}[\varphi]=\sum_j\frac{1-\|\mathbf b_j\|}{2},\quad\mathbf b_j=(\langle X_j\rangle,\langle Y_j\rangle,\langle Z_j\rangle).$$

(Jensen: $\alpha W^Z_\alpha=\log\mathbb E e^{\alpha K}\ge\alpha\mathbb E K$, $K=\sum_jP^-_j$.) It is the $\alpha\to0$ limit, invariant under local unitaries and permutations. Since $W^Z_\alpha\ge W_\alpha\ge\mathrm{LB}$, it is also a valid lower bound for the fixed-basis value. Uses: (1) prune candidates with $\mathrm{LB}\ge$ best; (2) initialize axes at the Bloch vectors $\mathbf b_j/\|\mathbf b_j\|$; (3) report the gap $J-\mathrm{LB}$.

Also assert: $W_\alpha\le n g_\alpha$, $g_\alpha=\frac1\alpha\log\frac{1+e^\alpha}{2}$; and $W_\alpha=0$ iff $\varphi$ is a product state.

---

## 3. Inner solver `intrinsic_walshness(phi, alpha, inits)`

For each initial axis configuration:
1. $\chi=\bigotimes_kA_\alpha(\hat{\mathbf n}_k)\varphi$.
2. Sweep $j$: $\chi_{-j}=A_\alpha(\hat{\mathbf n}_j)^{-1}\chi$ ($A^{-1}=\cosh\frac\alpha2I+\sinh\frac\alpha2\hat{\mathbf n}\cdot\boldsymbol\sigma$); $v^a_j=\mathrm{Re}\langle\varphi|\sigma^a_j|\chi_{-j}\rangle$ (assert imaginary part negligible); update $\hat{\mathbf n}_j$; $\chi\leftarrow A_\alpha(\hat{\mathbf n}_j)\chi_{-j}$.
3. After each sweep $E=\langle\varphi|\chi\rangle$ (real, $>0$, monotone decreasing up to $10^{-12}$ relative). Stop when decrease $\le10^{-10}\max(1,|E|)$ or after 200 sweeps.
4. Return $W=n/2+\log E_{\min}/\alpha$, best axes, all restart values.

**Continuation.** For $\alpha\ge1$ optimize $\alpha=0.25\to0.5\to1(\to2)$, warm-starting axes.
**Restarts.** Search mode: Bloch axes, parent axes, one random. Final mode: Bloch axes + 32 random, with continuation.
**Single-site ops.** Reshape to `(2**j,2,2**(n-j-1))` and `einsum`; never build $2^n\times2^n$ matrices (except in $n\le3$ tests).

**Local unitaries from axes.** $V_j=W_j^\dagger$, $W_j=\exp(-\tfrac i2\theta_j\hat{\mathbf m}_j\cdot\boldsymbol\sigma)$, $\hat{\mathbf m}_j\propto\hat{\mathbf x}\times\hat{\mathbf n}_j$, $\theta_j=\arccos n_{j,x}$. Special cases: $\hat{\mathbf n}=\hat{\mathbf x}\Rightarrow V=I$; $\hat{\mathbf n}=-\hat{\mathbf x}\Rightarrow V=iZ$. Assert $V_j^\dagger XV_j=\hat{\mathbf n}_j\cdot\boldsymbol\sigma$ to $10^{-12}$. The free X-rotation in $V_j$ does not change $W$ but changes $\psi_U$; always use this minimal-rotation choice.

**Final check.** $W^Z_\alpha$ via FWHT of $V\varphi$ equals the solver value to $10^{-8}$.

---

## 4. Outer search over the Clifford part

The state $\psi$ is a dense $2^n$ vector from ED; $n\le18$ makes one inner solve cost $O(n2^n)$ per sweep, so for $n>12$ reduce restarts/beam width (log wall time per depth).

### 4.1 `cliff→local` ($U=VC$)

Objective $J(C)=W_\alpha[C\psi]$. Beam search over words of $G^{ab}_{ij}$:
- beam width $B=4$, screen size $K=16$ children per parent (smallest LB), max depth $D_{\max}=n$ (not $3n$; PoC), patience 2;
- children: apply one gate to $\varphi$, compute LB ($3n$ expectation values), prune $\mathrm{LB}\ge J_{\text{best}}-10^{-12}$, evaluate $J$ in search mode warm-started from the parent;
- new beam = $B$ best by $J$, then LB, then word length; dedupe by fingerprint (sorted single-site Bloch lengths and sorted two-site purities, rounded to $10^{-9}$);
- on a new global best, re-evaluate in final mode; stop when $J<10^{-8}$, depth $D_{\max}$, or patience exhausted;
- polish at $\alpha=1,2$.

Return the word $w$, the axes/$V$, and $\psi_U=VC\psi$ with $H_U=VCHC^\dagger V^\dagger$.

### 4.2 `local→cliff` ($U=CV$)

1. Optimize $V$ on $\psi$ with the inner solver (this is the `local` basis), giving $\varphi_0=V\psi$.
2. Search for $C$ minimizing the **fixed-basis** $W^Z_\alpha[C\varphi_0]$ directly (no further local freedom, so it is evaluated by FWHT, which is cheap and exact).
3. Gate set: because local Cliffords are not free here, use $\{H_i,S_i,\mathrm{CZ}_{ij}\}$ (generators of the Clifford group), same beam scheme, max depth $2n$. LB remains a valid prune ($W^Z\ge\mathrm{LB}$).
4. Optional refinement: alternate (re-optimize $V$ on $C^\dagger$-pulled-back state) a couple of times; skip if the first pass already shows no gain. Report which was done.

$H_U=CVHV^\dagger C^\dagger$.

### 4.3 Dropped for the PoC

Exhaustive search over all Cliffords, simulated annealing, penalties for Hamiltonian locality ($\lambda=0$), the generic CLI/JSON layout. Add back only if the beam search visibly stalls.

---

## 5. Code layout

```
Netket/CLANQS/
  walshness.py            # conventions, fwht, walshness_Z, bloch_vectors, lower_bound,
                          # intrinsic_walshness, axes_to_unitaries, apply_G, beam_search
  Walshness_Kitaev.py     # jupytext py:percent, paired with .ipynb
  Walshness_Kitaev.ipynb
```

Use `stim` (installed) only for the Pauli conjugation in §6 and the Clifford unit tests.

---

## 6. Hamiltonian transformation and term counting

Represent $H$ as a Pauli sum $\sum_k w_kP_k$ (from `op_kitaev.operators/weights`).

1. **Clifford part**: conjugate each Pauli, $P\mapsto CPC^\dagger$ (verify direction with a CNOT test; a Clifford maps Pauli to Pauli, so the term count cannot grow, equal strings merge).
2. **Local part**: $V_j\sigma^aV_j^\dagger=\sum_bR^{(j)}_{ba}\sigma^b$, $R_{ba}=\frac12\mathrm{Tr}(\sigma^bV_j\sigma^aV_j^\dagger)$. A weight-$w$ term expands into up to $3^w$ terms, so the count can grow; merge equal strings and drop coefficients $<10^{-12}$.
3. **Report per basis**: number of terms (before/after merge), max weight, and number of terms $>10^{-8}$.
4. **Expected answer to discuss in the notebook** (verify numerically):
   - `naive`: reference count.
   - `local` with generic axes: growth (each bond term up to 9 terms).
   - Clifford-only ($V$ restricted to Pauli-axis-permuting rotations, which Clifford allows): no growth by construction, but then walshness is not minimized over $V$.
   - The best case for "no growth" is a $V$ that happens to map each $\hat{\mathbf n}_j$ onto a Pauli axis (e.g. found axes snapping to $\pm x,\pm y,\pm z$ for the Kitaev point); check whether the optimizer's axes do this and report `max |n|` snapping error.
   - `cliff→local` and `local→cliff`: growth is bounded by that of the local layer alone (Clifford does not add terms); report the actual numbers.

Do not add a separate discrete-axis variant unless the free-axis result is inconclusive.

---

## 7. Tests (pass before the experiments)

- **T1 (bit order).** $Z_0$ on index $2^{n-1}$ gives $-1$; ordering matches `Spin(0.5,n).all_states()`.
- **T2 (product states).** $W^Z_\alpha[|{-}\rangle^{\otimes n}]=n$, $[|{+}\rangle^{\otimes n}]=0$, $[|0\rangle^{\otimes n}]=ng_\alpha$.
- **T4 (inner solver).** Random product states give intrinsic $W<10^{-8}$.
- **T5 (GHZ).** Even $n\in\{4,6,8\}$: intrinsic $W=n/2$ within $10^{-6}$ and $\mathrm{LB}=n/2$.
- **T7.** Solver value equals FWHT value of $V\varphi$ to $10^{-8}$; $V_j^\dagger XV_j=\hat{\mathbf n}_j\cdot\boldsymbol\sigma$.
- **T9 (gates).** All 9 $G^{ab}$ agree between projector form and standard-gate form, $G^2=I$.
- **T10.** GHZ with Clifford allowed reaches $J=0$ in $\le n-1$ gates.
- **T11 (transformed Hamiltonian).** For a random small $H$ (n=4) and random $U=VC$, the spectrum of $H_U$ built from Pauli conjugation equals that of $H$, and $\langle\psi_U|H_U|\psi_U\rangle=\langle\psi|H|\psi\rangle$.

---

## 8. RBM experiment (in the notebook)

Follow the structure of `Kitaev.ipynb`:
- `nk.models.RBM(alpha=5, param_dtype=np.complex64)`, `MCState` with `MetropolisLocal`, `N_Samples=2**9`, `nk.driver.VMC_SR`, Sgd lr `0.002`, `N_iters=7000`, diag_shift `3e-4`; **one seed**, identical for all four bases. Treat differences as suggestive only (single seed).
- Train on $H_U$ for each basis; the exact reference energy is the same for all ($U$ is unitary).
- Plot $|E-E_{\text{exact}}|$ vs. iteration (log scale) for all four bases in one figure; also rolling average as in `Kitaev.ipynb`.
- Save logs to `logs/` as in `Kitaev.ipynb` with the metadata (basis name, $W$, $\alpha$, term counts).
- Use local paths (not the `/storage/niggeni/...` ones in `Kitaev.ipynb`). The Julia/JuliaCall setup is **not** needed.
- Also compute a Hamming-weight histogram $P(|\mathbf r|_1=k)$ of the Walsh spectrum for $\psi$ in each basis.

NetKet 3.22.4 caveat: `nk.observable.InfidelityOperator` on a `FullSumState` omits the complex conjugate (wrong for complex targets). Not used here since the RBM is trained by energy, but if infidelity to $\psi_U$ is logged, compute it independently as $1-|\langle\psi_U|\phi\rangle|^2$ from `to_array()`.

---

## 9. Pitfalls

- **Order.** $U=VC$ means $C$ acts first on $\psi$; $U=CV$ means $V$ first. Do not mix them in the Hamiltonian transform ($H_U=UHU^\dagger$).
- **Bounds.** The inner minimum is an upper bound (local minima); LB is rigorous. Report both.
- **Endianness** differs between libraries (`stim`, NetKet); test on a CNOT and a basis state.
- **Free phases.** Free X-rotation in $V$ and Paulis in $C$ do not change $W$ but do change $\psi_U$; keep the canonical choices.
- **Overflow.** $E\le e^{n\alpha/2}$; work in log space.
- **Local Cliffords for term counting.** A rotation $V$ that lands exactly on Pauli axes costs no extra terms; a generic $V$ does. The count is not invariant under the $V$ gauge freedom (X-rotation), so use the canonical $V$ of §3 and report that choice.
- **Single-seed RBM runs** are noisy; do not claim a speedup from one run without an error estimate.

---

## 10. Report back

- Table: basis, $W^Z_1$, $W^Z_2$, LB, word length, #terms, max weight, final energy error.
- Pass/fail of every test.
- Energy-convergence figure for the four bases; Hamming-weight histograms.
- Term-growth discussion with the explicit answer to "does any approach avoid growth".
- Diagnosis of any failure. Do not tune hyperparameters on the same instance used for the final table.
