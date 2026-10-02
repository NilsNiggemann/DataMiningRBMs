# Suggestions and TODO triage

Companion to `main.tex`. Written after reading the note and tracing each `\section{todo}`
item through the code. Three parts:

1. **[Results that need re-checking](#1-results-that-need-re-checking)** — findings that
   affect statements already in the note.
2. **[TODO-by-TODO triage](#2-todo-by-todo-triage)** — for each item in `\section{todo}`:
   what the actual obstacle is, and a concrete recipe.
3. **[Further suggestions](#3-further-suggestions)** — methodology, ensemble design,
   code consolidation.
4. **[Suggested order of work](#4-suggested-order-of-work)**

Claims below that I verified by running code are marked **[verified]** with the command.
Claims I am not certain about are marked **(not sure)**. Everything else is a reading of
the source.

---

## 1. Results that need re-checking

### 1.1 The Heisenberg forest-model failure is built on meaningless CE features **[verified]**

`main.tex` §"Performance of the forest model on spin systems with $SU(2)$ symmetry"
shows `fail_forest_heisenberg.png` — the random forest, trained on the random-Hamiltonian
dataset, failing on the $J_1$-$J_2$ models. One of the features fed to it is `CEEntropy`,
defined in `datamine/perfvsclusterexpn.ipynb` (cell 3), which **hardcodes** the Hilbert
space:

```python
def CEEntropy(psi):
    hilbert = nk.hilbert.Spin(1/2, N=16)      # <-- hardcoded
    cluster_coeffs = ce.fwht_coeffs_in_cluster_col_order(np.log(psi), hilbert)
```

The Heisenberg dataset is **12 sites**, not 16 (square 3×4, triangular 3×4, kagome 2×2 —
see `get_J1J2_Hamiltonian`, `Netket/tunable_Heisenberg_models.py:34`). Passing a
$2^{12}$-component state with an $N=16$ Hilbert space **does not raise an error**: the
permutation gather at `ClusterExpansion/cluster_expansion.py:75` (`psi_arr[perm]`) runs
under JAX, and JAX *clamps* out-of-bounds indices rather than raising. The permutation has
65536 entries but only 4096 of them are in range, so the function happily returns 65536
finite coefficients built from 4096 real amplitudes and 61440 clamped copies of `psi[4095]`.

```
$ check_time_evolution/.venv/bin/python -c "
import numpy as np, sys; sys.path.append('ClusterExpansion')
import netket as nk, cluster_expansion as ce
c = ce.fwht_coeffs_in_cluster_col_order(np.log(np.random.rand(2**12)+1j*np.random.rand(2**12)),
                                        nk.hilbert.Spin(0.5, 16))
print('no error; len', len(c), 'all finite:', np.all(np.isfinite(np.array(c))))"
no error; len 65536 all finite: True
```

**Consequence.** `CEEntropy` (and `CEdecay`, same hardcoding) on the Heisenberg rows is
noise, so the model was asked to extrapolate on a feature that carries no information
about those states. The failure may well be real, but this figure does not establish it.
Also note that in `perfvsclusterexpn.ipynb` cell 19 the same notebook *does* build a
12-site Hilbert space (`hilb_test = nk.hilbert.Spin(1/2, N=12)`) for the coefficient plot
— so the two analyses in the same notebook disagree about the system size.

**Minimal re-run.** Re-compute the feature columns for the Heisenberg frame with the
correct $L$, then re-evaluate the stored forest. Cheap. But see §3.1: even fixed, a model
trained at $L=16$ and tested at $L=12$ is confounded, because IPR, $S_\mathrm{PCA}$ and
all CE quantities are $L$-dependent.

**Guard worth adding regardless:** one assertion in `fwht_coeffs_in_cluster_col_order`,
`assert len(logpsi) == 2**hilbert.size`, removes this entire class of silent failure.

### 1.2 `get_J1J2kagome_Hamiltonian` does not build a kagome model **[verified]**

`Netket/tunable_Heisenberg_models.py:132` sets `J1_bonds = [a1, a2]`, i.e. translations by
the Bravais vectors, of length 1. On kagome with `site_offsets = [[0,0], [1/2,0],
[1/4,√3/4]]` the **nearest-neighbour distance is 1/2**. A translation by $a_1$ or $a_2$
therefore connects a site to the *same sublattice* in the adjacent unit cell:

```
translation a1: site 0 -> site 6, |t| = 1.0000
translation a2: site 0 -> site 3, |t| = 1.0000
```

So the "$J_1$" bonds form three independent triangular lattices, one per sublattice. The
`J2bonds` vector `0.5*(a1 - a2)` *is* a genuine NN kagome bond (length 1/2), but it is
skipped for `subl == 0` (line 144). Net effect: **sublattice 0 is completely decoupled**
from the other 8 spins, and the model is not kagome at any $J_2$.

For reference, the same check on the other two lattices: triangular uses $a_1$, $a_2$,
$a_1-a_2$, all of length 1 — these are exactly the three NN bond directions, so
`get_J1J2triangular_Hamiltonian` is correct and matches the tunable-frustration
construction it cites. The square $J_1$ bonds via `g.edges()` are also correct.

### 1.3 `get_J1J2Square_Hamiltonian` $J_2$ bonds are the wrong set, and double counted **[verified]**

Line 73: `NNs = np.where(g.distances()[i] == 2)`. `Lattice.distances()` returns **graph hop
distance**, not Euclidean distance. For the default 3×4 PBC lattice, site 0's hop-distance-2
set is

```
hop-dist-2 neighbours of site 0: [2, 5, 7, 9, 11]
  j= 2  delta (0,2)   <- straight second neighbour, not a diagonal
  j= 5  delta (1,1)
  j= 7  delta (1,3) ≡ (1,-1)
  j= 9  delta (2,1) ≡ (-1,1)
  j=11  delta (2,3) ≡ (-1,-1)
```

The four diagonals are right, but site 2 — the straight $(0,\pm2)$ neighbour — is included
as well. (It appears only along $y$: with $L_x=3$ and PBC, $x\pm2 \equiv x\mp1$, which is
already a NN and therefore at hop distance 1. So the spurious bond set is anisotropic,
which also breaks the lattice symmetry.)

Second issue: $J_1$ is summed over `g.edges()` (each bond once) whereas $J_2$ loops over
all `i` and all `j in NNs` — each unordered $J_2$ pair is visited twice. So the effective
ratio is $2J_2/J_1$, not $J_2/J_1$.

A fix that avoids both problems is to select on the Euclidean displacement (as the
triangular and kagome builders already do) rather than on `g.distances()`, e.g. translations
by $(1,1)$ and $(1,-1)$ applied once per site.

### 1.4 The forest MSE is selection-biased

`datamine/perfvsclusterexpn.ipynb` cell 8 enumerates **all 63 non-empty feature subsets**,
fits a `RandomForestRegressor` on each with a fixed `train_test_split(random_state=42)`,
and keeps `argmin(test MSE)`. Cell 9 then reports that same minimum as the model's error.
The test set has been used for model selection, so the quoted number — the
$1.5\times10^{-4}$ in `main.tex` §"Random forest model" — is optimistic by an unknown
amount, and the "best feature set" is partly a draw from 63 noisy estimates.

Cheap fix: a three-way split (select on validation, report on a test set touched once), or
nested CV. With ~700 samples and 5–6 features this costs minutes.

Related fragility: cell 9 re-derives `traindf` from `best_features + ['Delta_E']` and then
filters on `traindf['E_gap']` — a column that only exists if `E_gap` happened to be selected
into `best_features`. The cell therefore raises `KeyError` for any other winning subset. Worth
restructuring so the reported scatter is guaranteed to correspond to the fitted model.

### 1.5 CE features are not invariant under the state's gauge freedom

`CEEntropy` and `cluster_expansion_decay` sort the full coefficient vector including the
**constant Walsh coefficient** $c_\emptyset$, which is just the mean of $\log\psi$. Two
consequences:

- Rescaling $\psi \to \lambda\psi$ shifts $c_\emptyset$ by $\log\lambda$ and leaves every
  other coefficient untouched — so both features depend on the arbitrary normalisation.
  Since $c_\emptyset$ is typically the largest coefficient by far, it dominates the sorted
  list and the `polyfit` slope in `cluster_expansion_decay`.
- The global phase returned by `lanczos_ed` is arbitrary and enters
  $\mathrm{Im}\,c_\emptyset$.

**Fix**: before transforming, fix a gauge (e.g. normalise so $\max_s|\psi(s)| = 1$ and
rotate the phase so the largest-amplitude component is real positive), then **drop
$c_\emptyset$** from the spectrum. This should be part of the definition in the note, not
just the code.

### 1.6 States are stored in `complex64` — infidelity floor ≈ $10^{-7}$

`DEFAULT_PARAMS["param_dtype"] = np.complex64` (`Netket/optimization.py:91`), and
`write_output` (line 204) stores `vstate.to_array()` at that dtype. Single precision has
~$10^{-7}$ relative accuracy, so infidelities and $\Delta E$ below roughly $10^{-7}$ are
not resolved — a hard floor on the dynamic range of the very quantity being regressed.
`complex128` is the safer default (it also improves SR conditioning); the cost at $L=16$
full-summation is small.

### 1.7 Miscellaneous

- **Negative-gap filter applied inconsistently.** `df[df.E_gap > 0]` appears in
  `datamine/NN_model_regression.ipynb` and `ana_notebooks/optimal_rotations.ipynb`, but
  **not** in `datamine/firstpass.ipynb`. See §2.5 — it belongs in the loader.
- **Infidelity definition mismatch.** `main.tex` §"Structure of the data set" defines
  $\mathcal{I} = 1 - |\langle\psi_\mathrm{RBM}|\psi_0\rangle|$; the code computes
  $1-|\langle\cdot\rangle|^2$ (`Netket/analysis.py:350`). Pick one; the squared version is
  the conventional one.
- **`sqrt(pi)/3` vs `pi/sqrt(3)`.** `ClusterExpansion/collect_res.ipynb` cell 8 uses
  `sqrt(pi)/3` ≈ 0.591 as the uniform-phase reference, where the note derives
  $\pi/\sqrt{3}$ ≈ 1.814. The correct constant is used in `phase_nonrandomness` in
  `perfvsclusterexpn.ipynb`, so this looks like a stray typo in one notebook rather than a
  systematic error — worth confirming it did not propagate into a figure.

---

## 2. TODO-by-TODO triage

In the order they appear in `main.tex` §"todo".

### 2.1 "Are there zeros in the random Ham psis?"

**Why it matters.** Every CE feature takes `np.log(psi)`. An exact zero gives $-\infty$;
an amplitude of $10^{-14}$ gives $-32$, a value far outside the range of the rest of the
vector, which then dominates the Walsh spectrum. So this question is a *precondition* for
trusting any CE-derived quantity, not a side curiosity.

**Expectation.** For this ensemble — Gaussian $J_{\alpha\beta}$ with a uniform field, no
imposed symmetry — exact zeros are non-generic (they are a measure-zero condition on the
couplings). But *exponentially small* amplitudes are guaranteed: a $2^{16}$-component
normalised vector has typical smallest amplitude far below $10^{-7}$, i.e. below the
`complex64` storage floor of §1.6, at which point the stored value is numerical noise and
its log is meaningless.

**Recipe.** Add diagnostic columns in the loader: `min_abs_psi`, `n_below_1e-12`,
`frac_below_eps(psi, eps)`. Histogram them over the dataset, split by `E_gap > 0`. Then
decide the regulator in §2.2 with that distribution in hand. This is a 20-line addition to
`Netket/analysis.py` plus one plot.

Note that the *symmetric* ensembles do have forced zeros: any $SU(2)$- or $U(1)$-symmetric
$H$ has $\psi_0$ supported only on one $S_z$ sector, so a strict majority of components are
exactly zero. That is the situation §2.4 has to deal with.

### 2.2 "Properly define cluster expansion of complex amplitudes which can be zero"

This is the central definitional problem, and five other TODOs depend on it. Four options,
in increasing order of principle:

**(a) Split modulus and phase.** Define the CE on $\log|\psi(s)|$ (real, no branch
ambiguity, well-defined wherever $\psi\neq0$), and treat the phase separately — either as
a second Walsh expansion of the unwrapped $\varphi(s)$, or via the phase-only overlap
already implemented as `get_phase_overlap` in `ClusterExpansion/collect_res.ipynb` cell 7.

  This is already what `check_time_evolution/cumulants_time_evolution.ipynb` does
  (`logabs = np.log(np.abs(psi_t[k]))`), and it is presumably why the Fermi-Dirac fits of
  the rank spectrum there come out so clean ($R^2$ close to 1) while the CE features on the
  random-Hamiltonian dataset are noisy. That contrast is itself evidence for (a).

**(b) $\tau$-continuation with winding numbers** — the scheme already sketched in `main.tex`
§"Defining complex phases of the cluster expansion". Track
$\psi(s,\tau) = \langle s|e^{-H\tau}|\psi_\mathrm{start}\rangle$ from a $\tau=0$ reference
with no zeros (the uniform state), take small enough steps that $\varphi$ never jumps by
$2\pi$, and accumulate $m(s,\tau)$. This gives a genuinely single-valued $\log\psi$ and is
the *right* definition — but it costs one imaginary-time trajectory per instance, so it
is not viable for the full dataset.

**(c) Regularise:** expand $\log(|\psi| + \delta)$ with a fixed $\delta$, and report the
$\delta$-dependence of every derived quantity. Cheap; honest as long as the dependence is
shown.

**(d) Linear (no-log) Walsh transform** — already implemented as the `MOD*` functions in
`ClusterExpansion/cluster_expansion.py:218-248`. Always well-defined, but it is no longer a
cumulant expansion: there is no exponential resummation, so a product state is not a
single coefficient. Useful as a **baseline**: any claim that the cumulant expansion is
informative should be shown against it.

**Recommendation.** Adopt **(a)** as the working definition, keep **(d)** as the baseline
in every comparison plot, and run **(b)** on a handful of instances to check that (a)
tracks it. In all cases, fix the gauge and drop $c_\emptyset$ first (§1.5).

Whatever is chosen should be written into `main.tex` as *the* definition — at the moment
`ClusterExpand_infid.py` transforms `psi` directly while `rand_hams_cluster_expn.ipynb`
transforms `np.log(psi)`, and `collect_res.ipynb` plots both as "exact" and "exp" without
the note saying which is meant.

### 2.3 "Define truncated cumulant expansion by sorting for the real part of the cumulants"

**Small.** Mirror `_get_topkIMAG_indices_jit` (`cluster_expansion.py:184`) with
`jnp.real` in place of `jnp.imag`.

**Better, same effort.** There are currently three near-identical copies of the
compress/reconstruct machinery (`compress_and_reconstruct_cached`,
`..._cached_IMAG`, `MODcompress_and_reconstruct_cached`) differing only in the sort key
and in whether the result is exponentiated. Collapse to one function with `sort_key ∈
{"abs", "real", "imag"}` and `log_space: bool`. Then the interesting experiment becomes
one loop: for the same set of instances, compare the truncation curves under
$|c|$, $|\mathrm{Re}\,c|$, $|\mathrm{Im}\,c|$.

**Physical motivation to state in the note:** $\mathrm{Re}\,c_A$ controls $\log|\psi|$ and
$\mathrm{Im}\,c_A$ the phase, so the comparison directly asks whether RBM difficulty tracks
amplitude structure or sign structure — which is the note's central question, restated in
the CE language.

### 2.4 "Cumulant expansion for Heisenberg with $\tilde\psi = (\psi^\mathrm{reduced},\dots,1)$"

**The obstacle.** For an $SU(2)$-symmetric $H$, $\psi_0$ lives in the $S_z=0$ sector, of
dimension $\binom{L}{L/2}$ — never a power of two — so the FWHT cannot be applied to the
reduced vector, while the full vector is exactly zero outside the sector so $\log\psi$ is
$-\infty$ there.

**The literal implementation.** Choose $\tilde L = \lceil\log_2\binom{L}{L/2}\rceil$, fix a
canonical ordering of the sector basis states, pad to $2^{\tilde L}$ with ones (so the
padding contributes $\log 1 = 0$), and FWHT. For $L=12$: $\binom{12}{6}=924 \to \tilde L=10$,
1024 components, 100 of them padding.

**The caveat, which should go in the note.** After this embedding the Walsh monomials
$s_{A'}$ index subsets of the $\tilde L$ *artificial* bits, not subsets of physical sites.
So the resulting coefficients are not cluster amplitudes, the notion of "$k$-body" is lost,
and the whole expansion depends on the arbitrary ordering of sector states. It is still a
legitimate *compressibility* measure — "how many Walsh coefficients does this vector need"
— but it is no longer a cluster expansion, and the note should not call it one.

**The alternative that keeps locality.** Stay on the full $2^L$ space and expand
$\log(|\psi| + \delta)$ with $\delta$ fixed (§2.2c). The sector structure then shows up as
a specific, identifiable set of large coefficients (the ones enforcing $\sum_i s_i = 0$)
rather than being hidden. The truncated state can be projected back into the sector before
computing infidelity. The price is $\delta$-dependence, which must be shown.

**Recommendation:** do both on the same handful of states and compare the resulting
complexity measures. If they agree on the ordering of instances, use the cheap one; if they
disagree, that disagreement is a result worth reporting.

### 2.5 "Cull all results where gap was negative"

`E_gap = exact_energies[1] - exact_energies[0]` (`Netket/analysis.py:115`), with the comment
"already sorted from lanczos_ed". A negative gap means that assumption failed for that row.

**What I checked.** The natural suspicion — from the `9912b99` commit message "need to
hermitize the H_rot before optimization" — is that the rotated Hamiltonian is
non-Hermitian, so `lanczos_ed` falls back to a non-symmetric solver returning unsorted
eigenvalues. At netket 3.19.2 this is **not** what happens **[verified]**:

```
plain   is_hermitian: True     max|H - H†| = 0.0
rotated is_hermitian: True     max|H - H†| = 1.8e-15
plain energies [-22.254 -12.687 -11.303  -9.999]
rot   energies [-22.254 -12.687 -11.303  -9.999]   (identical, sorted, match dense eigh)
```

So the more likely causes are **Lanczos non-convergence** at $k=10$ for near-degenerate
spectra, or a different netket version on the cluster where the data was produced.
I would not assume either without checking one of the affected files.

**Three things to do, in order:**

1. **Move the filter into the loader.** `load_outputs_to_dataframe` and
   `load_outputs_to_dataframe_mult_thread` (`Netket/analysis.py:39, 155`) should take
   `validate=True` and drop (or flag) rows with `E_gap < 0`. Currently each notebook is
   responsible for remembering, and `firstpass.ipynb` does not.
2. **Recognise that ψ₀ is suspect too.** If the eigenvalues came back unsorted, then
   `psi_0 = eigenstates[:, 0]` (`optimization.py:226`) is not necessarily the ground state.
   These rows should be *excluded*, not merely have their gap set to NaN — and any figure
   made without the filter should be regenerated.
3. **Make the criterion principled.** At generation time, store the ED residual
   $\|H\psi_0 - E_0\psi_0\|$ in the h5 alongside the energies. Then "cull" becomes a
   threshold on a physically meaningful quantity rather than a symptom check, and it also
   catches the rows where Lanczos converged to the wrong state *without* producing a
   negative gap.

Also worth reporting in the note: **how many** instances are affected. If it is 2% the
conclusions are safe; if it is 20% the ensemble statistics change.

### 2.6 "Save logpsi, not psi!"

**For the variational state this is easy and worth doing.** The RBM *is* a
$\log\psi$ machine: `vstate.log_value(hilbert.all_states())` gives $\log\psi$ directly,
with no branch ambiguity and no underflow, whereas `write_output` currently stores
`vstate.to_array()` (`optimization.py:204`) and every downstream CE analysis then applies
`np.log` to it. Add a `logpsi` dataset next to `psi` — keeping both preserves backward
compatibility with the existing h5 files.

Combine with the `complex128` change of §1.6 while touching this code.

**For $\psi_0$ from ED there is no such shortcut** — ED returns amplitudes, and taking
their log runs straight into §2.2. So this TODO fixes half the problem; the other half is
the definitional one.

**Note on the existing data:** the datasets are already written. Rather than regenerating
everything, add the field going forward and record in the note which datasets have it.

### 2.7 "Complexity hypothesis: $N$ for which $I(\psi^{(N)}_0,\psi_0) < \epsilon$"

**This is already computable from data you have.** `ClusterExpansion/ClusterExpand_infid.py`
produces `infidels_exact_opt_mat[i, j]` — instance $i$, truncation level
`compr_idx_list[j]` (100 log-spaced values from $2^1$ to $2^{16}$). That matrix *is* the
family of curves $\mathcal{I}(N)$; $S_\epsilon$ is just its first crossing of $\epsilon$:

```python
def S_eps(infid_curve, N_list, eps):
    below = np.where(infid_curve < eps)[0]
    if len(below) == 0: return np.nan          # never reaches eps
    j = below[0]
    if j == 0: return N_list[0]
    # log-linear interpolation between the bracketing points
    x0, x1 = np.log(N_list[j-1]), np.log(N_list[j])
    y0, y1 = np.log(infid_curve[j-1]), np.log(infid_curve[j])
    return np.exp(x0 + (np.log(eps) - y0) * (x1 - x0) / (y1 - y0))
```

**Suggestions for how to report it:**

- Use several $\epsilon$ (e.g. $10^{-1}, 10^{-2}, 10^{-3}$) rather than one. If the ranking
  of instances by $S_\epsilon$ is $\epsilon$-dependent, the measure is not capturing a
  single scale and that is worth knowing.
- The natural normalised predictor is $\log_2 S_\epsilon / L$, which is $O(1)$ and
  comparable across system sizes — important given §3.1.
- The reference line already drawn in the notebooks is the RBM parameter count,
  $N_\mathrm{params} = L\cdot\alpha L + L + \alpha L = 288$ for $L=16, \alpha=1$. (The
  notebooks use `16**2 + 16*2 = 288` in some places and `16**2+32 = 288` in others — same
  number, but `get_N_params` in `rand_hams_cluster_expn.ipynb` cell 6 is the canonical one.)
  The sharp question is whether $\mathcal{I}_\mathrm{RBM} \approx \mathcal{I}_\mathrm{CE}(N_\mathrm{params})$,
  i.e. whether the RBM performs like an optimal $N_\mathrm{params}$-term cumulant truncation.
  The scatter plot in `collect_res.ipynb` cell 17 is exactly this comparison and deserves to
  be a figure in the note.
- Note the curves currently saturate: many instances never reach small $\epsilon$ even at
  $N=2^{16}$, because of the log-branch problem of §2.2. Settling that definition should
  clean these curves up considerably.

### 2.8 "CE entropy"

Implemented as `CEEntropy` in `datamine/perfvsclusterexpn.ipynb` cell 3. Three changes:

1. **Move it into a shared module** (`Netket/analysis.py`, or better a new `features.py`)
   with no hardcoded $N$ — take the Hilbert space as an argument, or infer
   $L=\log_2(\mathrm{len}(\psi))$ and assert it is an integer. This removes §1.1 at the root.
2. **Fix the gauge** (§1.5): drop $c_\emptyset$, normalise, fix the global phase.
3. **Pin it with three cheap tests**, which double as documentation of what it means:
   - product state $\Rightarrow$ $S_\mathrm{CE} = 0$ (one nonzero coefficient per site,
     none higher);
   - uniform state $\Rightarrow$ $S_\mathrm{CE} = 0$ (all weight in $c_\emptyset$, which is
     dropped — so this test also verifies the gauge fix);
   - Haar-random state $\Rightarrow$ $S_\mathrm{CE} \to 1$ (flat spectrum, by the
     $1/\log k$ normalisation already in the definition).

   If the current implementation does not give ≈0 for a product state, that is a bug, and
   the tests will find it in seconds.

Also worth deciding explicitly: the current definition normalises $|c_A|$ by $\sum_A|c_A|$
(an $\ell_1$ normalisation) and takes a Shannon entropy. A participation-ratio /
Rényi-2 version of the same spectrum is cheaper to interpret and less sensitive to the long
tail of tiny coefficients. Reporting both costs nothing.

### 2.9 "Conditional mutual information (Preskill et al.)"

I am not certain which paper is meant — most likely the line of work on conditional mutual
information and the Markov length in Gibbs states **(not sure)**. Worth confirming the
reference before building on its framing.

**The measurement itself is cheap at $L=16$** and is a genuinely new kind of predictor: all
current predictors (IPR, $S_\mathrm{PCA}$, $\sigma_\varphi$, uniform overlap, CE entropy)
are basis-dependent functions of the amplitude list, whereas CMI is a property of the
*entanglement structure*. That makes it a good test of the note's framing question — whether
NQS performance is about entanglement or about sign/amplitude structure — because it is the
one predictor that would side with entanglement.

**Concrete recipe** for the 4×4 lattice:

- Tripartition $A | B | C$ with $B$ a "shielding" region of width $w$: e.g. $A$ = row 0,
  $B$ = rows $1..w$, $C$ = the rest.
- $I(A{:}C|B) = S_{AB} + S_{BC} - S_B - S_{ABC}$, each $S$ from the eigenvalues of a reduced
  density matrix obtained by reshaping $\psi$ and taking an SVD — the machinery already
  exists in `check_time_evolution/ed_definitions.py:96` (`entanglement_entropy`), which just
  needs generalising from a contiguous bipartition to an arbitrary site subset.
- Sweep $w$; the decay rate of $I(A{:}C|B)$ with $w$ defines a Markov length $\xi_M$.
- **Hypothesis to state and test:** RBM performance tracks *short* $\xi_M$ — an RBM with
  $\alpha=1$ has each hidden unit coupled to all visible units, but the learned weights are
  effectively short-ranged, so a state that is approximately a Markov chain at short range
  should be easy.

Caveat: at $L=16$ in 2D, the region geometry matters a lot and there are few distinct
widths available, so treat this as exploratory rather than a clean scaling analysis.

### 2.10 "Find optimal basis for a stabilizer Hamiltonian"

**Frame this as a ground-truth test of the pipeline, not just another experiment.** A
stabilizer Hamiltonian is the one case where the answer is known analytically: there exists
a Clifford $C$ with $C^\dagger|\psi_0\rangle$ a product state, so in that basis the exact
cumulant expansion terminates after $L$ single-site terms, uniform-state overlap can be made
$O(1)$, and the RBM should reach machine precision. Any pipeline that cannot find this has a
problem — which makes it the sharpest available test of both the rotation optimiser and the
CE truncation.

**Important caveat, which is itself the informative part.**
`rotation.get_rotation_objective` (`Netket/rotation.py:49`) optimises three angles for a
**single global** one-site rotation: `apply_local_rotation_to_state` (line 36) loops over
sites applying *the same* `U_single` to each. For a general stabilizer state the
disentangling Clifford is site-dependent and typically *entangling* (CZ gates). So the test
will fail with the current search space, for a well-understood reason. Recommended
sequencing:

1. Start with a stabilizer model whose disentangler *is* within reach — e.g. the 1D cluster
   model, or a product of NN CZ gates applied to $|+\rangle^{\otimes L}$ — and confirm the
   CE truncation becomes exact at $N = $ number of stabilizer generators.
2. Then enlarge the search space: site-dependent single-qubit rotations, then a brickwork of
   two-site Cliffords (which is exactly §2.12).
3. Then increase $L$ and check that the basis found at small $L$ (or its translation-invariant
   extension) still performs best — the actual TODO.

This makes a good candidate for the repo's first real test file.

### 2.11 "Heisenberg: rotate every other site (breaks global $SU(2)$)"

**Implementation.** `apply_local_rotation_to_state` and
`construct_hamiltonian_bonds_unitary` (`Netket/optimization.py:47`) both take a single `U`.
Generalise to a list of per-site $U_i$; a sublattice pattern is then just
`[U_A if is_even(i) else U_B for i in sites]`. This is a small, self-contained change and it
also unlocks §2.10 step 2.

**Built-in validation, which makes this worth doing first among the Heisenberg items.** On a
*bipartite* lattice the answer is known: the $\pi$ rotation about $y$ on one sublattice is
the Marshall sign rule, which makes the AFM Heisenberg ground state real and positive — hence
maximal uniform-state overlap. So the optimiser, run on a bipartite $J_1$-only model, *must*
recover (approximately) $(\alpha,\beta,\gamma) = (0,\pi,0)$ on sublattice B. If it does not,
the objective or the optimiser is broken, and everything downstream is suspect.

**The physics.** On triangular and kagome no such rotation exists — that is precisely what
frustration means here. The interesting quantity is then the *residual*: how close to the
uniform state can a sublattice rotation get, as a function of $J_2$, and does that residual
predict $\Delta E$ better than the un-rotated features do? That is a direct test of the
note's central hypothesis in the one regime where it currently fails.

Note that `scipy.optimize.minimize` with `x0=[0,0,0]` (as in
`Netket/run_coup_optimal_basis.py:42`) is a local optimiser on a periodic, multi-modal
objective — it will not find the Marshall solution from the identity. Use several random
restarts, or a coarse grid followed by refinement.

### 2.12 "Sample random translation-invariant two-body Cliffords"

**Why this is the cleanest experiment on the list.** A Clifford conjugation maps Paulis to
Paulis, so $C^\dagger H C$ with $H$ of the form of Eq. (5) is *still* a two-local Hamiltonian
in the same model class — possibly with longer range if $C$ acts on bonds, but still
expressible in the existing `construct_hamiltonian_bonds` machinery. So the transformed
problems can go straight through the existing pipeline with no new infrastructure.

**Design.**

- The two-qubit Clifford group has 11520 elements; sampling uniformly is standard, or sample
  from a generating set (H, S, CZ) to keep the layer shallow.
- Apply as a translation-invariant brickwork layer (commuting with translation by 2), so the
  transformed $H$ retains a translation symmetry and the ensemble stays comparable to the
  unrotated one.
- For each sampled $C$, record: (i) uniform-state overlap of $C^\dagger|\psi_0\rangle$ —
  **cheap, no RBM needed**; (ii) CE truncation infidelity at fixed $N = N_\mathrm{params}$ —
  also cheap; (iii) optimised RBM energy — expensive.
- **The question:** do (i) and (ii) select the same $C$ as (iii)?

This is the sharpest possible statement of the note's thesis. If a cheap, state-only
quantity picks the basis that the expensive optimisation would have picked, that is the
result; if it does not, the counterexamples are the interesting instances. Either way it is
publishable, in the spirit of the note's own §"Proposal Summary".

**Cost control:** (i) and (ii) can be evaluated over hundreds of Cliffords per Hamiltonian
for the price of one RBM optimisation, so run a broad scan on the cheap quantities first and
spend the RBM budget only on the extremes (best and worst predicted).

---

## 3. Further suggestions

### 3.1 Generalisation, not more features

The result as it stands is essentially single-feature: uniform-state overlap dominates
(`mutualinfos_fmv1.png`, and the note says so explicitly). That is a *good* result — it is
simple and it has a mechanism (§"Instructive Example" with $H_{XY}$ vs $H_{XZ}$ is genuinely
convincing). The way to strengthen it is an honest generalisation test, not a longer feature
list.

- **Match the system size.** The Heisenberg test set is $L=12$; training was $L=16$. IPR,
  $S_\mathrm{PCA}$, CE entropy and $S_\epsilon$ all scale with $L$, so the model is being
  asked to extrapolate in a variable it never saw vary. Either generate $L=16$ Heisenberg
  instances (3×4 → 4×4 is one line in `get_J1J2_Hamiltonian`, at 16× the ED cost — still
  cheap) or use only $L$-normalised features ($\log\mathrm{IPR}/L$, $\log_2 S_\epsilon/L$,
  $S_\mathrm{PCA}$ is already normalised).
- **Report rank correlations alongside RF importances.** Spearman $\rho$ between each
  predictor and $\log\Delta E$ is interpretable, has an error bar, and does not degrade when
  features are correlated. RF feature importance on correlated features is unstable — it
  splits credit arbitrarily between them — and it is currently carrying a lot of the note's
  argument.
- **Report the null.** With one dominant feature, quote how much a multi-feature model beats
  uniform-overlap-alone. If the gain is small, say so; per the note's own §"Proposal
  Summary", a negative result here is still a result.

### 3.2 Ensemble design

`main.tex` §"First impressions" observes that the predictor distributions are too narrow to
draw conclusions. That is a **sampling-design** problem, and drawing more samples from the
same ensemble will not fix it. Options, roughly in order of effort:

- Vary $\sigma$ per bond type, or draw $\sigma$ itself from a broad (e.g. log-uniform)
  distribution, so the ensemble spans weak-disorder to strong-disorder regimes rather than
  concentrating at $|\vec{\mathcal{J}}| \approx 4.5$ as the note's own footnote observes.
- Interpolate towards simple models: $H = (1-\varepsilon)H_\mathrm{simple} + \varepsilon
  H_\mathrm{rand}$. `Netket/run_coup_epsilon_TFI.py` already does exactly this with a TFI
  base and $\varepsilon \in [0.01, 0.3]$ — it appears not to have been carried into the
  analysis, and it is the most direct way to get a *broad, controlled* spread of predictors.
  This looks like the highest-value quick win in the repo.
- Mix in the $SU(2)$-symmetric ensemble of Eq. (6) as a labelled subset of the *same*
  dataset, so symmetric and generic instances are compared within one analysis rather than
  as train/test.

### 3.3 Code consolidation

The failures in §1.1–§1.3 are all the same shape: a definition living in a notebook,
duplicated, drifting, failing silently. Three concrete steps:

- **One `features.py`**, imported by every notebook, holding all hypothesis functions with
  explicit $L$ handling. Today `uniform_state_overlap` is defined **twice** in
  `Netket/analysis.py` (lines 353 and 422 — identical, but the second silently shadows the
  first), `infidelity` is re-defined inside `ClusterExpansion/ClusterExpand_infid.py`
  shadowing the imported one, and `CEEntropy`/`CEdecay`/`phase_nonrandomness` live only in a
  notebook.
- **Assertions at the boundaries.** `assert len(psi) == 2**hilbert.size` in the FWHT helper;
  `assert np.all(np.diff(exact_energies) >= 0)` in the loader. Both are one line and both
  would have caught a real bug in this repo.
- **A `tests/` directory** with the §2.8 CE-entropy tests, the §2.10 stabilizer test, and a
  lattice test asserting NN/NNN bond counts and lengths for each of the three $J_1$-$J_2$
  builders (which would have caught §1.2 and §1.3 immediately).

### 3.4 Reproducibility

The h5 files record the optimisation parameters as attributes, but not the **git commit** or
the **netket version**. Given §2.5 — where the leading hypothesis for the negative gaps is a
version difference between the cluster and the local machine — this is not hypothetical.
Two extra `f.attrs` entries in `write_output`.

Related: `write_output` swallows the log-file read failure and stores `en_var = False`
(`optimization.py:221-222`), which then flows into the dataframe as a valid-looking value;
and both `run_coup_optimal_basis.py` and `run_tunable_frustration_Heisenberg.py` end with a
bare `except Exception: print("failed")` that discards the traceback. When a cluster array
job silently produces fewer files than expected, there is currently no way to find out why.

### 3.5 On the note itself

- §"Chosen problem restrictions" / §"Outline of the project" describe 1D and $K$-range
  interactions, while the actual dataset is a 2D 4×4 PBC square lattice with a single
  $J^{(x)}$ and $J^{(y)}$ matrix (`python/utils/helpers.py::makeJpairs`, 21 parameters as in
  the appendix). The appendix Eq. (17) has it right; the main text should be brought in line
  so a reader knows what was actually computed.
- The `Flowchart.tex` decision tree is a nice summary of the qualitative claim and is not
  currently referenced from `main.tex` — worth including as a figure, with the `Ex: ?` box
  filled in once §2.12 produces a concrete example.
- §"Cluster expansion as a complexity measure" writes $\xi(s) = H^{\otimes L}\psi(s)$ but the
  text and the code both mean the transform of $\log\psi$. Fix the typo.

---

## 4. Suggested order of work

Ordered by (impact on existing conclusions) ÷ (effort):

1. **§1.1 + §3.3 assertions** — re-compute the Heisenberg features at the correct $L$ and add
   the length assertion. Hours. Determines whether the note's Heisenberg section stands.
2. **§1.2, §1.3** — decide whether to fix the kagome and square builders. Note that fixing
   them invalidates `data/tunable_frustration_Heisenberg_rbm`, so this is a call about
   whether the $J_1$-$J_2$ comparison matters to the paper. If it does, the data must be
   regenerated; if the triangular family alone suffices, that one is correct as written.
3. **§2.5** — quantify how many rows have a negative gap, and move the filter into the loader.
   Hours. Determines whether any other figure needs regenerating.
4. **§1.4** — re-do the model selection honestly and re-quote the MSE. Hours.
5. **§2.2** — settle the CE definition. This is the intellectual bottleneck: §2.3, §2.4,
   §2.7, §2.8 and the CE half of §2.12 all inherit from it. Days.
6. **§2.7** — $S_\epsilon$ from data already on disk, once §2.2 is settled. Hours.
7. **§2.11** — sublattice rotations, with the Marshall check as validation. Days, and it is
   the most likely route to fixing the Heisenberg failure.
8. **§2.12** — the Clifford scan. The headline experiment, but it should come after the
   cheap predictors are trustworthy.

§2.9 (CMI) and §2.10 (stabilizer) are more open-ended; both are worth doing but neither
blocks anything else.
