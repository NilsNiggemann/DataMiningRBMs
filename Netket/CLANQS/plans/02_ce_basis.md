# Plan: basis optimized for the truncated-cumulant-expansion infidelity (5th basis `CE`)

Extends `01_walshness.md`. Same models, same RBM settings, same notebook `Walshness_Kitaev.{py,ipynb}`.

## 0. Decisions (user, 2026-10-02)
- **Basis family:** Clifford + local, $U = V\,C$ (same order as `cliff→local`), with both $C$ and $V$
  chosen by the CE objective (beam search over $G^{ab}_{ij}$, inner local solve).
- **Truncation:** $K$ = number of RBM parameters = `get_N_params(18, 5)` = 18·90 + 18 + 90 = **1728**.
- **log ψ:** principal branch with an ε regulator (below).
- **Models:** both (`kitaev_honeycomb`, `kitaev_honeycomb_gapped_3`).

## 1. Objective
For a state $\phi$ (walshness.py bit convention):
1. Gauge: $\phi \leftarrow \phi\,\overline{\phi_{s^*}}/|\phi_{s^*}|^2$ with $s^*=\arg\max|\phi|$
   (max amplitude = 1, real positive; SUGGESTIONS.md §1.5). Needed because the principal log
   is not invariant under a global phase.
2. $\ell(s) = \log\max(|\phi(s)|, \varepsilon) + i\arg\phi(s)$, $\varepsilon = 10^{-12}$.
3. $c = \mathrm{FWHT}(\ell)$; keep the $K$ largest $|c_r|$ **including** $c_\emptyset$ (as in
   `cluster_expansion._get_topk_indices_jit`), zero the rest.
4. $\phi_K = \exp(\mathrm{FWHT}^{-1} c_K)$; $\;I_{CE}(\phi;K) = 1 - |\langle\hat\phi|\hat\phi_K\rangle|^2$
   (normalized).

Top-K magnitudes are invariant under qubit relabeling and global bit flips, so the
walshness.py ordering gives the same $I_{CE}$ as the NetKet ordering of `cluster_expansion.py`
(checked by T15). The repo implementation is not used in the hot loop: it passes a
262144-entry static tuple to `jit` each call and is not set up for gradients.

**Pre-check (done, existing bases, $I_{CE}$ at K = 360 / 1728 / 6912 / 27648):**

| model | naive | local | cliff→local | local→cliff |
|---|---|---|---|---|
| kitaev | .93/.77/.75/.55 | .99/.96/.87/.73 | **.65/.31/.08/.004** | 1.0/.97/.89/.67 |
| gapped_3 | .92/.95/.91/1.0 | .87/.86/.82/.66 | .89/**.81**/.70/.76 | = local |

- $I_{CE}$ is **not monotone in K**. Truncating the phase part of a wrapped principal log can
  make things worse. That is a property of the principal-log CE (SUGGESTIONS.md §2.2 option (a), modulus-only, would avoid it), not a bug.
- For pure Kitaev, the Walshness `cliff→local` basis is already by far the best CE basis.

## 2. Inner solver `ce_local(phi, K, inits)` → min over $V=\bigotimes_j V_j$
- Full SU(2) per site (3 Euler angles, 54 parameters). Unlike $W^Z$, $I_{CE}$ is not
  invariant under the X-diagonal phase, so the axis-only parametrization is not enough.
- JAX `value_and_grad`, x64, top-K index set from `jax.lax.top_k` under `stop_gradient`
  (gradient within the current top-K cell); `scipy.optimize.minimize(L-BFGS-B, jac=True)`,
  `maxiter` 150 in search mode and 500 in final mode.
- Benchmark: about 0.5 s per value+grad on 1 CPU with `argsort` (n=18). `top_k` should be faster;
  measure it.
- Inits: identity, Walshness axes → unitaries (`wn.axes_to_unitaries`), parent $V$, and random
  SU(2) (8 in search mode, 32 in final mode).
- The objective is non-smooth: the top-K set jumps, and amplitudes cross the branch cut.
  Restarts are the only mitigation. Report the restart spread.

## 3. Outer search (mode `"CE"` of `wn.beam_search`)
- Gates `wn.G_gates(n)` (1377 per parent), B=4, D_max=n, patience 2, objective
  $J(C)=\min_V I_{CE}[VC\psi]$.
- **Screening:** the Walshness LB is not a bound for $I_{CE}$. Replace it by a forward pass
  $I_{CE}[V_{\text{parent}}\,g\,\phi_{\text{parent}}]$ for all 1377 children (cheap, no grad),
  then inner-solve the best 16 per parent.
- **Extra candidate:** CE-local solve on $C_W\psi$ with $C_W$ the Walshness `cliff→local` word
  (one final-mode solve). The `CE` basis is the better of beam result and this candidate; report both.
- Cost estimate: per depth ~5 k screens + 64 children × 3 inits × ~150 grads, on 28 workers,
  ≈ 5 min/depth, ≤ 18 depths ⇒ ≤ 1.5 h per model (not sure, depends on top_k speed and
  patience). Workers: spawn, `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false
  intra_op_parallelism_threads=1"`, ≤ 28 workers, ≤ 30 CPUs total.

## 4. Code
- New `ce_basis.py`, which keeps the JAX import out of `walshness.py`: `ce_infidelity`, `ce_local`,
  `ce_final`, and the screening/eval helpers used by the beam search.
- `walshness.beam_search`: add mode `"CE"`, which plugs in the screen and eval from `ce_basis`
  (lazy import). Store general `Vs` in the result (not axes).
- Notebook `transform()`: use `o["Vs"]` when present, else `axes_to_unitaries(o["axes"])`.
  `local_conjugate` already takes general unitaries. `snap_err` = NaN for `CE`.

## 5. Tests (must pass before the runs; added to `run_tests`)
- **T12** $K=2^n$ ⇒ $I_{CE} < 10^{-12}$ (random 10-qubit state).
- **T13** random product state, rotated by random local SU(2): `ce_local` reaches
  $I_{CE}(K=n+1) < 10^{-8}$ (solver + gradient sanity).
- **T14** ring graph state $\prod_{\langle ij\rangle} CZ_{ij}|+\rangle^{\otimes n}$:
  $I_{CE}(K = 1+n+\#\text{edges}) < 10^{-12}$ in the naive basis.
- **T15** agrees with `ClusterExpansion/cluster_expansion.py` (`fwht_coeffs_in_cluster_col_order`
  + `compress_and_reconstruct_cached`, NetKet ordering) on a gauge-fixed random 10-qubit
  state for several K.
- **T16** JAX gradient vs central finite differences at a point (fixed top-K set).

## 6. Notebook additions
- Section "CE-optimized basis", after the Walshness basis optimization. Cache
  `data/bases_ce_{model}.pkl`, flag `RECOMPUTE_CE`. Print the word, history and restart spread.
- `BASES += ["CE"]`, 5th color. The table gains the column `I_CE_1728` for **all** bases. The
  convergence figure, Hamming histograms and summary table pick up `CE` automatically.
- New figure: $I_{CE}(K)$ vs K ($2^4\ldots2^{18}$), one line per basis, one panel per model,
  vertical line at K = 1728. Saved as `figs/walshness_kitaev_ce_vs_K.png`.
- RBM: the existing loop trains only the missing logs (2 runs × 3 CPUs, ~25–60 min).
- Execute per the `jupytext-sync` skill under `taskset -c 0-29`.

## 7. Report back
- In chat: the convergence figure, the Hamming histograms and the $I_{CE}(K)$ figure (SendUserFile),
  plus the 5-basis table (W1, I_CE_1728, terms, max weight, E_err_last500, infidelity).
- Answer: does the CE-optimized basis beat the Walshness `cliff→local` basis, (i) on $I_{CE}$,
  (ii) on RBM energy error? Single seed ⇒ suggestive only.
- Term growth of $H_U$ for `CE` (general SU(2) $V$ ⇒ growth expected unless $V$ snaps to Clifford).

## 8. Known issues carried over
- Pure Kitaev: Lanczos returned only 2 vectors at E0 (handoff said 3-fold). Optimize for
  `vecs[:,0]` as before. The RBM infidelity is to the 2 found vectors.
- `I_CE` non-monotone in K, and non-smooth (see §1, §2).
