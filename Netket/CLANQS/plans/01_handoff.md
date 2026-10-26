# Handoff: Walshness PoC (plan `01_walshness.md`), state as of 2026-10-02 ~14:40

## Status
- **Done:** `walshness.py` (all numerics + self-tests T1–T11, all pass), `Walshness_Kitaev.{py,ipynb}`
  (jupytext-paired, py:percent), basis optimization for both models (cached).
- **Running (detached, `setsid nohup`, PPID 1):** full notebook execution incl. 8 RBM runs
  (2 models × 4 bases, 7000 iters, parallel, 3 CPUs each on CPUs 0–23, ~35–60 min, started ~14:35).
  Progress: `logs/nbrun.log`; finished when its last line is `EXIT CODE <n>`.
  Check: `pgrep -fc spawn_main` (9 while training), `ls logs/*_walsh_*.json` (8 when done;
  each log is written only at the end of its run).
- **Remaining (plan §10):** once `EXIT CODE 0`, read the executed `.ipynb` outputs and add a
  markdown results/discussion section to `Walshness_Kitaev.py` (table, test pass/fail, convergence
  figure, Hamming histograms, term-growth answer, diagnosis), then sync + re-execute (cheap, all
  cached). If `EXIT CODE` ≠ 0: read the traceback in `logs/nbrun.log`, fix, re-run.

## Files (all in `Netket/CLANQS/`)
- `walshness.py` — conventions, `fwht`, `walshness_Z`, `bloch_vectors`, `lower_bound`,
  `two_site_rdms`, inner solver (`_descent` numba-fused, `intrinsic_walshness`, `final_mode`),
  `axis_to_unitary`, gate sets (`G_gates`, `HSCZ_gates`), `beam_search` (modes `"intrinsic"` =
  cliff→local, `"Z"` = local→cliff), Hamiltonian transforms (`clifford_conjugate` via stim,
  `local_conjugate`), `train_rbm`, `run_tests`. `python walshness.py` runs the tests (~10 s).
- `Walshness_Kitaev.py/.ipynb` — the experiment. Flags `RECOMPUTE_BASES`, `RETRAIN` (default False).
- Caches: `data/gs_*.npz` (Lanczos ground states), `data/bases_*.pkl` (optimized U per basis),
  `logs/{model}_walsh_{basis}.json` (RBM RuntimeLog + metadata + final infidelity).
  `data/` is gitignored; `*.ipynb` is gitignored.
- Figure: `figs/walshness_kitaev_convergence.png`.

## How to run
- Python: `../../check_time_evolution/.venv/bin/python` (NetKet 3.19.2, stim 1.16.0, numba 0.61.2).
  jupytext is not in that venv: use `uvx jupytext --sync Walshness_Kitaev.py`.
- Execute (≤30 CPUs, shared machine!):
  `taskset -c 0-29 ../../check_time_evolution/.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Walshness_Kitaev.ipynb`
  The notebook asserts ≤30 CPUs in its affinity. With caches present it takes ~2 min.

## Decisions / deviations from the plan (keep reporting these)
- **Pure Kitaev 3×3 pbc GS is 3-fold degenerate** (E0 = −14.291503 ×3, next −14.074638).
  User chose: keep it, optimize U for Lanczos `vecs[:,0]`, report W for all 3 vectors (ambiguity
  cell), RBM infidelity measured to the whole GS manifold. gapped_3 GS unique, gap 0.005 only.
- `VMC_SR` is `netket.experimental.driver.VMC_SR` in 3.19.2 (not `nk.driver`).
- local→cliff: all 189 children per parent evaluated exactly by FWHT (no K=16 LB screen; LB is
  blind to 1-qubit gates). Alternating refinement (§4.2.4) not done. W2 re-opt not computed there.
- Search mode inner solves run without α-continuation; final mode uses 0.25→0.5→1(→2).
- Dedupe fingerprint adds rounded J in mode "Z" (fixed-basis J is not local-invariant).
- Bug fixed: `axis_to_unitary` used `arccos(n_x)`, inaccurate near n=−x; now `arctan2(|x×n|, n_x)`.
- RBM trained on exact (unsnapped, unthresholded) H_U; a separate cell reports snapped-axes W and
  term counts.

## Results so far (basis optimization, α=1)
| model | basis | W^Z_1 | word | notes |
|---|---|---|---|---|
| kitaev | naive | 10.161 | – | LB = 9.0 (all Bloch vectors 0) |
| kitaev | local | 10.039 | – | snap err 0.26 |
| kitaev | cliff→local | **8.181** | 9 G-gates on 9 disjoint bonds (dimer cover) | axes snap to ±Pauli (err ~3e-8) → U is effectively Clifford |
| kitaev | local→cliff | 9.589 | 20 gates (H,S,CZ) | |
| gapped_3 | naive | 9.820 | – | LB 8.98 |
| gapped_3 | local | 9.009 | – | |
| gapped_3 | cliff→local | **6.013** | 9 YCZ gates, disjoint pairs | snap err 0.18 |
| gapped_3 | local→cliff | 9.009 | 0 gates (no improvement found) | |

Term counts of H_U (merged / >1e-4 / max weight): kitaev naive 27/27/2, local 243/233/2,
cliff→local 87/27/4 (60 residual terms <1e-6 from the 3e-8 snapping error; snapped exactly → 27),
local→cliff 243/233/7; gapped_3 naive 297/297/2, local 297/297/2 (already contains all bond
Pauli pairs + fields, so no growth after merging), cliff→local 2691/969/4, local→cliff 297/297/2.
Tentative answer to "is there an approach without term growth": yes — Clifford-only U (here:
pure-Kitaev cliff→local, whose optimal V is a local Clifford), and any local V on a Hamiltonian
that already contains all Pauli pairs per bond (gapped_3). Verify against the final notebook.

## Pitfalls for the next agent
- Do not run anything with >30 CPUs / >40 GB. Beam search uses 28 spawn workers.
- Don't overwrite the `.ipynb` blindly; follow the `jupytext-sync` skill (sync, then execute).
- Single seed: treat RBM differences as suggestive; report the last-500-iteration mean ± std.
