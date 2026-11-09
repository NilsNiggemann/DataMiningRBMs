# Plan: iterate workstream `vit-quench` (bootstrap, then step 001)

## Context
`/iterate:iterate` found no `iter/` dir, so this run bootstraps workstream `vit-quench` (repo root `iter/vit-quench/`).
Problem (plans/vit_quench_handoff.md): small ViT NQS (L=10, V2, 1716 real params) fails 1D TFI quenches under TDVP/p-tVMC
(I~0.5-1 at g_f=0.5, NaN at g_f=1,2,3); RBM reference reaches 7e-3 / 7e-4 at t=1. Goal: "decent" quench results.

## Bootstrap (no step yet; stop after showing files, per skill)
- `GOAL.md` (<=30 lines): goal; success = I(t=1) <~ 1e-2 at g_f=0.5 and 3.0, L=10, no NaN, control g_f=1.5 flat; constraints = local <=30 CPUs/40 GB, no Adam (SR lr + diag_shift only), Leonardo ONLY `--qos=boost_qos_dbg`, <=30 min/job, each sbatch needs user OK; pointers to plans/vit_quench_handoff.md, plans/plan_vit_quench.md, vit/*.py, vit/leonardo_ptvmc_scan.sbatch.
- `notes/knowledge.md` seeded (tag 000) from handoff sec. 6 (integrator check, fits, real-body failure, V2 results, rtol_smooth scan).
- empty `notes/INDEX.md`; `plans/next.md` (step 001); `plans/backlog.md` (handoff sec. 7 ideas: wider head nhid>d, per-patch readout, full QKV, p-tVMC/implicit integrator, MPS baseline, L=16).
- `reports/index.html` from `assets/index.html` ({{NAME}} = "ViT quench"), `updates.json` = `[]`, `open.md` + `.open.base.md`.
- Note: the pasted claude.ai artifact is a Claude Docs doc (not the iterate viewer); the iterate page is published as a separate artifact on step 001, I will not write into that doc unless you say so.

## Proposed step 001: `001-pipeline-check-and-ptvmc-gpu`
1. Pipeline check (handoff Q1): RBM via same driver (dt=0.005, rtol_smooth 1e-4) at g_f=0.5 locally -> is ViT failure genuine?
2. Read existing p-tVMC logs (vit/ptvmc_V2_g*.log, figs_ptvmc_infidelity_mu.png) to see how far p-tVMC already gets.
3. Leonardo smoke test (<=30 min, `--qos=boost_qos_dbg`, 4 GPUs, one config each; TMAX<=1) of `leonardo_ptvmc_scan.sbatch`: vary max_iter/diag_shift at g_f=0.5, 3.0. Ask OK before sbatch.
4. Compare I(t) vs RBM; pick the best config.
Done when: RBM-through-ViT-driver number known and one p-tVMC config reaches I(t=1)<~1e-2 at g_f=0.5 or the gap is quantified. Figures: I(t) ViT vs RBM per g_f from saved npz.

## Verification
Check `iter/vit-quench/` file set; step 001 later: npz outputs exist, report figs from saved data, page published, commit `iter(vit-quench) 001: ...`.
