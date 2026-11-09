# Goal: decent ViT results for 1D TFI quenches

Small Vision-Transformer NQS (L=10, rung V2, ~1.7k real params) should follow quenches of the 1D transverse-field
Ising chain (open BC, J=1, g_i=1.5 -> g_f in [0,3], t<=1) with low infidelity, like the RBM reference.

## Success criteria
- L=10: I(t=1) <~ 1e-2 at g_f=0.5 and g_f=3.0 (RBM: 7e-3 and 7e-4), no NaN, control g_f=1.5 stays flat.
- Then, if cheap: L=12-16 (exact sums) and the other g_f.

## Constraints
- Local: <=30 CPUs, <=40 GB (shared machine). Python: check_time_evolution/.venv/bin/python.
- Leonardo only with `--qos=boost_qos_dbg`, <=30 min per job; ask the user before every sbatch.
- NQS fits/optimisation: never Adam; only vary SR lr, diag_shift, iterations.
- Brief, structured replies; flag uncertain claims.

## Background
- check_time_evolution/plans/vit_quench_handoff.md (status 2026-11-03, results + open questions)
- check_time_evolution/plans/plan_vit_quench.md (older S0-S16 checklist)
- Code: check_time_evolution/vit/{vit1d.py,vit_tdvp_quench.py,vit_ptvmc_quench.py,leonardo_ptvmc_scan.sbatch}
- RBM reference: check_time_evolution/rbm_tdvp_quench.py
