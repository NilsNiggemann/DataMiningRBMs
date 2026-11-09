# Next: finish step 002-ptvmc-heatmap-vs-mumax (no summary/report yet; step 001 was skipped on purpose)
Pending jobs (launched 2026-11-09 08:25 UTC): 15 local runs x 2 threads (CPUs 32-61, ~8 GB), ViT V2, lr 0.5, diag_shift 1e-6,
 max_iter 1500 (g_f<1.2) / 2000 (<1.8, incl. control 1.54) / 5000 (>=1.8), dt .025, t<=1, timeout 23 h. Per-iteration loss saved (loss_hist).
 Out: data/vit_tdvp_long/g<gf>/*.npz; logs check_time_evolution/vit/logs/long_g*.log (one line per time step).
 Check: `ps -eo args | grep -c "[.]venv/bin/python vit_ptvmc_losshist"` (30 = still running, 2 procs per run); `grep -c t= <log>` = steps done (of 40).
 Progress 11:08 UTC: g_f=0 at 9/40 steps, g_f=3 at 18/40 -> ETA low g_f ~21:00-23:00 UTC 2026-11-09 (late steps slower; rough), high g_f ~6 h after start.
Also still running? run H (data/vit_tdvp_scan2, g0.69 ds 1e-6) finished or hit its 14 h cap; check data completeness.
Tasks:
1. When done: heatmap with the long-run data (extend quench2d/tfi1d_heatmaps_ptvmc.py: panel c currently = 100-it scan with rows g0.69/g3.0 replaced by scan2 runs B/H) and
   loss-vs-iteration convergence plots for all g_f (vit/losshist_figs.py is a template, g3 only so far).
2. Compare I vs mu_max (levels where I crosses 1e-4/1e-2; RBM vs ViT), is the training converged (loss at it N vs end)?
3. summary.md, knowledge.md, INDEX, report + figs, publish (page not yet created: bootstrap.md), commit.
Done when: long-run heatmap + convergence analysis written; answer whether ViT p-tVMC reaches I(t=1) <~ 1e-2 everywhere.
