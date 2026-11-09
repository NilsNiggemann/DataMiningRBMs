2026-11-07 002 start; plan approved (ViT 100 it)
launched 28 procs Sat Nov  7 15:01:09 UTC 2026 cpus 0-27 (taskset), out data/vit_tdvp_scan/{rbm,vit}, logs vit/logs/scan_*
2026-11-07 15:17 mem 14.6 GB total (28 procs). RBM g3 finished in ~15 min (I(1)=8.4e-4). ViT: high-g ~30 s/step.. low-g ~7 min/step (g<=1: 2-3 steps in 15 min => ~5 h+ total).
wrote check_time_evolution/quench2d/tfi1d_heatmaps_ptvmc.py (reads data/vit_tdvp_scan/{rbm,vit}, partial runs ok; tested runs, figs/tfi1d_heatmaps_ptvmc.* and tfi1d_scatter_ptvmc_mumax.*). Not yet viewed/final.
PENDING: 28 procs, check: ps -eo args | grep -c vit_ptvmc_quench; logs vit/logs/scan_*.log
2026-11-07 19:57 all 28 runs finished (last 16:55 UTC); plotted
2026-11-07 20:02 UTC launched 10 procs scan2 (ViT, longer/other SR settings at g3.0 and g0.69), out data/vit_tdvp_scan2/<tag>, logs vit/logs/scan2_*.log, timeout 3h (=23:00 UTC = 00:00 CET); partial npz saved each step
2026-11-07 20:07 UTC CORRECTION: user meant noon tomorrow. Killed the 3h scan2 runs, relaunched 10 procs: g3 1000/2000 it, g0.69 500 it, timeout 14h (ends 10:07 UTC Nov 8 < noon CET=11:00 UTC). dirs data/vit_tdvp_scan2/<tag>, logs vit/logs/scan2_*
2026-11-08 ~07:00 UTC user: 'is it converged in training' -> per-iteration loss not saved; vit_ptvmc_losshist.py (driver copy + loss_hist) running g3.0 tmax 0.6 600 it ds 1e-4/1e-6, out data/vit_tdvp_losshist/<tag>, logs vit/logs/losshist_*
2026-11-09 08:25 UTC user: 'very accurate quenches'. scaling test (g0.69,60it): 1thr 0.9 s/it, 2thr 0.5, 4/8/16 ~0.6 -> 2 threads. Launched 15 runs x2 threads (CPUs 32-61), ViT V2, lr .5 ds 1e-6, max_iter 1500 (g<1.2)/2000 (<1.8)/5000, timeout 23h, out data/vit_tdvp_long/g<gf>, logs vit/logs/long_g*.log, loss_hist saved (vit_ptvmc_losshist.py)
