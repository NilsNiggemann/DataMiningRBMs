2026-11-07 start; plan approved
- RBM via same p-tVMC driver (dt .025, lr .5, ds 1e-4, 30 it): I(t=1)=6.1e-3 (g0.5), 7.3e-4 (g3.0) [vit/ptvmc_rbm_g*.log] -> pipeline OK, matches TDVP RK45 (7.3e-3/7e-4); task 1 already done by earlier run, no rerun
- ViT V2 p-tVMC same settings: I(t=1)=0.113 (g0.5), 0.079 (g3.0); per-step opt infidelity floor ~1e-3..2e-3 late (vs RBM 6e-5/1e-5) -> optimisation-limited?
- BLOCKED task 3: 'ssh leonardo' fails, host key of login.leonardo.cineca.it changed (known_hosts:23, RSA SHA256:Cd8/hP3K5W9ZwZghM0w19GCo0VGY1z7k4/QPpCD+1oI). Not bypassed; waiting for user to verify. No job submitted.
- staged code+init fit to /leonardo_work/ICT26_CMSP_0/nniggema/vitq on Leonardo (venv: ~/Projects/SuperfluidMottTriangular/.venv, netket 3.22.4/jax 0.10.2 vs local 3.19.2/0.5.3)
- copied leonardo skill from SuperfluidMottTriangular to .claude/skills/leonardo (adapted header note); login-node CPU smoke of vit_ptvmc_quench.py (2 steps) ran OK on netket 3.22.4 (only deprecation warnings)
- SUBMITTED job 63775437 (boost_qos_dbg, 30 min, 4 GPUs, configs per leonardo_ptvmc_scan.sbatch; OUT=/data/vit_tdvp_gpu/run0..3, logs in vit/logs/). Check: ssh leonardo 'sacct -j 63775437 -X -n -o State%20,Elapsed'; rsync results back
- job 63775437 FAILED at start (14:04 CET): JAX_PLATFORMS=gpu invalid in jax 0.10 ('rocm' not known) -> changed to cuda in sbatch, restaged. No data produced. Awaiting user OK to resubmit.
- RESUBMITTED job 63777480 at 13:34:20 UTC (same configs, cuda fix); fetch scheduled 33 min later (~14:07 UTC) into data/vit_tdvp_gpu/
- job 63777480 TIMEOUT at 33:19 (30 min limit); fetched data/vit_tdvp_gpu/run0..3 + check_time_evolution/vit/logs/ at 14:07 UTC. (stray top-level data/vit_tdvp_gpu/L10_V2_ptvmc_*gf0.5000.npz = login-node smoke test, ignore)
- results p-tVMC V2 dt .025, lr .5 (baseline 30 it, ds 1e-4: I(1)=0.113 g0.5, 0.079 g3.0):
  run0 g0.5 100it ds1e-4: I(1)=3.6e-2 (finished, ~20 min) | run1 g0.5 300it ds1e-4: reached t=0.8 only, I=1.0e-2 (baseline at 0.8: 2.8e-2) |
  run2 g0.5 300it ds1e-6: I(1)=6.7e-2 | run3 g3.0 300it ds1e-4: I(1)=0.104 (worse than baseline), opt floor 2.9e-3
- => more inner iterations help at g0.5 (3x better at 100 it) but not g3.0; still above 1e-2 target; RBM floor opt~6e-5 vs ViT 5e-5..3e-3. Not yet made figure/summary/report/commit.
- CPU single-thread timing (local .venv, 1 thread, taskset, load avg 67/192 cores): 4 steps g0.5 30it: 271 s wall; g3.0 100it: 236 s wall (incl. ~30-60 s startup, not separated) -> ~1.8 s/it (g0.5) vs ~0.45 s/it (g3.0); per-it cost varies 4x (SR cg solver? not verified). plans/next.md rewritten: 15 g_f x (dt .025, 300 it) on CPU.
