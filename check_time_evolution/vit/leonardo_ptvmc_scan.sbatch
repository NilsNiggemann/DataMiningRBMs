#!/bin/bash
# Is the ViT p-tVMC error an optimisation problem? One node, 4 A100, one independent run per GPU.
# Same quench, different inner-optimiser settings (see CONFIGS). Results: $OUT/run<i>/L10_V2_ptvmc_dt0.025_gf*.npz
# (saved after every step, so a killed job keeps its partial curve).
#
# Submit from check_time_evolution/vit AFTER the user agrees:
#   smoke test (<=30 min): TMAX=0.1 sbatch --qos=boost_qos_dbg --time=00:30:00 leonardo_ptvmc_scan.sbatch
#   production:            sbatch leonardo_ptvmc_scan.sbatch
# Needs a venv with netket + jax[cuda12] (default: repo-root .venv, as in slurm_runners/); override with VENV=...
#SBATCH --job-name=vit_ptvmc_scan
#SBATCH --partition=boost_usr_prod
#SBATCH --account=ICT25_CMSP_0
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/vit_ptvmc_scan-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"; mkdir -p logs

module purge 2>/dev/null || true
module load cuda/12.3 python
unset LD_LIBRARY_PATH                      # pip CUDA wheels bundle their own libs; modules must not shadow them
source "${VENV:-${SLURM_SUBMIT_DIR}/../../.venv}/bin/activate"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda                  # the driver defaults to cpu via setdefault
export XLA_PYTHON_CLIENT_PREALLOCATE=false # so nvidia-smi memory.used means something

python - <<'E'
import jax; assert jax.default_backend() in ("gpu", "cuda") and jax.device_count() == 4, jax.devices(); print(jax.devices())
E

OUT="${OUT:-../../data/vit_tdvp_gpu}"; TMAX="${TMAX:-1.0}"; RUNG=V2; SEED=1234
INIT="init_L10_${RUNG}_s${SEED}.msgpack"; mkdir -p "$OUT"

# CONFIGS (one per GPU): "g_f max_iter lr diag_shift". Reference (CPU run): max_iter 30, lr 0.5, diag_shift 1e-4.
# tol is 1e-9, so converged steps stop early and a larger max_iter costs little where it is not needed.
CONFIGS=("0.5 100  0.5 1e-4"
         "0.5 300  0.5 1e-4"
         "0.5 300  0.5 1e-6"   # smaller initial damping (backtracking raises it again if the loss rises)
         "3.0 300  0.5 1e-4")

# t=0 fit of the ViT (SR infidelity fit, lr 0.05, 3000 it; no Adam). Done once if the cache is missing.
if [ ! -f "$OUT/$INIT" ]; then
  CUDA_VISIBLE_DEVICES=0 python vit_tdvp_quench.py --rung $RUNG --L 10 --fit_only --lr 0.05 --n_iter 3000 \
      --seed $SEED --out "$OUT" 2>&1 | tee "logs/fit_${SLURM_JOB_ID}.log"
fi
# GPU power / memory log (power draw says more than utilisation.gpu; A100 limit 457 W)
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits -l 5 \
    > "logs/gpu_${SLURM_JOB_ID}.csv" &
SMI=$!

for i in 0 1 2 3; do
  read -r GF IT LR DS <<< "${CONFIGS[$i]}"
  RUN="$OUT/run$i"; mkdir -p "$RUN"; cp "$OUT/$INIT" "$RUN/"
  ( CUDA_VISIBLE_DEVICES=$i python vit_ptvmc_quench.py --rung $RUNG --gf "$GF" --tmax "$TMAX" --max_iter "$IT" \
        --lr "$LR" --diag_shift "$DS" --seed $SEED --out "$RUN" 2>&1 \
      | while IFS= read -r l; do echo "$(date +%T) $l"; done > "logs/run${i}_${SLURM_JOB_ID}.log" ) &   # timestamps give s/step
done
wait
kill $SMI 2>/dev/null || true
echo "done"
