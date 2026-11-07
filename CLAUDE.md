# General instructions for LLM Assistants

- **Shared machine**: This is a shared machine. Even if more resources are
  available, do not use more than 30 CPUs or 40 GB of memory for any job.

- **Verbosity**: Brief by default. Expand when asked, or when topic is complex.
- **Multiple valid approaches**: More than one reasonable way: present options/tradeoffs let user choose.
- **Scope discipline**: Do only what is asked. If you notice an unrelated
  issue, flag it briefly at the end rather than acting on it — unless it's a
  safety or correctness bug, in which case flag it immediately even mid-task.
- **Analysis reporting**: When investigating (debugging, reading a paper,
  etc.), give the conclusion plus the key evidence (e.g. file:line, the
  specific formula or assumption) rather than narrating the full search.
- **Clarifying questions**: Batch them and ask upfront before starting a task,
  rather than interrupting partway through.
- **Formatting**: Default to structured replies (bullets, bold key terms) even
  for short answers.
- **Confidence and correctness**: Never state something with more confidence
  than you have — flag uncertain claims explicitly, e.g. append "(not sure)".
  Before implementing something based on an assumption the user stated, check
  whether that assumption is actually correct; do not silently agree with and
  build on a wrong premise. This applies to any factual claim, not just ones
  that affect code.
## Clusters

GPU jobs → Leonardo (`leonardo` skill).

- No git checkout on Leonardo: code is staged by `rsync` to
  `/leonardo_work/ICT26_CMSP_0/nniggema/vitq/` and results are fetched back with `rsync`.
- Venv: `~/Projects/SuperfluidMottTriangular/.venv` on Leonardo (netket 3.22.4, jax 0.10.2 + CUDA).
- sbatch scripts in `check_time_evolution/vit/` (template `leonardo_ptvmc_scan.sbatch`),
  account `ICT25_CMSP_0`.

## Figures (on top of the global `python-plotting` skill)

- Single-column width by default; stack panels or use at most 3 narrow columns. Long
  descriptions (Hamiltonians etc.) go in the caption.
- Tick labels 8 pt (`plt.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize": 8})`).
- Heatmap colorbars horizontal on top, one per column when panels share the scale:
  `fig.colorbar(mesh, ax=axs[:, col], orientation="horizontal", location="top", pad=0.03, aspect=14, label=...)`.
- Heatmap ticks: white where the colormap edge is dark, black where light
  (`ax.tick_params(which="both", color=...)`); tick labels stay black.
- Text on heatmaps: white with a soft dark halo, stacked faint strokes:
  `[patheffects.Stroke(linewidth=w, foreground="black", alpha=0.25) for w in (3.2, 2.4, 1.6)] + [patheffects.Normal()]`.
