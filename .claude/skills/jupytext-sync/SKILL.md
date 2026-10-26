---
name: jupytext-sync
description: >
  Keep paired notebooks in sync (either direction, newer file wins) AND re-executed after your edits. Whenever you edit or re-test/re-plot a notebook's .py (py:percent) or
  .ipynb file in this repo, run `uv run jupytext --sync <file>` and then execute the .ipynb in place
  (nbconvert --execute --inplace), so code and outputs are current. Use on any edit to a notebook under Notebooks/ or any file with
  `formats: ipynb,py:percent` in its header.
---

# jupytext-sync

Notebooks here are paired (`formats: ipynb,py:percent` in the header). Edit the `.py`
(easier to diff), never leave the `.ipynb` stale.

Check which side is newer before touching either (`ls -l --time-style=full-iso Notebooks/NN_notebooks/<name>.{py,ipynb}`).
The user may have edited the `.ipynb` in Jupyter; never overwrite it blindly from a stale `.py`.

**1. Sync (works in both directions).** `jupytext --sync` copies the code of the *newer* file
into the older one: `.py` newer -> updates the `.ipynb` (keeping its outputs); `.ipynb` newer ->
updates the `.py` from it. Run it first, always, and re-read the `.py` before editing if the
`.ipynb` was the newer one (so you edit the user's latest version):

```bash
uv run jupytext --sync Notebooks/NN_notebooks/<name>.py
```

**2. After your own edits: sync again, then re-execute** (the sync only copies code; the `.ipynb`
keeps its old figures until executed, which counts as stale):

```bash
uv run jupytext --sync Notebooks/NN_notebooks/<name>.py
uv run jupyter nbconvert --to notebook --execute --inplace Notebooks/NN_notebooks/<name>.ipynb
```

- Also do this when you only "retest"/re-plot/fix a plot: the user looks at the `.ipynb`, so its stored figures must show the new result.
- If only the user edited the `.ipynb` (step 1 pulled it into the `.py`), no re-execution is needed unless asked; their outputs are already current.
- Executing bumps the `.ipynb` mtime, so it ends up newer than the `.py` with identical code; that is fine, the next sync is then a no-op.
- If both files changed since the last sync, `--sync` refuses: reconcile by hand (diff the `.py` against `jupytext --to py:percent -o - <ipynb>`).
- `*.ipynb` is gitignored, so `git status` won't show it; check mtimes instead.
- If execution fails, fix the code and rerun; report it if it can't be fixed.
