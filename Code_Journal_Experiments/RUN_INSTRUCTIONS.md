# AntiFLipper experiment run instructions

This is the short operational guide for running the automated journal experiments. The pipeline handles the experiment list, checkpoints, resumption, analysis, checklist updates, and ZIP packaging.

## 1. Open PowerShell in the experiment folder

```powershell
cd D:\Code\Papers\AntiFlipper\Code_Journal_Experiments
```

## 2. Select the CUDA-enabled Python environment

Use the Python environment that has the CUDA-enabled PyTorch installation used for the original experiments.

If that environment is already activated, no additional setting is needed. Otherwise, identify its `python.exe` and set:

```powershell
$env:ANTIFLIPPER_PYTHON = "C:\path\to\gpu-environment\python.exe"
```

You can verify the selected environment with:

```powershell
& $env:ANTIFLIPPER_PYTHON -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

The output must show `CUDA: True` before running the core or full schedule. The launcher intentionally refuses to run those profiles with CPU-only PyTorch.

If dependencies are missing, install them into the selected environment:

```powershell
& $env:ANTIFLIPPER_PYTHON -m pip install -r requirements.txt
```

Install the correct CUDA-enabled PyTorch build separately if necessary; do not replace a working GPU build with a CPU-only build.

## 3. Run the smoke test first

```powershell
.\run_all.cmd smoke
```

The smoke profile runs a small offline synthetic experiment. It verifies training, checkpointing, analysis, checklist generation, and ZIP packaging. It is only a pipeline test and must not be used as paper evidence.

## 4. Run the core experiment profile

```powershell
.\run_all.cmd
```

This runs the default `core` profile containing 258 independently checkpointed jobs:

- A1 headline replication and FLTrust comparison;
- A2 clean heterogeneity/false-positive tests;
- A3 malicious-client-ratio sweep;
- A5-lite hyperparameter sensitivity with matched controls.

No manual intervention is needed between jobs.

## 5. Optionally run the full profile

After the core profile has finished, run:

```powershell
.\run_all.cmd full
```

The full profile contains the core jobs plus the optional A4 partial-flip and A6 ablation experiments, for 330 jobs total. Jobs already completed under a different profile have separate result directories; do not delete core results.

## Interruption and automatic resumption

You may stop the process with `Ctrl+C`, close the terminal, or reboot the computer. To resume, activate/select the same Python environment and run the same command again:

```powershell
.\run_all.cmd
```

or:

```powershell
.\run_all.cmd full
```

Completed jobs are skipped. An interrupted job resumes after its last completed communication round using `checkpoint.pt`.

Do not delete or manually edit the following while a profile is unfinished:

- `manifests/`
- `results/`
- job `checkpoint.pt` files
- job `config.json` files

## Check progress

For the core profile:

```powershell
python cli.py status --profile core
```

For the full profile:

```powershell
python cli.py status --profile full
```

If `python` is not the selected GPU interpreter, use:

```powershell
& $env:ANTIFLIPPER_PYTHON cli.py status --profile core
```

You can also open:

- `CHECKLIST.md` for the live human-readable checklist;
- `checklist.csv` for its machine-readable version;
- `results/<profile>/<job-id>/console.log` for a specific job's output;
- `results/<profile>/<job-id>/error.txt` if a job failed.

Failed jobs are retried automatically on the next normal run unless `--no-retry-failed` is explicitly used.

## Regenerate analysis manually

Analysis is generated automatically after execution. To regenerate it without rerunning experiments:

```powershell
& $env:ANTIFLIPPER_PYTHON cli.py analyze --profile core
```

Results are written under:

```text
analysis/core/
```

Important files include:

- `REPORT.md`: automated interpretation and caveats;
- `per_run_metrics.csv`: one row per seed/configuration;
- `aggregate_mean_std_ci95.csv`: mean, standard deviation, and 95% CI;
- `figures/`: generated plots.

## Run the matched-workload timing benchmark manually

The launcher runs it automatically once for core/full. To rerun it manually:

```powershell
& $env:ANTIFLIPPER_PYTHON cli.py benchmark --profile core
```

The timing output is written to:

```text
analysis/core/timing/
```

## Create another ZIP package manually

Packaging is automatic, including after an interrupted/partial run. To package the current results again:

```powershell
& $env:ANTIFLIPPER_PYTHON cli.py package --profile core
```

Archives are saved in `packages/` using this naming format:

```text
AntiFlipper_Q1_<profile>_<COMPLETE|PARTIAL>_<YYYYMMDD_HHMMSS>_<manifest-hash>.zip
```

Examples:

```text
AntiFlipper_Q1_core_COMPLETE_20260830_231500_a1b2c3d4e5.zip
AntiFlipper_Q1_full_PARTIAL_20260825_103000_f6e7d8c9b0.zip
```

`LATEST_core.txt` or `LATEST_full.txt` identifies the newest archive for that profile.

The ZIP contains configurations, environment information, raw per-round metrics, summaries, logs, checklists, analysis tables, plots, and source code. Model checkpoint files are intentionally excluded from the ZIP because they are large and are needed only for local resumption.

## Recommended command sequence

```powershell
cd D:\Code\Papers\AntiFlipper\Code_Journal_Experiments
$env:ANTIFLIPPER_PYTHON = "C:\path\to\gpu-environment\python.exe"
.\run_all.cmd smoke
.\run_all.cmd
```

After the core profile finishes, review:

```text
analysis/core/REPORT.md
packages/LATEST_core.txt
```

Run the optional full profile only if the remaining timeline permits:

```powershell
.\run_all.cmd full
```

## Troubleshooting

### The launcher says CUDA is unavailable

The selected interpreter has CPU-only PyTorch or is the wrong environment. Set `ANTIFLIPPER_PYTHON` to the GPU environment's `python.exe` and verify `torch.cuda.is_available()` returns `True`.

### A dependency import fails

Install `requirements.txt` into the selected environment:

```powershell
& $env:ANTIFLIPPER_PYTHON -m pip install -r requirements.txt
```

FLAME can use either the external `hdbscan` package or `sklearn.cluster.HDBSCAN`.

### A job failed

Open its `error.txt` and `console.log`, correct the environment problem if necessary, and run the same profile again. The failed job will be retried; completed jobs remain skipped.

### The computer stopped during training

Run the same profile again. The last fully completed communication round is preserved. At worst, only the interrupted round is repeated.

### Disk space is becoming low

Do not delete active checkpoints. Old timestamped ZIP archives can be moved elsewhere, retaining the newest archive and all active `results/<profile>/` directories until the experiments finish.
