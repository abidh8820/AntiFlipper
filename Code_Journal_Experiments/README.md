# AntiFLipper journal experiment pipeline

This folder is isolated from `Code/`; it does not modify the submitted implementation or old results. It implements the reduced plan in `Note/updates2.md` as an unattended, resumable workflow.

For a concise step-by-step operating guide, see `RUN_INSTRUCTIONS.md`.

## One-command use

Activate the Python/CUDA environment used by the original notebooks, then double-click or run:

```powershell
.\run_all.cmd
```

That runs the **core** 258-job profile. To include the optional partial-flip and ablation experiments:

```powershell
.\run_all.cmd full
```

To run independent experiment jobs concurrently, pass a worker count as the
second argument. Start conservatively because every worker has its own model,
dataset objects, CPU memory, and CUDA context:

```powershell
.\run_all.cmd core 2
```

The parent process assigns every job exactly once and is solely responsible for
the shared checklist, analysis, and ZIP package. Each worker writes only to its
assigned job directory. Do not start a second launcher manually. On a 4 GB GTX
1050 Ti, start with two workers for MNIST and return to one if CUDA runs out of
memory. Parallel execution introduces resource contention, so use the separate
fixed-workload benchmark or a serial run for publication-quality timing.

To validate the pipeline cheaply before committing GPU time:

```powershell
.\run_all.cmd smoke
```

If Python is not discoverable, point the launcher at the correct interpreter:

```powershell
$env:ANTIFLIPPER_PYTHON = "C:\path\to\environment\python.exe"
.\run_all.cmd
```

The launcher checks dependencies but does not silently install or change CUDA packages. Install missing packages with `python -m pip install -r requirements.txt`, after installing the PyTorch build appropriate for the machine. The core/full launcher refuses CPU-only PyTorch so a long schedule is not accidentally launched in an environment that cannot meet the requested timeline; the smoke profile works on CPU.

FLAME prefers the external `hdbscan` package used by the original implementation and falls back to `sklearn.cluster.HDBSCAN` when the external package is absent. The captured environment identifies which dependencies were installed for the run.

## Interruption and resumption

Each job saves a checkpoint after every completed communication round. You may stop with Ctrl+C, reboot, or close the terminal. Run the same command again; completed jobs are skipped and the interrupted job resumes at its next unfinished round.

Checkpoint contents include the model, trust values, counters, excluded clients, detection rounds, per-round rows, baseline memory, and Python/NumPy/PyTorch RNG states. A configuration hash is part of every job name, preventing an altered experiment from accidentally loading an incompatible checkpoint.

`CHECKLIST.md` and `checklist.csv` are regenerated after each job. The machine-readable `state.json` inside each job is authoritative.

## Outputs

```text
Code_Journal_Experiments/
├── manifests/<profile>.json       exact immutable job list
├── results/<profile>/
│   ├── environment/               Python, platform, CUDA, GPU, pip freeze
│   └── <job-id>/
│       ├── config.json             full run configuration
│       ├── checkpoint.pt           local resume state (not put in ZIP)
│       ├── round_metrics.csv       every round and all analysis metrics
│       ├── partition_metadata.json partition sizes/hashes and attacker IDs
│       ├── latest_detection.json   attackers, trust, counters, detections
│       ├── summary.json            final and last-window metrics
│       ├── console.log             stdout/stderr
│       └── error.txt               traceback, only if failed
├── analysis/<profile>/
│   ├── per_run_metrics.csv
│   ├── aggregate_mean_std_ci95.csv
│   ├── REPORT.md
│   └── figures/
└── packages/
    └── AntiFlipper_Q1_<profile>_<COMPLETE|PARTIAL>_<YYYYMMDD_HHMMSS>_<manifest-hash>.zip
```

The ZIP includes the manifest, checklist, source code, environment metadata, raw round metrics, seed summaries, console/error logs, aggregate tables, report, and figures. Large model checkpoints are excluded because they are required for local resumption but not statistical analysis. A partial ZIP is still created after an interruption or run with failures.

## Useful commands

```powershell
python cli.py plan --profile core
python cli.py status --profile core
python cli.py run --profile core
python cli.py run --profile core --workers 2
python cli.py analyze --profile core
python cli.py benchmark --profile core
python cli.py package --profile core
```

For a live progress view in a second PowerShell window, run:

```powershell
.\monitor.cmd core
```

The monitor shows the current job, completed rounds, current-job ETA, rough total ETA, job counts, stored results/checkpoints/archives, and free disk space. It refreshes every 10 seconds. Press `Ctrl+C` to close the monitor; this does not stop the experiment.

The fixed-workload benchmark is separate from observed end-to-end round timing. Run it when convenient; it creates matched-update-count timing files under `analysis/<profile>/timing/`.

## Methodology encoded by the pipeline

- MNIST: 100 clients, 200 rounds, CNN, batch size 64.
- CIFAR-10: 20 clients, 100 rounds, ResNet18, batch size 32.
- Full client participation and three seeds by default.
- Attack strength is separated into malicious-client fraction, attack-round probability, and poisoned-example fraction.
- AntiFLipper uses population-relative trust `rho_i = N_eligible * tau_i` by default; the legacy absolute threshold remains an ablation.
- Trust coefficient, threshold, counter threshold/mode, grace period, and evaluation fraction are explicit saved parameters.
- A balanced root subset is removed before every method's matched partition so all methods train on identical examples; only FLTrust consumes that server root set. Its size and hash are recorded.
- Multi-Krum configurations are validated against `n > 2f + 2`; Bulyan is intentionally absent.
- Every clean heterogeneity and sensitivity control reports detection false positives, not merely global accuracy.

## Important interpretation limits

The pipeline faithfully operationalizes the scoped plan, but generated results still require scientific judgment before publication. FLTrust has a server-root-data assumption that AntiFLipper does not. Non-AntiFLipper methods do not explicitly output client classifications, so AntiFLipper detection precision/recall must not be presented as if directly comparable to their aggregation weights. The automated report preserves these caveats.
