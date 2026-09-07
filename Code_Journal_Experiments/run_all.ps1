param(
    [ValidateSet("smoke", "core", "full")]
    [string]$Profile = "core",
    [string]$Python = $env:ANTIFLIPPER_PYTHON,
    [ValidateRange(1, 64)]
    [int]$Workers = 1
)

$ErrorActionPreference = "Stop"
$PipelineRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $PipelineRoot

if (-not $Python) {
    $WorkspacePython = Join-Path $PipelineRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $WorkspacePython) {
        $Python = $WorkspacePython
    } else {
        $Python = "python"
    }
}

try {
    & $Python -c "import sys; print(sys.executable)"
} catch {
    throw "A working Python environment was not found. Activate the same environment used by your notebooks, or set ANTIFLIPPER_PYTHON to its python.exe path."
}
if ($LASTEXITCODE -ne 0) {
    throw "Python failed to start. Activate your ML environment or set ANTIFLIPPER_PYTHON."
}

& $Python -c "import torch, torchvision, numpy, sklearn, scipy, matplotlib; from sklearn.cluster import HDBSCAN; print('Dependencies OK; CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    throw "Dependencies are missing. Run: python -m pip install -r requirements.txt (install the correct CUDA PyTorch build first)."
}

if ($Profile -ne "smoke") {
    & $Python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "The core/full schedule requires a CUDA-enabled PyTorch environment for the requested timeline. Activate your GPU environment or set ANTIFLIPPER_PYTHON to its python.exe path. Use run_all.cmd smoke to test on CPU."
    }
}

& $Python cli.py plan --profile $Profile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$TimingFile = Join-Path $PipelineRoot "analysis\$Profile\timing\fixed_workload_timing.csv"
if (($Profile -ne "smoke") -and (-not (Test-Path -LiteralPath $TimingFile))) {
    & $Python cli.py benchmark --profile $Profile
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "The fixed-workload benchmark failed; experiment execution will continue and the error can be investigated separately."
    }
}
& $Python cli.py run --profile $Profile --workers $Workers
exit $LASTEXITCODE
