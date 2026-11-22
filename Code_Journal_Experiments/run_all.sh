#!/usr/bin/env bash
# Linux launcher. Usage: ./run_all.sh [profile] [workers]
#   ./run_all.sh cr 2        # class-residual experiments (84 jobs, pilot first)
#   ./run_all.sh smoke       # 1-minute pipeline check
# Ctrl+C is safe; run the same command again to resume. Completed jobs are skipped.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
profile=${1:-cr}
workers=${2:-2}
python=${ANTIFLIPPER_PYTHON:-python3}

"$python" -c "import torch, torchvision, numpy, sklearn, scipy, matplotlib; print('Dependencies OK; CUDA:', torch.cuda.is_available())"
if [[ $profile != smoke ]]; then
    "$python" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
        || { echo "CUDA is not available in $python; set ANTIFLIPPER_PYTHON to the GPU environment."; exit 1; }
fi
"$python" self_test.py
"$python" cli.py plan --profile "$profile"
exec "$python" cli.py run --profile "$profile" --workers "$workers"
