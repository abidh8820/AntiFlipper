#!/usr/bin/env bash
# Run several profiles back to back (used by the antiflipper-cr systemd user service).
#   ./run_queue.sh                      # cr, then fltrust_fix, 3 workers
#   WORKERS=2 ./run_queue.sh cr         # one profile, 2 workers
# Completed jobs are skipped, so re-running (or a systemd restart) resumes where it stopped.
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
python=${ANTIFLIPPER_PYTHON:-python3}
workers=${WORKERS:-3}
profiles=("$@")
[[ ${#profiles[@]} -eq 0 ]] && profiles=(cr fltrust_fix)

"$python" self_test.py || exit 1
status=0
for profile in "${profiles[@]}"; do
    echo "=== $(date '+%F %T') profile $profile, $workers workers"
    "$python" -u cli.py run --profile "$profile" --workers "$workers"
    code=$?
    echo "=== $(date '+%F %T') profile $profile exited with $code"
    # 130 = interrupted (Ctrl+C / systemd stop): stop the queue, do not start the next profile.
    [[ $code -eq 130 ]] && exit 130
    [[ $code -ne 0 ]] && status=$code
done
exit $status
