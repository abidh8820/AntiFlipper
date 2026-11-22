#!/usr/bin/env bash
# Run ./monitor.sh for the live monitor of the running profile (or pass --profile NAME), or ./monitor.sh --once for a snapshot.
set -eu

script_dir=$(dirname -- "${BASH_SOURCE[0]}")
watch=(--watch)
if [[ ${1:-} == --once ]]; then
    watch=()
    shift
fi

exec python3 "$script_dir/monitor.py" "${watch[@]}" "$@"
