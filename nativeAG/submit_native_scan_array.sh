#!/usr/bin/env bash
# Usage: ./submit_native_scan_array.sh --targets targets.tsv --partition gpu_h200 \
#          --max-concurrent 2 -- [shared scanner options, excluding --targets/--genes]
set -Eeuo pipefail
scan_package_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
scan_targets=""
scan_partition="gpu_h200"
scan_concurrency=2
while (($#)); do
    case "$1" in
        --targets) scan_targets="$2"; shift 2 ;;
        --partition) scan_partition="$2"; shift 2 ;;
        --max-concurrent) scan_concurrency="$2"; shift 2 ;;
        --) shift; break ;;
        *) printf 'Unknown submission option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ -n "$scan_targets" && -f "$scan_targets" ]] || { printf 'Provide --targets FILE\n' >&2; exit 2; }
[[ "$scan_concurrency" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid --max-concurrent\n' >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-python}"
scan_n="$("$PYTHON_BIN" - "$scan_package_root" "$scan_targets" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from scan_native_alphagenome import read_targets
print(len(read_targets(sys.argv[2])))
PY
)"
# Freeze the manifest so later edits cannot reassign array tasks to different genes.
scan_manifest_dir="${NATIVE_ARRAY_MANIFEST_DIR:-$PWD/native_array_manifests}"
mkdir -p -- "$scan_manifest_dir"
scan_snapshot="$(mktemp "$scan_manifest_dir/targets.XXXXXXXX.tsv")"
cp -- "$scan_targets" "$scan_snapshot"
scan_snapshot="$("$PYTHON_BIN" - "$scan_snapshot" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
export NATIVE_SCAN_ROOT="$scan_package_root"
export NATIVE_ARRAY_TARGETS="$scan_snapshot"
sbatch --partition="$scan_partition" --array="0-$((scan_n-1))%$scan_concurrency" \
    --export=ALL "$scan_package_root/run_native_scan.sbatch" "$@"
