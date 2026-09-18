#!/usr/bin/env bash
# HOMER enrichment pipeline orchestrator

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/config.sh"
mkdir -p "${OUT_ROOT}" "${PREPARSED_DIR}"

echo "[1/4] pool regions"
bash "${HERE}/scripts/pool_regions.sh"\\