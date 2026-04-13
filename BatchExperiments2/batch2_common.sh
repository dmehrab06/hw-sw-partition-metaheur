#!/bin/bash
set -euo pipefail

export HWSW_METHOD_RUNTIME_PROFILE="${HWSW_METHOD_RUNTIME_PROFILE:-balanced}"

bootstrap_batchexperiments2_family() {
  local family="$1"
  local root="$2"
  local python_bin="$3"
  local source_root="${BOOTSTRAP_SOURCE_ROOT:-$root/BatchExperiments}"
  local dest_root="${BOOTSTRAP_DEST_ROOT:-$root/BatchExperiments2}"

  "$python_bin" "$root/tools/bootstrap_batch_experiments2_results.py" \
    --source-root "$source_root" \
    --dest-root "$dest_root" \
    --families "$family"
}
