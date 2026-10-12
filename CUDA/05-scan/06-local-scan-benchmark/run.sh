#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x ./main ]]; then
    echo "Build main with the CMake or nvcc command in README.md first."
    exit 2
fi
if [[ $# -eq 0 ]]; then
    echo "Running a bounded smoke with three measured repetitions."
    set -- --size 65536 --warmup 3 --repetitions 3 --batch 3
fi
exec ./main "$@"
