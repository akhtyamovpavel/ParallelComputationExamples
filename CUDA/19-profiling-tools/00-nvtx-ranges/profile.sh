#!/bin/bash
# Скрипты для профилирования CUDA-приложения разными инструментами.
#
# Ссылки:
#   nvprof:         https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof
#   Visual Profiler: https://docs.nvidia.com/cuda/profiler-users-guide/index.html#visual-profiler
#   Nsight Systems: https://developer.nvidia.com/nsight-systems
#   Nsight Compute: https://developer.nvidia.com/nsight-compute

set -e

BINARY="./main"
if [ ! -f "$BINARY" ]; then
    echo "Binary not found. Build first: cmake .. && make"
    exit 1
fi

echo "=== 1. Nsight Systems (timeline + NVTX ranges) ==="
echo "  nsys profile --trace=cuda,nvtx -o nsys_report $BINARY"
echo ""
if command -v nsys &> /dev/null; then
    nsys profile --trace=cuda,nvtx -o nsys_report "$BINARY"
    echo "  -> Open with: nsys-ui nsys_report.nsys-rep"
else
    echo "  nsys not found; install Nsight Systems"
fi

echo ""
echo "=== 2. Nsight Compute (kernel metrics) ==="
echo "  ncu --set full $BINARY"
echo ""
if command -v ncu &> /dev/null; then
    ncu --set full "$BINARY"
else
    echo "  ncu not found; install Nsight Compute"
fi

echo ""
echo "=== 3. nvprof (legacy, CUDA <= 11) ==="
echo "  nvprof --print-gpu-trace $BINARY"
echo ""
if command -v nvprof &> /dev/null; then
    nvprof --print-gpu-trace "$BINARY"
else
    echo "  nvprof not found (deprecated since CUDA 12; use nsys/ncu)"
fi
