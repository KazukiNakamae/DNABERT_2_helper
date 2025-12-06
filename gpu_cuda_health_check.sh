#!/usr/bin/env bash

# GPU / CUDA OS-level health check
# Usage:
#   bash gpu_cuda_health_check.sh

set -u

echo "============================================================"
echo " GPU / CUDA OS-level Health Check"
echo "============================================================"

# ------------------------------------------------------------
# 1. Basic system info
# ------------------------------------------------------------
echo
echo "[ System Info ]"
echo "Hostname          : $(hostname)"
echo "Kernel            : $(uname -sr)"
echo "Architecture      : $(uname -m)"
echo "Date              : $(date)"

# ------------------------------------------------------------
# 2. Check for NVIDIA GPU via lspci
# ------------------------------------------------------------
echo
echo "[ PCI GPU Detection (lspci) ]"

if command -v lspci >/dev/null 2>&1; then
  NVIDIA_LINES=$(lspci | grep -i 'vga\|3d\|display' | grep -i nvidia || true)
  if [ -n "$NVIDIA_LINES" ]; then
    echo "Detected NVIDIA GPU(s) in lspci:"
    echo "$NVIDIA_LINES"
  else
    echo "No NVIDIA GPU detected by lspci (or lspci does not list it as VGA/3D/Display)."
  fi
else
  echo "lspci command not found. Cannot check PCI devices."
fi

# ------------------------------------------------------------
# 3. Check nvidia-smi (driver & basic GPU info)
# ------------------------------------------------------------
echo
echo "[ NVIDIA Driver / nvidia-smi ]"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi found at: $(command -v nvidia-smi)"
  echo
  echo "---- nvidia-smi (summary) ----"
  # Show only the top info and one GPU table page to keep it readable
  nvidia-smi
else
  echo "nvidia-smi NOT found."
  echo "You likely do NOT have the NVIDIA driver (or CUDA driver) properly installed."
fi

# ------------------------------------------------------------
# 4. Check CUDA Toolkit (nvcc & version)
# ------------------------------------------------------------
echo
echo "[ CUDA Toolkit (nvcc) ]"

if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc found at: $(command -v nvcc)"
  echo
  echo "---- nvcc --version ----"
  nvcc --version
else
  echo "nvcc NOT found in PATH."
  echo "This usually means the CUDA Toolkit is not installed or not in PATH."
fi

# ------------------------------------------------------------
# 5. Check /usr/local/cuda* directories
# ------------------------------------------------------------
echo
echo "[ /usr/local/cuda* directories ]"

CUDA_DIRS=$(ls -d /usr/local/cuda* 2>/dev/null || true)
if [ -n "$CUDA_DIRS" ]; then
  echo "Found the following CUDA-related directories:"
  echo "$CUDA_DIRS"
else
  echo "No /usr/local/cuda* directories found."
fi

# Show version.txt if present
for d in $CUDA_DIRS; do
  if [ -f "$d/version.txt" ]; then
    echo
    echo "---- $d/version.txt ----"
    cat "$d/version.txt"
  fi
done

# ------------------------------------------------------------
# 6. Check LD_LIBRARY_PATH and ldconfig for CUDA libs
# ------------------------------------------------------------
echo
echo "[ CUDA Libraries in ldconfig / LD_LIBRARY_PATH ]"

echo "LD_LIBRARY_PATH : ${LD_LIBRARY_PATH:-<not set>}"

if command -v ldconfig >/dev/null 2>&1; then
  echo
  echo "Searching ldconfig cache for CUDA libraries (libcudart.so, libcuda.so):"
  ldconfig -p 2>/dev/null | grep -E 'libcudart\.so|libcuda\.so' || echo "No CUDA-related libs found in ldconfig cache."
else
  echo "ldconfig command not found. Skipping ldconfig check."
fi

# ------------------------------------------------------------
# 7. Optional: Check kernel modules
# ------------------------------------------------------------
echo
echo "[ Kernel Modules ]"

if command -v lsmod >/dev/null 2>&1; then
  echo "Checking if nvidia kernel module is loaded:"
  if lsmod | grep -q '^nvidia'; then
    lsmod | grep '^nvidia'
  else
    echo "nvidia kernel module is NOT loaded."
  fi
else
  echo "lsmod not found. Skipping kernel module check."
fi

# ------------------------------------------------------------
# 8. Summary / Warnings
# ------------------------------------------------------------
echo
echo "============================================================"
echo " Summary / Warnings"
echo "============================================================"

WARNINGS=()

# Helper function
add_warning() {
  WARNINGS+=("$1")
}

# Detect basic conditions
HAS_NVIDIA_SMI=0
HAS_NVCC=0
HAS_NVIDIA_PCI=0

if command -v nvidia-smi >/dev/null 2>&1; then
  HAS_NVIDIA_SMI=1
fi

if command -v nvcc >/dev/null 2>&1; then
  HAS_NVCC=1
fi

if command -v lspci >/dev/null 2>&1; then
  if lspci | grep -i 'vga\|3d\|display' | grep -qi nvidia; then
    HAS_NVIDIA_PCI=1
  fi
fi

# Conditions

if [ "$HAS_NVIDIA_PCI" -eq 1 ] && [ "$HAS_NVIDIA_SMI" -eq 0 ]; then
  add_warning "NVIDIA GPU detected by lspci, but nvidia-smi is missing.
  -> The NVIDIA driver might not be installed or not loaded correctly."
fi

if [ "$HAS_NVIDIA_SMI" -eq 1 ]; then
  # If nvidia-smi works but nvcc is missing
  if [ "$HAS_NVCC" -eq 0 ]; then
    add_warning "nvidia-smi is available (driver seems installed), but nvcc is missing.
  -> This usually means the CUDA Toolkit is not installed or not added to PATH.
  -> If you need to compile CUDA kernels, install the CUDA Toolkit."
  fi
fi

if [ "$HAS_NVIDIA_PCI" -eq 0 ] && [ "$HAS_NVIDIA_SMI" -eq 0 ]; then
  add_warning "No NVIDIA GPU found via lspci, and nvidia-smi was not found.
  -> This system might not have an NVIDIA GPU, or virtualization/container settings
     hide the GPU from this environment."
fi

# Output warnings
if [ "${#WARNINGS[@]}" -eq 0 ]; then
  echo "No critical issues detected at OS level for GPU / CUDA. 🎉"
else
  i=1
  for w in "${WARNINGS[@]}"; do
    echo
    echo "[WARN $i]"
    echo "$w"
    i=$((i+1))
  done
fi

echo
echo "OS-level GPU / CUDA health check completed."
