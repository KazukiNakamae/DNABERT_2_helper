#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch / CUDA health check script

Usage:
    python torch_health_check.py
"""

import sys
import textwrap


def main():
    print("=" * 60)
    print(" PyTorch / CUDA Health Check")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Import PyTorch
    # --------------------------------------------------------
    try:
        import torch
    except ImportError:
            print("[ERROR] Failed to import PyTorch.")
            print("        Please check if PyTorch is installed (e.g., `pip show torch`).")
            sys.exit(1)

    # --------------------------------------------------------
    # 2. Basic information
    # --------------------------------------------------------
    print("\n[ PyTorch Info ]")
    print(f"torch.__version__          : {torch.__version__}")

    # torch.version.cuda is a string for CUDA builds, None for CPU-only builds
    cuda_version = getattr(torch.version, "cuda", None)
    print(f"torch.version.cuda         : {cuda_version}")

    # --------------------------------------------------------
    # 3. CUDA availability and device info
    # --------------------------------------------------------
    print("\n[ CUDA / GPU Info ]")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available()  : {cuda_available}")

    device_count = torch.cuda.device_count() if cuda_available else 0
    print(f"torch.cuda.device_count()  : {device_count}")

    if cuda_available and device_count > 0:
        try:
            current_device = torch.cuda.current_device()
        except Exception as e:
            current_device = None
            print(f"[WARN] torch.cuda.current_device() raised an error: {e}")
        print(f"torch.cuda.current_device(): {current_device}")

        # List all device names
        for idx in range(device_count):
            try:
                name = torch.cuda.get_device_name(idx)
            except Exception as e:
                name = f"<failed to get name: {e}>"
            print(f"torch.cuda.get_device_name({idx}) : {name}")
    else:
        print("torch.cuda.current_device(): <not available because CUDA is not available>")
        print("torch.cuda.get_device_name(): <not available because CUDA is not available>")

    # --------------------------------------------------------
    # 4. Additional status information (optional)
    # --------------------------------------------------------
    print("\n[ Additional Status ]")
    if cuda_available and device_count > 0:
        try:
            cap = torch.cuda.get_device_capability()
            print(f"compute capability          : {cap}")
        except Exception as e:
            print(f"[WARN] Failed to get compute capability: {e}")

        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory
            print(f"device 0 total_memory (MiB) : {total_mem / (1024**2):.1f}")
        except Exception as e:
            print(f"[WARN] Failed to get total_memory: {e}")
    else:
        print("CUDA device info is not available (CUDA disabled or no GPU detected).")

    # --------------------------------------------------------
    # 5. Simple tensor operation checks
    # --------------------------------------------------------
    print("\n[ Tensor Operation Check ]")
    try:
        x = torch.randn(2, 2)
        print("Tensor creation on CPU: OK, shape =", x.shape)
    except Exception as e:
        print("[ERROR] Failed to create tensor on CPU:", e)
        x = None

    if cuda_available and device_count > 0 and x is not None:
        try:
            x_gpu = x.to("cuda")
            y_gpu = x_gpu @ x_gpu.T
            print("Tensor operation on GPU: OK, shape =", y_gpu.shape)
        except Exception as e:
            print("[ERROR] Tensor operation on GPU failed:", e)
    else:
        print("GPU tensor operation check skipped (CUDA not available or no GPU).")

    # --------------------------------------------------------
    # 6. Warnings / Suggestions
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print(" Warnings / Suggestions")
    print("=" * 60)

    warnings = []

    # (1) CUDA not available
    if not cuda_available:
        warnings.append(textwrap.dedent("""
            - CUDA is not available.
              If you want to use a GPU, please check:
                * Whether a CUDA-enabled build of PyTorch is installed
                * Whether an NVIDIA GPU and the correct driver are installed
                * For container environments, whether `--gpus` option /
                  NVIDIA Container Toolkit is configured correctly
        """).strip())

    # (2) cuda.is_available True but device_count == 0
    if cuda_available and device_count == 0:
        warnings.append(textwrap.dedent("""
            - torch.cuda.is_available() is True, but device_count() == 0.
              There may be an issue with the driver or Docker configuration.
        """).strip())

    # (3) CUDA version info is None (likely CPU-only build)
    if cuda_version is None:
        warnings.append(textwrap.dedent("""
            - The installed PyTorch is likely a CPU-only build
              (torch.version.cuda is None).
              If you need GPU support, please reinstall a CUDA-enabled
              version of PyTorch.
        """).strip())

    # (4) Hints if machine has GPU but PyTorch cannot use it (general advice)
    if cuda_version is None and not cuda_available:
        warnings.append(textwrap.dedent("""
            - If your machine actually has an NVIDIA GPU but PyTorch
              cannot use it:
                * Check `nvidia-smi` to see if the OS recognizes the GPU
                * Reinstall PyTorch with a proper CUDA build, for example:
                  `pip3 install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`
        """).strip())

    if not warnings:
        print("No critical issues detected. 🎉")
    else:
        for i, w in enumerate(warnings, 1):
            print(f"\n[WARN {i}]")
            print(w)

    print("\nHealth check completed.")


if __name__ == "__main__":
    main()
