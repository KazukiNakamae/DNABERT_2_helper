#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Triton GPU health check script

Usage:
    python triton_health_check.py
"""

import sys
import textwrap


def main():
    print("=" * 60)
    print(" Triton GPU Health Check")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Import Triton
    # --------------------------------------------------------
    try:
        import triton
        import triton.language as tl
        from triton.runtime import driver
    except ImportError as e:
        print("[ERROR] Failed to import Triton.")
        print("        Please install Triton (e.g., `pip install triton`).")
        print("        Details:", e)
        sys.exit(1)

    # --------------------------------------------------------
    # 2. Basic Triton info
    # --------------------------------------------------------
    print("\n[ Triton Info ]")
    triton_version = getattr(triton, "__version__", "<unknown>")
    print(f"triton.__version__         : {triton_version}")

    # --------------------------------------------------------
    # 3. Device info from Triton runtime
    # --------------------------------------------------------
    print("\n[ Triton Runtime / Device Info ]")

    # Triton runtime uses its own driver wrapper; if CUDA is not available,
    # calls may raise errors, so we guard them.
    device_count = 0
    try:
        device_count = driver.get_device_count()
        print(f"driver.get_device_count()  : {device_count}")
    except Exception as e:
        print("[ERROR] Failed to query device count from Triton driver:", e)

    if device_count > 0:
        try:
            current_device = driver.active_device()
            print(f"driver.active_device()     : {current_device}")
        except Exception as e:
            print("[WARN] Failed to get active device from Triton driver:", e)
            current_device = None

        # Try to print some properties of device 0
        try:
            dev0 = driver.Device(0)
            name = dev0.name
            cc = dev0.compute_capability
            total_mem = dev0.total_memory
            print(f"device 0 name              : {name}")
            print(f"device 0 compute capability: {cc}")
            print(f"device 0 total_memory (MiB): {total_mem / (1024**2):.1f}")
        except Exception as e:
            print("[WARN] Failed to get device 0 properties:", e)
    else:
        print("No CUDA devices detected by Triton driver.")

    # --------------------------------------------------------
    # 4. Minimal Triton kernel test
    # --------------------------------------------------------
    print("\n[ Triton Kernel Test ]")

    # If there is no device, we skip kernel test
    if device_count == 0:
        print("Triton kernel test skipped (no CUDA devices detected).")
    else:
        try:
            import torch
        except ImportError:
            print("[WARN] PyTorch is not installed;")
            print("       Triton kernel test using torch tensors is skipped.")
        else:
            # Define a simple Triton kernel: y[i] = x[i] + 1
            @triton.jit
            def add_one_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(axis=0)
                block_start = pid * BLOCK_SIZE
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = x + 1
                tl.store(y_ptr + offsets, y, mask=mask)

            # Prepare small tensors on CUDA
            try:
                if not torch.cuda.is_available():
                    print("[WARN] torch.cuda.is_available() is False;")
                    print("       Triton kernel test using torch cuda tensors is skipped.")
                else:
                    device = "cuda"
                    n_elements = 1024
                    x = torch.arange(n_elements, dtype=torch.float32, device=device)
                    y = torch.empty_like(x)

                    BLOCK_SIZE = 256
                    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

                    add_one_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)

                    # Verify correctness on CPU
                    y_cpu = y.cpu()
                    expected = (x.cpu() + 1)
                    if torch.allclose(y_cpu, expected):
                        print("Triton kernel execution: OK")
                        print("Output matches expected values (x + 1).")
                    else:
                        print("[ERROR] Triton kernel ran, but output does not match expected values.")
            except Exception as e:
                print("[ERROR] Triton kernel execution failed:", e)

    # --------------------------------------------------------
    # 5. Warnings / Suggestions
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print(" Warnings / Suggestions")
    print("=" * 60)

    warnings = []

    if device_count == 0:
        warnings.append(textwrap.dedent("""
            - Triton could not see any CUDA devices (driver.get_device_count() == 0).
              Please check:
                * Whether an NVIDIA GPU is present and visible to the OS (`nvidia-smi`)
                * Whether the correct NVIDIA driver is installed
                * In container environments, whether `--gpus` option /
                  NVIDIA Container Toolkit is configured correctly
        """).strip())

    # If PyTorch kernel test was skipped or failed, we cannot know from here,
    # but this script keeps Triton-focused messages.

    if not warnings:
        print("No critical issues detected in Triton checks. 🎉")
    else:
        for i, w in enumerate(warnings, 1):
            print(f"\n[WARN {i}]")
            print(w)

    print("\nTriton health check completed.")


if __name__ == "__main__":
    main()
