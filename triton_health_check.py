#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Triton GPU health check script (v2; Triton 3.x 対応)

Usage:
    python DNABERT_2_helper/triton_health_check_v2.py
"""

import sys
import textwrap


def main():
    print("=" * 60)
    print(" Triton GPU Health Check (v2 for Triton 3.x)")
    print("=" * 60)

    warnings = []

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
        return

    print("\n[ Triton Info ]")
    triton_version = getattr(triton, "__version__", "<unknown>")
    print(f"triton.__version__         : {triton_version}")

    # --------------------------------------------------------
    # 2. Torch / CUDA device info
    # --------------------------------------------------------
    print("\n[ Torch / CUDA Info ]")
    device_count = 0
    torch_cuda_ok = False

    try:
        import torch
    except ImportError:
        print("[WARN] PyTorch is not installed; skip torch.cuda checks.")
        warnings.append(
            "PyTorch is not installed, so Triton will not be able to use "
            "torch.cuda integration. Install torch with CUDA support "
            "(e.g., torch==2.9.1+cu130)."
        )
    else:
        print("torch.__version__          :", torch.__version__)
        if not torch.cuda.is_available():
            print("torch.cuda.is_available()  : False")
            warnings.append(
                "torch.cuda.is_available() is False. "
                "Check NVIDIA driver, CUDA toolkit, and environment variables "
                "(e.g. CUDA_VISIBLE_DEVICES)."
            )
        else:
            torch_cuda_ok = True
            device_count = torch.cuda.device_count()
            current = torch.cuda.current_device()
            print("torch.cuda.is_available()  : True")
            print("torch.cuda.device_count()  :", device_count)
            print("torch.cuda.current_device():", current)
            print("torch.cuda.get_device_name():",
                  torch.cuda.get_device_name(current))

    # --------------------------------------------------------
    # 3. Triton runtime / device properties
    # --------------------------------------------------------
    print("\n[ Triton Runtime / Device Info ]")

    # Triton 3.x では driver.get_device_count() は存在しないので、
    # torch.cuda 情報を優先して使う。
    if device_count == 0:
        print("No CUDA devices reported by torch.cuda.")
    else:
        try:
            # active driver / target 情報
            target = driver.active.get_current_target()
            print("Triton target backend      :", target.backend)
            print("Triton target arch         :", target.arch)

            # デバイスプロパティを Triton 経由で取得
            props = driver.active.utils.get_device_properties(0)
            # props は dict 形式のはず
            print("device 0 name              :", props.get("name", "<unknown>"))
            print("device 0 multiprocessors   :", props.get("multiprocessor_count"))
            print("device 0 max_shared_mem    :", props.get("max_shared_mem"))
            print("device 0 warpSize          :", props.get("warpSize"))
        except Exception as e:
            print("[WARN] Failed to query Triton driver properties:", e)
            warnings.append(
                f"Triton runtime could not query device properties: {e!r}"
            )

    # --------------------------------------------------------
    # 4. Minimal Triton kernel test
    # --------------------------------------------------------
    print("\n[ Triton Kernel Test ]")

    if not torch_cuda_ok or device_count == 0:
        print("Triton kernel test skipped (no CUDA devices according to torch).")
        warnings.append(
            "Triton kernel test was skipped because torch.cuda "
            "did not report any available CUDA devices."
        )
    else:
        try:
            # y = x + 1 のシンプルなカーネル
            @triton.jit
            def add_one_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(axis=0)
                block_start = pid * BLOCK_SIZE
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = x + 1
                tl.store(y_ptr + offsets, y, mask=mask)

            n_elements = 1024
            x = torch.arange(n_elements, dtype=torch.float32, device="cuda")
            y = torch.empty_like(x)

            BLOCK_SIZE = 256
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

            add_one_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)

            if torch.allclose(y, x + 1):
                print("Triton kernel test: OK (y == x + 1)")
            else:
                print("Triton kernel test: FAILED (y != x + 1)")
                warnings.append("Triton kernel ran but produced incorrect result.")
        except Exception as e:
            print("[ERROR] Triton kernel test failed:", e)
            warnings.append(f"Triton JIT kernel failed to run: {e!r}")

    # --------------------------------------------------------
    # 5. Summary
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print(" Warnings / Suggestions")
    print("=" * 60)

    if not warnings:
        print("No critical issues detected in Triton checks. 🎉")
    else:
        for i, w in enumerate(warnings, 1):
            print(f"\n[WARN {i}]")
            print(textwrap.dedent(w).strip())

    print("\nTriton health check completed.")


if __name__ == "__main__":
    main()
