#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Triton GPU health check script (v2; Triton 3.x 対応・簡略安定版)

Usage:
    python DNABERT_2_helper/triton_health_check.py
"""

import textwrap


def main():
    print("=" * 60)
    print(" Triton GPU Health Check (v2 for Triton 3.x, stable)")
    print("=" * 60)

    warnings = []

    # --------------------------------------------------------
    # 1. Import Triton
    # --------------------------------------------------------
    try:
        import triton
        import triton.language as tl
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
            "torch.cuda integration. Install torch with CUDA support."
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
    # 3. Triton runtime / target info (最低限だけ)
    # --------------------------------------------------------
    print("\n[ Triton Runtime / Device Info ]")
    try:
        from triton.runtime import driver

        # Triton 3.x では active driver から target を取得
        target = driver.active.get_current_target()
        print("Triton target backend      :", target.backend)
        print("Triton target arch         :", target.arch)

        # ここでは Triton の device props 取得は行わず、
        # 詳細なハード情報は PyTorch 側に任せる
    except Exception as e:
        print("[WARN] Failed to query target info from Triton driver.")
        print("       Details:", repr(e))
        warnings.append(
            "Triton runtime failed to query target information. "
            "If PyTorch CUDA is working, Triton may still be usable."
        )

    # --------------------------------------------------------
    # 4. Triton Kernel Test
    # --------------------------------------------------------
    print("\n[ Triton Kernel Test ]")

    if not torch_cuda_ok:
        print("[INFO] Skip Triton kernel test because torch.cuda is not available.")
        warnings.append(
            "Skip Triton kernel test because torch.cuda.is_available() is False."
        )
    else:
        try:
            # Triton カーネルを通常の Python 関数として定義
            @triton.jit
            def add_one_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(axis=0)
                block_start = pid * BLOCK_SIZE
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = x + 1
                tl.store(y_ptr + offsets, y, mask=mask)

            # テスト用テンソル
            n_elements = 1024
            x = torch.arange(n_elements, dtype=torch.float32, device="cuda")
            y = torch.empty_like(x)

            BLOCK_SIZE = 256
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            # Triton カーネルを起動
            add_one_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)

            # 結果チェック
            if not torch.allclose(y, x + 1):
                raise RuntimeError("Triton kernel produced incorrect results.")

            print("[OK] Triton JIT kernel ran successfully and produced correct results.")
        except Exception as e:
            print("[ERROR] Triton kernel test failed:", repr(e))
            warnings.append(
                "Triton JIT kernel failed to run: " + repr(e)
            )

    # --------------------------------------------------------
    # 5. Warnings / summary
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print(" Warnings / Suggestions")
    print("=" * 60)

    if not warnings:
        print("No major problems detected. Triton seems to be working correctly.")
    else:
        for i, w in enumerate(warnings, 1):
            print(f"\n[WARN {i}]")
            print(textwrap.fill(w, width=70))

    print("\nTriton health check completed.")


if __name__ == "__main__":
    main()
