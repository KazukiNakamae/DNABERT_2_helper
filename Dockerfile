# syntax=docker/dockerfile:1

FROM --platform=linux/amd64 condaforge/miniforge3:latest
# Miniforge image overview: /opt/conda にインストール済み :contentReference[oaicite:4]{index=4}

SHELL ["bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# (任意) 便利ツール。最小で良ければ削除可
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# まず環境YAMLだけコピー（キャッシュを効かせる）
COPY env_dnabert_eval.yaml /tmp/env_dnabert_eval.yaml

# YAMLからconda環境作成
# env名は YAML の name: dnabert_eval に従う :contentReference[oaicite:5]{index=5}
RUN conda env create -f /tmp/env_dnabert_eval.yaml \
    && conda clean -a -y

# スクリプト実行に実務上ほぼ必要になる追加依存（必要に応じて調整）
# - scikit-learn: evaluate_predictions_csv.py の各種metrics用途
# - biopython: FASTA処理の定番（occlusionスクリプトで使っている可能性が高い）
# - einops: 推論に必要
RUN conda install -n dnabert_eval -c conda-forge -y \
      scikit-learn biopython einops \
    && conda clean -a -y

# GPUを使う場合のみ（デフォルトは0推奨：CPUで十分なら速く軽い）
ARG ENABLE_CUDA=0
RUN if [ "$ENABLE_CUDA" = "1" ]; then \
      # conda-forge由来のtorchを掴んでいる可能性があるため、一旦外してから入れ直す
      conda remove -n dnabert_eval -y pytorch torchvision torchaudio || true; \
      conda install -n dnabert_eval -y --override-channels -c pytorch -c nvidia \
        pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 && \
      conda clean -a -y ; \
    fi

# --- Disable Triton / FlashAttention (CPU inference) ---
# DNABERT-2 は Triton が import できると FlashAttention(Triton) 経路に入り、
# CPU では assert q.is_cuda ... で落ちるため、triton を環境から除去する
RUN set -eux; \
    # conda 管理の triton/torchtriton が居れば削除（居なければ無視）
    conda remove -n dnabert_eval -y triton torchtriton pytorch-triton 2>/dev/null || true; \
    # pip で入っているケースも潰す
    conda run -n dnabert_eval python -m pip uninstall -y triton torchtriton || true; \
    # 念のため site-packages に残骸があれば削除
    SP="$(conda run -n dnabert_eval python -c 'import site; print(site.getsitepackages()[0])')"; \
    rm -rf "${SP}/triton" "${SP}"/triton-* "${SP}"/torchtriton* /root/.triton; \
    conda clean -a -y

# conda環境をデフォルトで使う
ENV PATH=/opt/conda/envs/dnabert_eval/bin:$PATH

# (任意) リポジトリ全体をイメージに含める。運用で bind mount するなら削除してOK。
COPY . /workspace

# 簡易自己診断（ビルド時に失敗して欲しい場合は有効）
RUN python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())" \
 && python -c "import transformers; print('transformers', transformers.__version__)"

CMD ["bash"]
