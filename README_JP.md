# DNABERT_2_helper

[English version](README.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2)のファインチューニング、評価、デプロイを支援する非公式のヘルパースクリプト集です。

## 概要

DNABERT_2_helperは、ゲノム配列分類タスクにおいてDNABERT-2を実用的に活用するためのユーティリティスクリプト集です。このツールキットは、**「Workflow for Fine-tuning and Evaluating DNA Language Models for Specific Genomics Issues」**（Nakamae & Bono, Bio-protocol, 2025）で記述されたプロトコルの一部として紹介しています。

## 機能

- 🔍 **環境ヘルスチェック**: GPU/CUDAおよびPyTorch設定の検証
- 🔧 **ハイパーパラメータ最適化**: 効率的なハイパーパラメータチューニングのためのグリッドサーチラッパー
- 📊 **データセット準備**: サブセット作成とラベルバランシングユーティリティ
- 🚀 **CPU対応推論**: GPUを必要としないDockerベースの評価環境
- 📈 **包括的評価**: 正解率、F1、MCC、適合率、再現率の計算
- 🎨 **可視化**: モデル性能を比較する論文掲載品質のヒートマップ生成

## 必要要件

### ファインチューニング用（GPU環境）
- CUDA対応NVIDIA GPU（H100 80GBでテスト済み）
- CUDA Toolkit 12.1以上
- Python 3.8以上
- 120GB以上のRAM（推奨）
- Ubuntu 22.04 LTSまたは類似環境

### 推論用（CPU環境）
- Docker
- 16GB以上のRAM（推奨）
- macOSまたはLinux

## インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/KazukiNakamae/DNABERT_2_helper.git
cd DNABERT_2_helper
```

### 2. GPU環境のセットアップ（ファインチューニング用）

プロトコル論文のセクションB1に記載された詳細なセットアップ手順に従ってください。主な手順：

```bash
# conda環境の作成
conda create -n dna conda-forge::python=3.8
conda activate dna

# CUDA対応PyTorchのインストール
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 追加の依存関係をインストール
pip3 install -r modified_requirements.txt

# ヘルスチェックの実行
bash gpu_cuda_health_check.sh
python torch_health_check.py
```

### 3. CPU環境のセットアップ（推論用）

```bash
# CPU評価用Dockerイメージのビルド
docker buildx build --platform linux/amd64 \
  --build-arg ENABLE_CUDA=0 \
  -t kazukinakamae/dnabert-eval:amd64 \
  -f Dockerfile .

# キャッシュディレクトリの作成
mkdir -p .hf_cache
```

## クイックスタート

### 1. データセットの準備

データセットが以下のCSV形式であることを確認してください：
- `train.csv`, `dev.csv`, `test.csv`
- 必須カラム: `sequence`（DNA配列）, `label`（0または1）

### 2. ハイパーパラメータグリッドサーチ

```bash
bash grid_search_train_wrapper.sh \
  --train_script train.py \
  --data_path /path/to/dataset \
  --model_name_or_path zhihan1996/DNABERT-2-117M \
  --output_root /path/to/output \
  --run_name_prefix my_experiment \
  --nproc_per_node 1 \
  --batch_configs "8x8,16x8,32x8" \
  --lrs "1e-5,2e-5" \
  --epochs "2" \
  --warmup_ratios "0.05,0.1" \
  --weight_decays "0.01,0.03" \
  --extra_args "--kmer -1 --model_max_length 10 --fp16"
```

### 3. 最適なハイパーパラメータの選択

```bash
python select_best_run.py \
  --output_root /path/to/grid_search_output \
  --metric_name eval_loss \
  --metric_mode min \
  --num_train_epochs 8 \
  --print_full_command
```

### 4. ファインチューニング

[Northwestern University MAGICS Labsによる公式リポジトリ](https://github.com/MAGICS-LAB/DNABERT_2)をご参照ください。

### 5. 推論の実行

```bash
docker run --platform=linux/amd64 --rm \
  -v $(pwd):/DATA -w /DATA \
  -v $(pwd)/.hf_cache:/root/.cache/huggingface \
  kazukinakamae/dnabert-eval:amd64 \
  python predict_hf_classifier_csv.py \
    --model_dir /path/to/fine_tuned_model \
    --input_file test.csv \
    --output_file predictions.csv \
    --text_column sequence \
    --max_length 10 \
    --batch_size 32
```

### 6. 性能評価

```bash
docker run --platform=linux/amd64 --rm \
  -v $(pwd):/DATA -w /DATA \
  kazukinakamae/dnabert-eval:amd64 \
  python evaluate_predictions_csv.py \
    --gold_file test.csv \
    --pred_file predictions.csv \
    --out_prefix eval_results \
    --gold_label_column label \
    --pred_label_column pred_label \
    --average macro \
    --pos_label 1
```

### 7. 結果の可視化

```bash
# 複数の実験からメトリクスを統合
docker run --platform=linux/amd64 --rm \
  -v $(pwd):/DATA -w /DATA \
  kazukinakamae/dnabert-eval:amd64 \
  python merge_metrics_csvs.py \
    --input_glob "eval_summary/**/*_metrics.csv" \
    --recursive \
    --out_csv merged_metrics.csv

# ヒートマップの生成
docker run --platform=linux/amd64 --rm \
  -v $(pwd):/DATA -w /DATA \
  kazukinakamae/dnabert-eval:amd64 \
  python plot_heatmaps_from_merged.py \
    --input_csv merged_metrics.csv \
    --outdir eval_summary \
    --dpi 500 \
    --formats png,tiff,pdf \
    --annotate
```

## スクリプトリファレンス

### ヘルスチェックスクリプト

#### `gpu_cuda_health_check.sh`
OSレベルでのGPUとCUDAインストールの検証。

```bash
bash gpu_cuda_health_check.sh
```

#### `torch_health_check.py`
PyTorchがGPUを利用可能か確認。

```bash
python torch_health_check.py
```

### データ準備スクリプト

#### `subset_dataset_csv.py`
グリッドサーチやクイックテスト用のデータセットサブセットを作成。

```bash
python subset_dataset_csv.py \
  --input_dir /path/to/dataset \
  --output_dir /path/to/subset \
  --subset_ratio 0.2 \
  --seed 1 \
  --label_column label
```

**オプション:**
- `--input_dir`: train.csv, dev.csv, test.csvを含むディレクトリ
- `--output_dir`: サブセット出力ディレクトリ
- `--subset_ratio`: サンプリングするデータの割合（0.0-1.0）
- `--seed`: 再現性のための乱数シード
- `--label_column`: ラベルカラムの名前

### トレーニングスクリプト

#### `grid_search_train_wrapper.sh`
DNABERT-2ファインチューニングのためのハイパーパラメータグリッドサーチを自動化。

```bash
bash grid_search_train_wrapper.sh \
  --train_script train.py \
  --data_path /path/to/dataset \
  --model_name_or_path zhihan1996/DNABERT-2-117M \
  --output_root /path/to/output \
  --run_name_prefix experiment \
  --nproc_per_node 1 \
  --batch_configs "8x8,16x8" \
  --lrs "1e-5,2e-5" \
  --epochs "2,4" \
  --warmup_ratios "0.05,0.1" \
  --weight_decays "0.01,0.03" \
  --extra_args "--kmer -1 --model_max_length 10 --fp16"
```

**主要オプション:**
- `--batch_configs`: "バッチサイズ x 勾配累積ステップ"形式
- `--lrs`: カンマ区切りの学習率
- `--epochs`: トレーニングエポック数
- `--warmup_ratios`: ウォームアップ比率またはステップ数
- `--weight_decays`: 重み減衰強度

#### `select_best_run.py`
グリッドサーチ結果から最適なハイパーパラメータ組み合わせを特定。

```bash
python select_best_run.py \
  --output_root /path/to/grid_output \
  --metric_name eval_loss \
  --metric_mode min \
  --num_train_epochs 8 \
  --print_full_command
```

**オプション:**
- `--metric_name`: 最適化するメトリクス（eval_loss、accuracy、f1など）
- `--metric_mode`: "min"または"max"
- `--print_full_command`: 即実行可能なトレーニングコマンドを生成

#### `resume_same_task_same_data.py`
チェックポイントからファインチューニングを再開。

```bash
python resume_same_task_same_data.py \
  --checkpoint_dir /path/to/checkpoint \
  --data_path /path/to/dataset \
  --output_dir /path/to/output \
  --add_epochs 4 \
  --do_test_eval
```

### ベースラインスクリプト

#### `onehot_cnn_baseline.py`
3層one-hot CNNベースラインモデルを訓練。

```bash
python onehot_cnn_baseline.py \
  --train_csv train.csv \
  --dev_csv dev.csv \
  --test_csv test.csv \
  --outdir baseline_output \
  --seq_len 700 \
  --epochs 30 \
  --batch_size 128 \
  --lr 1e-3 \
  --weight_decay 1e-4
```

#### `predict_motif_baseline_csv.py`
モチーフベースのベースライン予測を生成。

```bash
python predict_motif_baseline_csv.py \
  --input test.csv \
  --output motif_predictions.csv \
  --pattern WCW \
  --start 19
```

**オプション:**
- `--pattern`: モチーフパターン（例: WCW、ACW）
- `--start`: モチーフマッチングの0始まり開始位置

### 推論スクリプト

#### `predict_hf_classifier_csv.py`
ファインチューニング済みDNABERT-2モデルを使用して予測を生成。

```bash
python predict_hf_classifier_csv.py \
  --model_dir /path/to/model \
  --input_file test.csv \
  --output_file predictions.csv \
  --text_column sequence \
  --max_length 10 \
  --batch_size 32 \
  --trust_remote_code
```

**オプション:**
- `--model_dir`: ファインチューニング済みモデルディレクトリへのパス
- `--text_column`: 配列を含むカラム名
- `--max_length`: トークン化された配列の最大長
- `--batch_size`: 推論バッチサイズ

#### `predict_onehot_cnn_csv.py`
訓練済みone-hot CNNベースラインモデルで推論を実行。

```bash
python predict_onehot_cnn_csv.py \
  --model best_model.pt \
  --input test.csv \
  --output cnn_predictions.csv \
  --batch_size 256 \
  --device cpu
```

### 評価スクリプト

#### `evaluate_predictions_csv.py`
分類メトリクスを計算。

```bash
python evaluate_predictions_csv.py \
  --gold_file test.csv \
  --pred_file predictions.csv \
  --out_prefix results \
  --gold_label_column label \
  --pred_label_column pred_label \
  --average macro \
  --pos_label 1
```

**出力:**
- `{out_prefix}_metrics.csv`: 正解率、F1、MCC、適合率、再現率
- `{out_prefix}_confusion.png`: 混同行列ヒートマップ

#### `filter_evalres_seq_label.py`
予測結果に基づいてテストデータをフィルタリング。

```bash
python filter_evalres_seq_label.py \
  --input predictions.csv \
  --output filtered_test.csv \
  --pred-value 0
```

**ユースケース:** 詳細分析のための非モチーフ配列の抽出

### 可視化スクリプト

#### `merge_metrics_csvs.py`
複数の実験からメトリクスを集約。

```bash
python merge_metrics_csvs.py \
  --input_glob "eval_summary/**/*_metrics.csv" \
  --recursive \
  --rep_regex "(?:^|/|_)(?:rep|repeat|r)(?P<rep>\\d+)(?:/|_|$)" \
  --out_csv merged_metrics.csv
```

#### `plot_heatmaps_from_merged.py`
論文掲載品質のヒートマップを生成。

```bash
python plot_heatmaps_from_merged.py \
  --input_csv merged_metrics.csv \
  --outdir figures \
  --dpi 500 \
  --formats png,tiff,pdf \
  --agg mean \
  --baseline_model cnn_baseline \
  --draw_baseline_separator \
  --annotate
```

**オプション:**
- `--agg`: レプリケートの集約方法（mean、median）
- `--baseline_model`: ベースライン比較を強調
- `--annotate`: ヒートマップセルにメトリック値を表示

## 使用例

### 例1: RNAオフターゲット予測

```bash
# 1. PiCTUREパイプラインを使用してデータセットを準備（プロトコルセクションA1参照）
# 出力: FD1/dataset_v1_union_40bp_balanced/

# 2. グリッドサーチ用のサブセットを作成
python subset_dataset_csv.py \
  --input_dir FD1/dataset_v1_union_40bp_balanced \
  --output_dir FD1/dataset_subset20 \
  --subset_ratio 0.2 \
  --seed 1 \
  --label_column label

# 3. グリッドサーチを実行
bash grid_search_train_wrapper.sh \
  --train_script train.py \
  --data_path FD1/dataset_subset20 \
  --model_name_or_path zhihan1996/DNABERT-2-117M \
  --output_root fd1_grid_output \
  --run_name_prefix fd1_rnaofftarget \
  --nproc_per_node 1 \
  --batch_configs "8x8,16x8,32x8" \
  --lrs "1e-5,2e-5" \
  --epochs "2" \
  --warmup_ratios "0.05,0.1" \
  --weight_decays "0.01,0.03" \
  --extra_args "--kmer -1 --model_max_length 10 --fp16"

# 4. 最適なハイパーパラメータを選択
python select_best_run.py \
  --output_root fd1_grid_output \
  --metric_name eval_loss \
  --metric_mode min \
  --num_train_epochs 8 \
  --print_full_command

# 5. 最適なハイパーパラメータでファインチューニング（ステップ4のコマンドをコピー）

# 6. テストセットで評価
docker run --platform=linux/amd64 --rm \
  -v $(pwd):/DATA -w /DATA \
  -v $(pwd)/.hf_cache:/root/.cache/huggingface \
  kazukinakamae/dnabert-eval:amd64 \
  python predict_hf_classifier_csv.py \
    --model_dir fd1_output \
    --input_file FD1/dataset_v1_union_40bp_balanced/test.csv \
    --output_file fd1_predictions.csv \
    --text_column sequence \
    --max_length 10 \
    --batch_size 32

# 7. メトリクスを計算
docker run --platform=linux/amd64 --rm \
  -v $(pwd):/DATA -w /DATA \
  kazukinakamae/dnabert-eval:amd64 \
  python evaluate_predictions_csv.py \
    --gold_file FD1/dataset_v1_union_40bp_balanced/test.csv \
    --pred_file fd1_predictions.csv \
    --out_prefix fd1_eval \
    --gold_label_column label \
    --pred_label_column pred_label \
    --pos_label 1
```

### 例2: プロモーター分類

```bash
# 1. EPDnewを使用してデータセットを準備（論文のプロトコルセクションA2参照）

# 2. グリッドサーチを実行
bash grid_search_train_wrapper.sh \
  --train_script train.py \
  --data_path mammal_promoter_classifier/dataset_subset20 \
  --model_name_or_path zhihan1996/DNABERT-2-117M \
  --output_root mammal_promoter_grid \
  --run_name_prefix mammal_promoter \
  --nproc_per_node 1 \
  --batch_configs "8x8,16x8,32x8" \
  --lrs "1e-5,2e-5" \
  --epochs "2" \
  --warmup_ratios "0.05,0.1" \
  --weight_decays "0.01,0.03" \
  --extra_args "--kmer -1 --model_max_length 175 --fp16"

# 3. 比較用のCNNベースラインを訓練
python onehot_cnn_baseline.py \
  --train_csv mammal_promoter_classifier/dataset/train.csv \
  --dev_csv mammal_promoter_classifier/dataset/dev.csv \
  --test_csv mammal_promoter_classifier/dataset/test.csv \
  --outdir cnn_baseline \
  --seq_len 700 \
  --epochs 30

# 4. ヒートマップで結果を比較（両モデルを実行後）
python merge_metrics_csvs.py \
  --input_glob "eval_summary/**/*_metrics.csv" \
  --recursive \
  --out_csv merged_metrics.csv

python plot_heatmaps_from_merged.py \
  --input_csv merged_metrics.csv \
  --outdir figures \
  --baseline_model cnn_baseline \
  --annotate
```

## 引用

研究でDNABERT_2_helperを使用する場合は、以下を引用してください：

```bibtex
@article{nakamae2025dnabert2helper,
  title={Workflow for Fine-tuning and Evaluating DNA Language Models for Specific Genomics Issues},
  author={Nakamae, Kazuki and Bono, Hidemasa},
  journal={Bio-protocol},
  year={2025},
  note={In press}
}
```

また、元のDNABERT-2論文も引用してください：

```bibtex
@article{zhou2024dnabert2,
  title={DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome},
  author={Zhou, Zhihan and Ji, Yanrong and Li, Weijian and Dutta, Pratik and Davuluri, Ramana and Liu, Han},
  journal={arXiv preprint arXiv:2306.15006},
  year={2024}
}
```

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています。詳細は[LICENSE](LICENSE)ファイルをご覧ください。

## 関連リンク

- [公式DNABERT-2リポジトリ](https://github.com/MAGICS-LAB/DNABERT_2)
- [PiCTUREパイプライン](https://github.com/KazukiNakamae/PiCTURE)
- [RNAOffScan](https://github.com/KazukiNakamae/RNAOffScan)
- [EPDnewデータベース](https://epd.expasy.org/epd/)

## 連絡先

質問や問題については：
- GitHubでIssueを開く
- 連絡先: kazuki-nakamae@hiroshima-u.ac.jp

---

**注記**: これは非公式のヘルパーツールキットです。公式のDNABERT-2実装については、[MAGICS-LABリポジトリ](https://github.com/MAGICS-LAB/DNABERT_2)を参照してください。
