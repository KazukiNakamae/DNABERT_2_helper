# DNABERT_2_helper

[日本語版はこちら / Japanese version](README_JP.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An unofficial collection of helper scripts for fine-tuning, evaluating, and deploying [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2) models.

## Overview

DNABERT_2_helper provides a set of utility scripts to facilitate the practical use of DNABERT-2 for genomic sequence classification tasks. This toolkit was developed as part of the protocol described in **"Workflow for Fine-tuning and Evaluating DNA Language Models for Specific Genomics Issues"** (Nakamae & Bono, Bio-protocol, 2025).

## Features

- 🔍 **Environment Health Checks**: Verify GPU/CUDA and PyTorch configurations
- 🔧 **Hyperparameter Optimization**: Grid search wrapper for efficient hyperparameter tuning
- 📊 **Dataset Preparation**: Subset creation and label balancing utilities
- 🚀 **CPU-friendly Inference**: Docker-based evaluation environment without GPU requirements
- 📈 **Comprehensive Evaluation**: Compute accuracy, F1, MCC, precision, and recall metrics
- 🎨 **Visualization**: Generate publication-ready heatmaps comparing model performance

## Requirements

### For Fine-tuning (GPU Environment)
- NVIDIA GPU with CUDA support (tested on H100 80GB)
- CUDA Toolkit 12.1+
- Python 3.8+
- 120GB+ RAM (recommended)
- Ubuntu 22.04 LTS or similar

### For Inference (CPU Environment)
- Docker
- 16GB+ RAM (recommended)
- macOS or Linux

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/KazukiNakamae/DNABERT_2_helper.git
cd DNABERT_2_helper
```

### 2. GPU Environment Setup (for fine-tuning)

Follow the detailed setup instructions in Section B1 of the protocol paper. Key steps include:

```bash
# Create conda environment
conda create -n dna conda-forge::python=3.8
conda activate dna

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional dependencies
pip3 install -r modified_requirements.txt

# Run health checks
bash gpu_cuda_health_check.sh
python torch_health_check.py
```

### 3. CPU Environment Setup (for inference)

```bash
# Build Docker image for CPU-based evaluation
docker buildx build --platform linux/amd64 \
  --build-arg ENABLE_CUDA=0 \
  -t kazukinakamae/dnabert-eval:amd64 \
  -f Dockerfile .

# Create cache directory
mkdir -p .hf_cache
```

## Quick Start

### 1. Prepare Your Dataset

Ensure your dataset has the following CSV format:
- `train.csv`, `dev.csv`, `test.csv`
- Required columns: `sequence` (DNA sequence), `label` (0 or 1)

### 2. Hyperparameter Grid Search

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

### 3. Select Best Hyperparameters

```bash
python select_best_run.py \
  --output_root /path/to/grid_search_output \
  --metric_name eval_loss \
  --metric_mode min \
  --num_train_epochs 8 \
  --print_full_command
```

### 4. Fine-tuning

Please refer to [Official repository of DNABERT-2 provided by Northwestern University MAGICS Labs](https://github.com/MAGICS-LAB/DNABERT_2).

### 5. Run Inference

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

### 6. Evaluate Performance

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

### 7. Visualize Results

```bash
# Merge metrics from multiple experiments
docker run --platform=linux/amd64 --rm \
  -v $(pwd):/DATA -w /DATA \
  kazukinakamae/dnabert-eval:amd64 \
  python merge_metrics_csvs.py \
    --input_glob "eval_summary/**/*_metrics.csv" \
    --recursive \
    --out_csv merged_metrics.csv

# Generate heatmaps
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

## Scripts Reference

### Health Check Scripts

#### `gpu_cuda_health_check.sh`
Verify GPU and CUDA installation at the OS level.

```bash
bash gpu_cuda_health_check.sh
```

#### `torch_health_check.py`
Check PyTorch and GPU integration.

```bash
python torch_health_check.py
```

### Data Preparation Scripts

#### `subset_dataset_csv.py`
Create a subset of dataset for grid search or quick testing.

```bash
python subset_dataset_csv.py \
  --input_dir /path/to/dataset \
  --output_dir /path/to/subset \
  --subset_ratio 0.2 \
  --seed 1 \
  --label_column label
```

**Options:**
- `--input_dir`: Directory containing train.csv, dev.csv, test.csv
- `--output_dir`: Output directory for subset
- `--subset_ratio`: Proportion of data to sample (0.0-1.0)
- `--seed`: Random seed for reproducibility
- `--label_column`: Name of the label column

### Training Scripts

#### `grid_search_train_wrapper.sh`
Automate hyperparameter grid search for DNABERT-2 fine-tuning.

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

**Key Options:**
- `--batch_configs`: Format "batch_size x accumulation_steps"
- `--lrs`: Comma-separated learning rates
- `--epochs`: Number of training epochs
- `--warmup_ratios`: Warmup ratio or step count
- `--weight_decays`: Weight decays strength

#### `select_best_run.py`
Identify the best hyperparameter combination from grid search results.

```bash
python select_best_run.py \
  --output_root /path/to/grid_output \
  --metric_name eval_loss \
  --metric_mode min \
  --num_train_epochs 8 \
  --print_full_command
```

**Options:**
- `--metric_name`: Metric to optimize (eval_loss, accuracy, f1, etc.)
- `--metric_mode`: "min" or "max"
- `--print_full_command`: Generate ready-to-run training command

#### `resume_same_task_same_data.py`
Resume fine-tuning from a checkpoint.

```bash
python resume_same_task_same_data.py \
  --checkpoint_dir /path/to/checkpoint \
  --data_path /path/to/dataset \
  --output_dir /path/to/output \
  --add_epochs 4 \
  --do_test_eval
```

### Baseline Scripts

#### `onehot_cnn_baseline.py`
Train a 3-layer one-hot CNN baseline model.

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
Generate motif-based baseline predictions.

```bash
python predict_motif_baseline_csv.py \
  --input test.csv \
  --output motif_predictions.csv \
  --pattern WCW \
  --start 19
```

**Options:**
- `--pattern`: Motif pattern (e.g., WCW, ACW)
- `--start`: 0-based start position for motif matching

### Inference Scripts

#### `predict_hf_classifier_csv.py`
Generate predictions using a fine-tuned DNABERT-2 model.

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

**Options:**
- `--model_dir`: Path to fine-tuned model directory
- `--text_column`: Column name containing sequences
- `--max_length`: Maximum tokenized sequence length
- `--batch_size`: Inference batch size

#### `predict_onehot_cnn_csv.py`
Run inference with a trained one-hot CNN baseline model.

```bash
python predict_onehot_cnn_csv.py \
  --model best_model.pt \
  --input test.csv \
  --output cnn_predictions.csv \
  --batch_size 256 \
  --device cpu
```

### Evaluation Scripts

#### `evaluate_predictions_csv.py`
Compute classification metrics.

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

**Outputs:**
- `{out_prefix}_metrics.csv`: Accuracy, F1, MCC, precision, recall
- `{out_prefix}_confusion.png`: Confusion matrix heatmap

#### `filter_evalres_seq_label.py`
Filter test data based on prediction results.

```bash
python filter_evalres_seq_label.py \
  --input predictions.csv \
  --output filtered_test.csv \
  --pred-value 0
```

**Use case:** Extract non-motif sequences for detailed analysis

### Visualization Scripts

#### `merge_metrics_csvs.py`
Aggregate metrics from multiple experiments.

```bash
python merge_metrics_csvs.py \
  --input_glob "eval_summary/**/*_metrics.csv" \
  --recursive \
  --rep_regex "(?:^|/|_)(?:rep|repeat|r)(?P<rep>\\d+)(?:/|_|$)" \
  --out_csv merged_metrics.csv
```

#### `plot_heatmaps_from_merged.py`
Generate publication-quality heatmaps.

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

**Options:**
- `--agg`: Aggregation method for replicates (mean, median)
- `--baseline_model`: Highlight baseline comparisons
- `--annotate`: Show metric values on heatmap cells

## Examples

### Example 1: RNA Off-target Prediction

```bash
# 1. Prepare dataset using PiCTURE pipeline (see protocol Section A1)
# Output: FD1/dataset_v1_union_40bp_balanced/

# 2. Create subset for grid search
python subset_dataset_csv.py \
  --input_dir FD1/dataset_v1_union_40bp_balanced \
  --output_dir FD1/dataset_subset20 \
  --subset_ratio 0.2 \
  --seed 1 \
  --label_column label

# 3. Run grid search
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

# 4. Select best hyperparameters
python select_best_run.py \
  --output_root fd1_grid_output \
  --metric_name eval_loss \
  --metric_mode min \
  --num_train_epochs 8 \
  --print_full_command

# 5. Fine-tune with optimal hyperparameters (copy command from step 4)

# 6. Evaluate on test set
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

# 7. Compute metrics
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

### Example 2: Promoter Classification

```bash
# 1. Prepare dataset using EPDnew (see protocol Section A2)

# 2. Run grid search
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

# 3. Train CNN baseline for comparison
python onehot_cnn_baseline.py \
  --train_csv mammal_promoter_classifier/dataset/train.csv \
  --dev_csv mammal_promoter_classifier/dataset/dev.csv \
  --test_csv mammal_promoter_classifier/dataset/test.csv \
  --outdir cnn_baseline \
  --seq_len 700 \
  --epochs 30

# 4. Compare results with heatmaps (after running both models)
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

## Citation

If you use DNABERT_2_helper in your research, please cite:

```bibtex
@article{nakamae2025dnabert2helper,
  title={Workflow for Fine-tuning and Evaluating DNA Language Models for Specific Genomics Issues},
  author={Nakamae, Kazuki and Bono, Hidemasa},
  journal={Bio-protocol},
  year={2025},
  note={In press}
}
```

Additionally, please cite the original DNABERT-2 paper:

```bibtex
@article{zhou2024dnabert2,
  title={DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome},
  author={Zhou, Zhihan and Ji, Yanrong and Li, Weijian and Dutta, Pratik and Davuluri, Ramana and Liu, Han},
  journal={arXiv preprint arXiv:2306.15006},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related Links

- [Official DNABERT-2 Repository](https://github.com/MAGICS-LAB/DNABERT_2)
- [PiCTURE Pipeline](https://github.com/KazukiNakamae/PiCTURE)
- [RNAOffScan](https://github.com/KazukiNakamae/RNAOffScan)
- [EPDnew Database](https://epd.expasy.org/epd/)

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: kazuki-nakamae@hiroshima-u.ac.jp

---

**Note**: This is an unofficial helper toolkit. For the official DNABERT-2 implementation, please refer to the [MAGICS-LAB repository](https://github.com/MAGICS-LAB/DNABERT_2).
