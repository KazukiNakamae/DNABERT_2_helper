#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Occlusion attribution (softmax classifier) with sliding-window coverage for long sequences.

Inputs:
  --model_dir : HF save_pretrained directory (local)
  --fasta     : FASTA with nucleotide sequences
  --outdir    : output directory

Outputs (per FASTA record):
  <outdir>/<seq_id>/<seq_id>.<mode>.bed          (bedGraph-like 4 cols, 0-based half-open)
  <outdir>/<seq_id>/<seq_id>.<mode>.heatmap.png
  <outdir>/<seq_id>/<seq_id>.summary.json
  <outdir>/index.html
  <outdir>/run_meta.json

Attribution method:
  - Token-space occlusion in each chunk (span_tokens, stride_tokens)
  - Map token scores -> original char coordinates via offset_mapping
  - Aggregate across overlapping chunks by averaging per base

Modes:
  logit   : z_k (k = target_label or default)
  prob    : softmax(z)_k
  logprob : log_softmax(z)_k
  margin  : (num_labels==2 only) z_pos - z_neg (pos_label is configurable)
  loss    : CrossEntropyLoss(z, y); if --label omitted, uses pseudo-label (see summary)

Notes:
  - Sliding window chunks are created using tokenizer overflow mechanism:
      return_overflowing_tokens=True, truncation=True, max_length=..., stride=chunk_stride_tokens
  - offset_mapping requires a Fast tokenizer. If not fast, script falls back to an approximate mapping,
    but for k-mer/BPE DNA tokenizers you should strongly prefer fast tokenizers.
"""

import argparse
import os
import math
import json
import html
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -----------------------------
# FASTA reader
# -----------------------------
def read_fasta(path: str) -> List[Tuple[str, str, str]]:
    records = []
    seq_id, desc, seq_chunks = None, "", []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    records.append((seq_id, desc, "".join(seq_chunks)))
                header = line[1:].strip()
                if not header:
                    raise ValueError("Empty FASTA header found.")
                parts = header.split(None, 1)
                seq_id = parts[0]
                desc = parts[1] if len(parts) > 1 else ""
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if seq_id is not None:
            records.append((seq_id, desc, "".join(seq_chunks)))
    return records


# -----------------------------
# Offset mapping fallback (non-fast tokenizers)
# -----------------------------
def _clean_token(tok: str) -> str:
    for prefix in ("##", "▁", "Ġ"):
        if tok.startswith(prefix):
            tok = tok[len(prefix):]
    return tok

def approximate_offsets_for_dna(sequence: str, tokens: List[str]) -> List[Tuple[int, int]]:
    seq_upper = sequence.upper()
    pos = 0
    offsets = []
    allowed = set("ACGTN")
    for tok in tokens:
        if tok in ("[CLS]", "[SEP]", "[PAD]", "[UNK]", "<s>", "</s>"):
            offsets.append((0, 0))
            continue
        t = _clean_token(tok).upper()
        t = "".join([c for c in t if c in allowed])
        if not t:
            offsets.append((0, 0))
            continue
        found = seq_upper.find(t, pos)
        if found == -1:
            found = seq_upper.find(t)
        if found == -1:
            offsets.append((0, 0))
            continue
        start = found
        end = found + len(t)
        offsets.append((start, end))
        pos = end
    return offsets


# -----------------------------
# Scoring
# -----------------------------
def score_from_logits(
    logits: torch.Tensor,
    mode: str,
    k: int,
    y: Optional[torch.Tensor],
    pos_label: int,
) -> torch.Tensor:
    """
    logits: [B, C]
    returns: [B]
    """
    if mode == "logit":
        return logits[:, k]
    if mode == "prob":
        return F.softmax(logits, dim=-1)[:, k]
    if mode == "logprob":
        return F.log_softmax(logits, dim=-1)[:, k]
    if mode == "margin":
        if logits.size(1) != 2:
            raise ValueError("margin mode requires num_labels==2.")
        # margin is always defined as pos - neg (independent of k)
        return logits[:, pos_label] - logits[:, 1 - pos_label]
    if mode == "loss":
        if y is None:
            raise ValueError("loss mode requires y.")
        return F.cross_entropy(logits, y, reduction="none")
    raise ValueError(mode)


@dataclass
class ChunkInfo:
    input_ids: torch.Tensor        # [T]
    attention_mask: torch.Tensor   # [T]
    offsets: List[Tuple[int, int]] # length T, offsets w.r.t original string (fast tokenizer) or approximated
    chunk_index: int


# -----------------------------
# Sliding-window chunking
# -----------------------------
def make_chunks_with_overflow(
    tokenizer,
    sequence: str,
    max_length: int,
    chunk_stride_tokens: int,
) -> Tuple[List[ChunkInfo], Dict]:
    """
    Returns a list of chunks. Offsets are in original string coordinate if fast tokenizer.
    """
    meta = {
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "max_length": max_length,
        "chunk_stride_tokens": chunk_stride_tokens,
    }

    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(
            sequence,
            return_tensors=None,
            truncation=True,
            max_length=max_length,
            stride=chunk_stride_tokens,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        # For a single sequence, enc fields are lists of chunks
        input_ids_list = enc["input_ids"]
        attn_list = enc.get("attention_mask", [None] * len(input_ids_list))
        offsets_list = enc["offset_mapping"]

        chunks: List[ChunkInfo] = []
        for i, (ids, attn, offs) in enumerate(zip(input_ids_list, attn_list, offsets_list)):
            chunks.append(
                ChunkInfo(
                    input_ids=torch.tensor(ids, dtype=torch.long),
                    attention_mask=torch.tensor(attn, dtype=torch.long) if attn is not None else None,
                    offsets=[tuple(x) for x in offs],
                    chunk_index=i,
                )
            )
        meta["num_chunks"] = len(chunks)
        return chunks, meta

    # Non-fast tokenizer fallback: single chunk (will truncate); we keep functionality but warn in report.
    enc = tokenizer(sequence, return_tensors=None, truncation=True, max_length=max_length, padding=False)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    offsets = approximate_offsets_for_dna(sequence, tokens)
    chunks = [
        ChunkInfo(
            input_ids=torch.tensor(enc["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(enc["attention_mask"], dtype=torch.long) if "attention_mask" in enc else None,
            offsets=offsets,
            chunk_index=0,
        )
    ]
    meta["num_chunks"] = 1
    meta["warning"] = "Tokenizer is not fast; overflow chunks and true offsets are unavailable. Results may be incomplete for long sequences."
    return chunks, meta


# -----------------------------
# Occlusion per chunk -> char aggregation
# -----------------------------
def occlusion_chunk_to_char_scores(
    model,
    chunk: ChunkInfo,
    device: str,
    modes: List[str],
    span_tokens: int,
    stride_tokens: int,
    k: int,
    y_loss: Optional[int],
    pos_label: int,
    baseline_id: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Returns:
      - char_scores_by_mode for this chunk (in original string coordinates via offsets; but returned as sparse add-to-global arrays later)
        Here we return token_scores only; mapping to chars is done outside to allow global aggregation.
      - base_scores_by_mode
    """
    input_ids = chunk.input_ids.to(device).unsqueeze(0)          # [1, T]
    attn = chunk.attention_mask.to(device).unsqueeze(0) if chunk.attention_mask is not None else None

    with torch.no_grad():
        logits_base = model(input_ids=input_ids, attention_mask=attn).logits  # [1, C]

    y_tensor = None
    if "loss" in modes:
        if y_loss is None:
            raise ValueError("loss requested but y_loss is None.")
        y_tensor = torch.tensor([y_loss], dtype=torch.long, device=device)

    base_scores = {}
    for m in modes:
        if m == "margin" and logits_base.size(-1) != 2:
            continue
        if m == "loss":
            s = score_from_logits(logits_base, m, k=k, y=y_tensor, pos_label=pos_label)
        else:
            s = score_from_logits(logits_base, m, k=k, y=None, pos_label=pos_label)
        base_scores[m] = float(s.item())

    T = input_ids.size(1)
    deltas = {m: np.zeros(T, dtype=np.float64) for m in modes}
    counts = np.zeros(T, dtype=np.float64)

    start = 0
    while start < T:
        end = min(start + span_tokens, T)
        occ_ids = input_ids.clone()
        occ_ids[0, start:end] = baseline_id

        with torch.no_grad():
            logits_occ = model(input_ids=occ_ids, attention_mask=attn).logits

        for m in modes:
            if m == "margin" and logits_occ.size(-1) != 2:
                continue
            if m == "loss":
                base = base_scores[m]
                occ = float(score_from_logits(logits_occ, m, k=k, y=y_tensor, pos_label=pos_label).item())
                delta = occ - base  # loss: increase => supportive
            else:
                base = base_scores[m]
                occ = float(score_from_logits(logits_occ, m, k=k, y=None, pos_label=pos_label).item())
                delta = base - occ  # others: decrease => supportive
            deltas[m][start:end] += delta

        counts[start:end] += 1.0
        if end == T:
            break
        start += stride_tokens

    token_scores_by_mode = {}
    for m in modes:
        if m == "margin" and logits_base.size(-1) != 2:
            continue
        token_scores_by_mode[m] = deltas[m] / np.maximum(counts, 1e-12)

    return token_scores_by_mode, base_scores


def aggregate_chunks_to_global_char_scores(
    sequence: str,
    chunks: List[ChunkInfo],
    model,
    tokenizer,
    device: str,
    modes: List[str],
    span_tokens: int,
    stride_tokens: int,
    target_label: Optional[int],
    pos_label: int,
    label_for_loss: Optional[int],
) -> Tuple[Dict[str, np.ndarray], Dict, Dict[str, float]]:
    """
    Returns:
      global_char_scores_by_mode: dict mode -> [L]
      meta: details (target class, predicted, pseudo label info, chunks, etc.)
      global_base_scores: base score per mode for a representative (whole-seq summary; we use chunk0 base by default)
    """
    L = len(sequence)
    num_labels = int(model.config.num_labels)

    # Determine baseline token id for occlusion
    baseline_id = tokenizer.mask_token_id
    if baseline_id is None:
        baseline_id = tokenizer.pad_token_id
    if baseline_id is None:
        baseline_id = tokenizer.unk_token_id
    if baseline_id is None:
        raise RuntimeError("Could not determine baseline token id (mask/pad/unk).")

    # First pass: compute logits/preds per chunk for reporting and for default target label decision
    chunk_preds = []
    chunk_logits_sum = None
    for ch in chunks:
        ids = ch.input_ids.to(device).unsqueeze(0)
        attn = ch.attention_mask.to(device).unsqueeze(0) if ch.attention_mask is not None else None
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=attn).logits  # [1, C]
        pred = int(torch.argmax(F.softmax(logits, dim=-1), dim=-1).item())
        chunk_preds.append(pred)
        chunk_logits_sum = logits.detach().cpu()[0] if chunk_logits_sum is None else (chunk_logits_sum + logits.detach().cpu()[0])

    # Decide target class k:
    # - If user provided target_label: use it
    # - Else if binary: default to pos_label (most typical for promoter=1)
    # - Else: use argmax of summed logits across chunks (rough global decision)
    if target_label is not None:
        k = int(target_label)
    else:
        if num_labels == 2:
            k = int(pos_label)
        else:
            k = int(torch.argmax(chunk_logits_sum).item())

    if k < 0 or k >= num_labels:
        raise ValueError(f"target_label k={k} out of range for num_labels={num_labels}")

    # Label for loss:
    # - If user provided label: use it
    # - Else: use k as pseudo-label (documented)
    y_loss = int(label_for_loss) if label_for_loss is not None else int(k)

    global_char_scores = {m: np.zeros(L, dtype=np.float64) for m in modes}
    global_char_counts = {m: np.zeros(L, dtype=np.float64) for m in modes}

    global_base_scores = {}  # we will set from the first chunk's base scores (for report)
    for idx, ch in enumerate(chunks):
        token_scores_by_mode, base_scores = occlusion_chunk_to_char_scores(
            model=model,
            chunk=ch,
            device=device,
            modes=modes,
            span_tokens=span_tokens,
            stride_tokens=stride_tokens,
            k=k,
            y_loss=y_loss,
            pos_label=pos_label,
            baseline_id=baseline_id,
        )
        if idx == 0:
            global_base_scores = base_scores

        # Map token scores -> original char coords via offsets
        for m, token_scores in token_scores_by_mode.items():
            for (s, e), w in zip(ch.offsets, token_scores.tolist()):
                if s is None or e is None or s == e:
                    continue
                s = int(max(0, min(L, s)))
                e = int(max(0, min(L, e)))
                if s >= e:
                    continue
                global_char_scores[m][s:e] += float(w)
                global_char_counts[m][s:e] += 1.0

    # Finalize by averaging where covered
    finalized = {}
    for m in modes:
        if m == "margin" and num_labels != 2:
            continue
        denom = np.maximum(global_char_counts[m], 1.0)
        finalized[m] = global_char_scores[m] / denom

    meta = {
        "sequence_length_bp": L,
        "num_labels": num_labels,
        "target_class_k": k,
        "pos_label": pos_label,
        "loss_label_y": y_loss,
        "chunk_preds": chunk_preds,
        "num_chunks": len(chunks),
        "baseline_token_id": int(baseline_id),
        "note_target_selection": (
            "k is user-specified via --target_label; "
            "if omitted: binary -> pos_label, else -> argmax of summed chunk logits."
        ),
        "note_loss_label": (
            "Loss uses --label if provided; otherwise uses k as pseudo-label."
        ),
    }
    return finalized, meta, global_base_scores


# -----------------------------
# BED (bedGraph-like)
# -----------------------------
def write_bedgraph_like(path: str, chrom: str, values: np.ndarray, precision: int = 6):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    q = np.round(values.astype(np.float64), precision)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# chrom start end score (0-based, half-open)\n")
        if len(q) == 0:
            return
        start = 0
        cur = q[0]
        for i in range(1, len(q) + 1):
            if i == len(q) or q[i] != cur:
                end = i
                if cur != 0.0:
                    f.write(f"{chrom}\t{start}\t{end}\t{cur}\n")
                if i < len(q):
                    start = i
                    cur = q[i]


# -----------------------------
# Heatmap plotting
# -----------------------------
def save_heatmap_png(path: str, values: np.ndarray, cols: int = 200, title: str = ""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    L = len(values)
    if L == 0:
        return
    rows = int(math.ceil(L / cols))
    mat = np.full((rows, cols), np.nan, dtype=np.float64)
    for r in range(rows):
        s = r * cols
        e = min((r + 1) * cols, L)
        mat[r, 0:(e - s)] = values[s:e]

    plt.figure(figsize=(max(8, cols / 25), max(2, rows / 3)))
    plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(label="attribution (delta)")
    if title:
        plt.title(title)
    plt.yticks(range(rows), [f"{r*cols}-{min((r+1)*cols, L)}" for r in range(rows)])
    plt.xlabel("position within row (bp)")
    plt.ylabel("bp window")
    plt.tight_layout()
    plt.savefig(path, dpi=600)
    plt.close()


# -----------------------------
# HTML report
# -----------------------------
def top_regions_from_scores(scores: np.ndarray, window_bp: int = 25, topk: int = 10):
    L = len(scores)
    if L == 0:
        return []
    window_bp = max(1, min(window_bp, L))
    cumsum = np.concatenate([[0.0], np.cumsum(scores)])
    means = []
    for s in range(0, L - window_bp + 1):
        e = s + window_bp
        m = (cumsum[e] - cumsum[s]) / window_bp
        means.append((s, e, float(m)))
    means.sort(key=lambda x: x[2], reverse=True)
    picked = []
    for s, e, m in means:
        if len(picked) >= topk:
            break
        if all(not (s < pe and e > ps) for ps, pe, _ in picked):
            picked.append((s, e, m))
    return picked if picked else means[:topk]

def render_html_report(outdir: str, run_meta: Dict, per_seq: List[Dict]):
    path = os.path.join(outdir, "index.html")
    css = """
    body { font-family: Arial, sans-serif; margin: 24px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 16px 0; }
    code { background: #f5f5f5; padding: 2px 6px; border-radius: 6px; }
    .small { color: #555; font-size: 0.95em; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }
    img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 8px; }
    ul { margin-top: 6px; }
    """
    def esc(x): return html.escape(str(x))

    lines = []
    lines.append("<!doctype html><html><head><meta charset='utf-8'>")
    lines.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    lines.append(f"<style>{css}</style></head><body>")
    lines.append("<h1>Occlusion Attribution Report (Sliding Window)</h1>")

    lines.append("<div class='card'>")
    lines.append("<h2>Run metadata</h2>")
    lines.append("<pre>" + esc(json.dumps(run_meta, indent=2, ensure_ascii=False)) + "</pre>")
    lines.append("<p class='small'>BED outputs use 0-based, half-open coordinates (bedGraph-like 4 columns).</p>")
    lines.append("</div>")

    for item in per_seq:
        lines.append("<div class='card'>")
        lines.append(f"<h2>{esc(item['seq_id'])}</h2>")
        if item.get("description"):
            lines.append(f"<div class='small'>{esc(item['description'])}</div>")

        lines.append("<ul>")
        lines.append(f"<li>Length: <code>{esc(item['length_bp'])}</code> bp</li>")
        lines.append(f"<li>num_chunks: <code>{esc(item['chunking']['num_chunks'])}</code></li>")
        lines.append(f"<li>Target class k: <code>{esc(item['attrib_meta']['target_class_k'])}</code></li>")
        lines.append(f"<li>Loss label y: <code>{esc(item['attrib_meta']['loss_label_y'])}</code></li>")
        lines.append("</ul>")

        lines.append("<details><summary>Chunking / Attribution metadata</summary>")
        lines.append("<pre>" + esc(json.dumps(item["chunking"], indent=2, ensure_ascii=False)) + "</pre>")
        lines.append("<pre>" + esc(json.dumps(item["attrib_meta"], indent=2, ensure_ascii=False)) + "</pre>")
        lines.append("</details>")

        lines.append("<details><summary>Base scores (from chunk 0)</summary>")
        lines.append("<pre>" + esc(json.dumps(item["base_scores_chunk0"], indent=2, ensure_ascii=False)) + "</pre>")
        lines.append("</details>")

        lines.append("<div class='grid'>")
        for mode in item["modes"]:
            bed_rel = item["artifacts"][mode]["bed_rel"]
            png_rel = item["artifacts"][mode]["png_rel"]
            top_regions = item["artifacts"][mode]["top_regions"]

            lines.append("<div>")
            lines.append(f"<h3>Mode: {esc(mode)}</h3>")
            lines.append(f"<div class='small'>BED: <a href='{esc(bed_rel)}'>{esc(os.path.basename(bed_rel))}</a></div>")
            lines.append(f"<img src='{esc(png_rel)}' alt='heatmap {esc(mode)}'>")
            lines.append("<div class='small'>Top regions (bp, 0-based):</div>")
            lines.append("<ul>")
            for (s, e, v) in top_regions:
                lines.append(f"<li><code>{s}-{e}</code> avg={esc(v)}</li>")
            lines.append("</ul>")
            lines.append("</div>")

        lines.append("</div>")  # grid
        lines.append("</div>")  # card

    lines.append("</body></html>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--modes", default="logit,prob,logprob,margin,loss")

    # Occlusion params (within each chunk)
    ap.add_argument("--span_tokens", type=int, default=5)
    ap.add_argument("--stride_tokens", type=int, default=1)

    # Sliding window chunking params (tokenizer overflow)
    ap.add_argument("--max_length", type=int, default=512, help="Tokenizer max_length per chunk (tokens).")
    ap.add_argument("--chunk_stride_tokens", type=int, default=128, help="Overlap (tokens) between chunks.")

    # Target/label control
    ap.add_argument("--target_label", type=int, default=None,
                    help="Target class k for logit/prob/logprob. If omitted: binary->pos_label; else -> argmax of summed chunk logits.")
    ap.add_argument("--label", type=int, default=None,
                    help="Label y for loss mode. If omitted, uses k as pseudo-label.")
    ap.add_argument("--pos_label", type=int, default=1,
                    help="For margin (num_labels==2): margin = z[pos] - z[neg]. Also used as default k in binary if --target_label omitted.")

    # Visualization
    ap.add_argument("--heatmap_cols", type=int, default=200)
    ap.add_argument("--top_window_bp", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True).to(device).eval()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid = {"logit", "prob", "logprob", "margin", "loss"}
    for m in modes:
        if m not in valid:
            raise ValueError(f"Invalid mode: {m}")

    records = read_fasta(args.fasta)
    if not records:
        raise ValueError("No FASTA records found.")

    run_meta = {
        "model_dir": os.path.abspath(args.model_dir),
        "fasta": os.path.abspath(args.fasta),
        "outdir": os.path.abspath(args.outdir),
        "device": device,
        "modes_requested": modes,
        "span_tokens": args.span_tokens,
        "stride_tokens": args.stride_tokens,
        "max_length": args.max_length,
        "chunk_stride_tokens": args.chunk_stride_tokens,
        "target_label": args.target_label,
        "label_for_loss": args.label,
        "pos_label": args.pos_label,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
    }

    per_seq_items = []

    for seq_id, desc, seq in records:
        seq = seq.strip()
        if not seq:
            continue

        seq_out = os.path.join(args.outdir, seq_id)
        os.makedirs(seq_out, exist_ok=True)

        chunks, chunk_meta = make_chunks_with_overflow(
            tokenizer=tokenizer,
            sequence=seq,
            max_length=args.max_length,
            chunk_stride_tokens=args.chunk_stride_tokens,
        )

        # Aggregate occlusion scores across all chunks to global coordinates
        global_scores_by_mode, attrib_meta, base_scores_chunk0 = aggregate_chunks_to_global_char_scores(
            sequence=seq,
            chunks=chunks,
            model=model,
            tokenizer=tokenizer,
            device=device,
            modes=modes,
            span_tokens=args.span_tokens,
            stride_tokens=args.stride_tokens,
            target_label=args.target_label,
            pos_label=args.pos_label,
            label_for_loss=args.label,
        )

        artifacts = {}
        modes_done = []
        for mode, scores in global_scores_by_mode.items():
            bed_path = os.path.join(seq_out, f"{seq_id}.{mode}.bed")
            png_path = os.path.join(seq_out, f"{seq_id}.{mode}.heatmap.png")

            write_bedgraph_like(bed_path, chrom=seq_id, values=scores, precision=6)
            save_heatmap_png(
                png_path,
                values=scores,
                cols=args.heatmap_cols,
                title=f"{seq_id} | {mode} | chunks={attrib_meta['num_chunks']} | occ(span={args.span_tokens}, stride={args.stride_tokens})",
            )

            top_regions = top_regions_from_scores(scores, window_bp=args.top_window_bp, topk=10)

            artifacts[mode] = {
                "bed_rel": f"{seq_id}/{os.path.basename(bed_path)}",
                "png_rel": f"{seq_id}/{os.path.basename(png_path)}",
                "top_regions": top_regions,
            }
            modes_done.append(mode)

        seq_summary = {
            "seq_id": seq_id,
            "description": desc,
            "length_bp": len(seq),
            "modes": modes_done,
            "chunking": chunk_meta,
            "attrib_meta": attrib_meta,
            "base_scores_chunk0": base_scores_chunk0,
            "artifacts": artifacts,
        }
        with open(os.path.join(seq_out, f"{seq_id}.summary.json"), "w", encoding="utf-8") as f:
            json.dump(seq_summary, f, indent=2, ensure_ascii=False)

        per_seq_items.append(seq_summary)

    render_html_report(args.outdir, run_meta, per_seq_items)

    with open(os.path.join(args.outdir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"run_meta": run_meta, "sequences": [x["seq_id"] for x in per_seq_items]}, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote report: {os.path.join(args.outdir, 'index.html')}")


if __name__ == "__main__":
    main()
