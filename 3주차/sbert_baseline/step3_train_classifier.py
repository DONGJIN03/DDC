"""
STEP3 - STEP2 캐시(.npy)로 "얼린 SBERT + Linear(768->9)" 분류기 학습/평가.

STEP2와 같은 컴퓨터(원격 GPU 서버)에서 이어서 실행하는 걸 전제로 한다 - .npy를
로컬로 옮기지 않고 그대로 사용. 분류기 자체는 가벼워서 CPU로도 충분하지만,
GPU가 있으면 자동으로 사용한다.

실행:
    python step3_train_classifier.py

전제: sbert_cache/<name>_train.npy, <name>_val.npy 가 이미 있어야 함 (STEP2 산출물).

산출물:
  - train_results/sbert_<name>.json   (1주차 step5_*.json / 2주차 stepD_*.json과 동일 스키마)
  - train_results/sbert_raw_stats.json 의 "train" 섹션 추가
    (학습 시간, 사용한 lr/epoch/batch_size/optimizer/seed)
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import common
from common import (
    CACHE_DIR, RESULTS_DIR, RAW_STATS_PATH, TARGET_SYMPTOMS, SBERT_MODELS,
    THRESHOLD, CLASSIFIER_LR, CLASSIFIER_EPOCHS, CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_OPTIMIZER, SEED,
)


def _load_raw_stats():
    if RAW_STATS_PATH.exists():
        with open(RAW_STATS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_raw_stats(stats):
    with open(RAW_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def _write_result(name, result):
    path = RESULTS_DIR / f"sbert_{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  result saved -> {path}", flush=True)


def compute_metrics(labels, preds):
    """1주차 train_utils.py의 compute_metrics와 완전히 동일한 계산 방식."""
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_label_f1 = f1_score(labels, preds, average=None, zero_division=0)
    metrics = {"macro_f1": float(macro_f1)}
    for sym, f1 in zip(TARGET_SYMPTOMS, per_label_f1):
        metrics[f"f1_{sym}"] = float(f1)
    return metrics


def train_one(name, hf_id, device, y_train, y_val, stats):
    print(f"=== STEP3: {name} ({hf_id}) 분류기 학습 ===", flush=True)

    result = {
        "name": name, "hf_id": hf_id, "status": "OK", "error": "",
        "train_time_s": None, "val_macro_f1": None,
    }
    for s in TARGET_SYMPTOMS:
        result[f"f1_{s}"] = None

    common.set_seed(SEED)
    t0 = time.time()
    try:
        X_train = np.load(CACHE_DIR / f"{name}_train.npy")
        X_val = np.load(CACHE_DIR / f"{name}_val.npy")

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

        n_features = X_train_t.shape[1]
        model = nn.Linear(n_features, len(TARGET_SYMPTOMS)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=CLASSIFIER_LR)
        loss_fn = nn.BCEWithLogitsLoss()

        n = X_train_t.shape[0]
        for epoch in range(CLASSIFIER_EPOCHS):
            perm = torch.randperm(n, device=device)
            epoch_loss = 0.0
            for start in range(0, n, CLASSIFIER_BATCH_SIZE):
                idx = perm[start:start + CLASSIFIER_BATCH_SIZE]
                xb, yb = X_train_t[idx], y_train_t[idx]

                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    epoch {epoch + 1}/{CLASSIFIER_EPOCHS} loss={epoch_loss / n:.4f}", flush=True)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_preds = (val_probs >= THRESHOLD).astype(int)

    except Exception as e:
        elapsed = time.time() - t0
        result["status"] = "FAILED"
        result["error"] = f"{type(e).__name__}: {e}"
        result["train_time_s"] = round(elapsed, 1)
        _write_result(name, result)
        print(f"  FAILED: {result['error']}", flush=True)
        return result

    elapsed = time.time() - t0
    metrics = compute_metrics(y_val, val_preds)

    result["train_time_s"] = round(elapsed, 1)
    result["val_macro_f1"] = metrics["macro_f1"]
    for s in TARGET_SYMPTOMS:
        result[f"f1_{s}"] = metrics[f"f1_{s}"]

    _write_result(name, result)
    print(f"  OK. train_time={result['train_time_s']}s val_macro_f1={result['val_macro_f1']:.4f}", flush=True)

    stats.setdefault(name, {})["train_time_s"] = round(elapsed, 1)
    stats[name]["classifier_lr"] = CLASSIFIER_LR
    stats[name]["classifier_epochs"] = CLASSIFIER_EPOCHS
    stats[name]["classifier_batch_size"] = CLASSIFIER_BATCH_SIZE
    stats[name]["classifier_optimizer"] = CLASSIFIER_OPTIMIZER
    stats[name]["seed"] = SEED
    stats[name]["threshold"] = THRESHOLD

    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== STEP3: 분류기 학습 (device={device}) ===", flush=True)

    train_df = common.load_split_csv(common.TRAIN_CSV)
    val_df = common.load_split_csv(common.VAL_CSV)
    _, y_train = common.get_texts_and_labels(train_df)
    _, y_val = common.get_texts_and_labels(val_df)

    stats = _load_raw_stats()

    all_results = {}
    for name, hf_id in SBERT_MODELS.items():
        train_npy = CACHE_DIR / f"{name}_train.npy"
        val_npy = CACHE_DIR / f"{name}_val.npy"
        if not train_npy.exists() or not val_npy.exists():
            print(f"=== {name}: 캐시({train_npy.name}/{val_npy.name}) 없음, STEP2 먼저 실행 필요 - 건너뜀 ===", flush=True)
            continue

        result = train_one(name, hf_id, device, y_train, y_val, stats)
        all_results[name] = result
        _save_raw_stats(stats)

    print("=== STEP3 완료 ===", flush=True)
    for name, r in all_results.items():
        print(f"  {name}: val_macro_f1={r.get('val_macro_f1')}", flush=True)


if __name__ == "__main__":
    main()
