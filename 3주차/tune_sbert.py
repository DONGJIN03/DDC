"""
SBERT(ko-sroberta) 파라미터 튜닝.

baseline(max_length=256, pooling=mean, lr=1e-3, head=Linear)에서 그룹별로
변인 하나씩만 바꿔서 비교한다 (변인 통제):
  - max_length: 384, 512          -> 재인코딩 필요 (GPU)
  - pooling:    max, cls          -> 재인코딩 필요 (같은 모델 가중치 재사용, pooling만 교체)
  - lr:         5e-4, 5e-3        -> baseline 캐시 그대로, 분류기만 재학습 (CPU도 충분)
  - head:       mlp(768->256->9)  -> baseline 캐시 그대로, 분류기만 재학습

전부 끝나면 그룹별 최선값을 자동으로 골라 "best_combo" 1개를 추가로 실행하고,
baseline + 변인 8개 + best_combo까지 전부 하나의 리스트로 저장한다.

실행: python tune_sbert.py   (원격 GPU 권장 - 재인코딩 구간 때문)
이미 끝난 config_id는 건너뛰므로 중단 후 재실행 가능.
산출물: tuning_results/tune_sbert_results.json
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score

import common_tuning as ct
from common_tuning import base_common, TARGET_SYMPTOMS

RESULT_PATH = ct.TUNING_RESULTS_DIR / "tune_sbert_results.json"


def _load_existing():
    if RESULT_PATH.exists():
        with open(RESULT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save(results):
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _set_pooling_mode(model, mode):
    """SentenceTransformer의 Pooling 모듈(model[1])은 mean/max/cls 각각을
    boolean 플래그로 갖고 있음 - forward()가 이 플래그들을 직접 참조하므로
    이렇게 직접 설정하면 모델을 다시 받을 필요 없이 pooling만 바뀐다."""
    pooling = model[1]
    pooling.pooling_mode_mean_tokens = (mode == "mean")
    pooling.pooling_mode_max_tokens = (mode == "max")
    pooling.pooling_mode_cls_token = (mode == "cls")
    pooling.pooling_mode_mean_sqrt_len_tokens = False
    print(
        f"    pooling set -> mean={pooling.pooling_mode_mean_tokens} "
        f"max={pooling.pooling_mode_max_tokens} cls={pooling.pooling_mode_cls_token}",
        flush=True,
    )


def get_or_encode(max_length, pooling, texts_train, texts_val, device):
    """(max_length, pooling) 조합의 임베딩을 캐시에서 찾거나 새로 인코딩."""
    is_baseline_encoding = (
        max_length == ct.SBERT_BASELINE["max_length"]
        and pooling == ct.SBERT_BASELINE["pooling"]
    )
    if is_baseline_encoding:
        train_path = ct.SBERT_BASELINE_CACHE["train"]
        val_path = ct.SBERT_BASELINE_CACHE["val"]
    else:
        tag = f"{ct.SBERT_TUNE_NAME}_ml{max_length}_pool{pooling}"
        train_path = ct.TUNING_CACHE_DIR / f"{tag}_train.npy"
        val_path = ct.TUNING_CACHE_DIR / f"{tag}_val.npy"

    if train_path.exists() and val_path.exists():
        print(f"  [skip encode] {train_path.name} / {val_path.name} 이미 존재", flush=True)
        return np.load(train_path), np.load(val_path)

    print(f"  encoding max_length={max_length} pooling={pooling} ...", flush=True)
    model = SentenceTransformer(ct.SBERT_TUNE_HF_ID, device=device)
    model.max_seq_length = max_length
    _set_pooling_mode(model, pooling)

    t0 = time.time()
    X_train = model.encode(
        texts_train, batch_size=base_common.ENCODE_BATCH_SIZE,
        convert_to_numpy=True, show_progress_bar=True,
    )
    X_val = model.encode(
        texts_val, batch_size=base_common.ENCODE_BATCH_SIZE,
        convert_to_numpy=True, show_progress_bar=True,
    )
    print(f"  -> encoded in {time.time() - t0:.1f}s, shape train={X_train.shape} val={X_val.shape}", flush=True)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(train_path, X_train)
    np.save(val_path, X_val)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return X_train, X_val


def compute_metrics(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics = {"macro_f1": float(macro_f1)}
    for sym, f1 in zip(TARGET_SYMPTOMS, per_label_f1):
        metrics[f"f1_{sym}"] = float(f1)
    return metrics


def build_head(n_features, n_labels, head):
    if head == "linear":
        return nn.Linear(n_features, n_labels)
    if head == "mlp":
        return nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, n_labels),
        )
    raise ValueError(f"unknown head: {head}")


def train_classifier(X_train, y_train, X_val, y_val, lr, head, device):
    base_common.set_seed(base_common.SEED)
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    model = build_head(X_train_t.shape[1], len(TARGET_SYMPTOMS), head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n = X_train_t.shape[0]
    t0 = time.time()
    for _epoch in range(base_common.CLASSIFIER_EPOCHS):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, base_common.CLASSIFIER_BATCH_SIZE):
            idx = perm[start:start + base_common.CLASSIFIER_BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_val_t)).cpu().numpy()
    val_preds = (val_probs >= base_common.THRESHOLD).astype(int)

    metrics = compute_metrics(y_val, val_preds)
    return metrics, round(elapsed, 1)


def run_config(config_id, max_length, pooling, lr, head, texts_train, texts_val, y_train, y_val, device):
    print(f"=== config {config_id}: max_length={max_length} pooling={pooling} lr={lr} head={head} ===", flush=True)
    X_train, X_val = get_or_encode(max_length, pooling, texts_train, texts_val, device)
    metrics, train_time_s = train_classifier(X_train, y_train, X_val, y_val, lr, head, device)

    result = {
        "config_id": config_id, "name": ct.SBERT_TUNE_NAME, "hf_id": ct.SBERT_TUNE_HF_ID,
        "max_length": max_length, "pooling": pooling, "lr": lr, "head": head,
        "train_time_s": train_time_s, "val_macro_f1": metrics["macro_f1"],
    }
    for sym in TARGET_SYMPTOMS:
        result[f"f1_{sym}"] = metrics[f"f1_{sym}"]
    print(f"  val_macro_f1={result['val_macro_f1']:.4f}", flush=True)
    return result


def load_baseline_result():
    with open(ct.SBERT_BASELINE_RESULT_JSON, encoding="utf-8") as f:
        r = json.load(f)
    r = dict(r)
    r["config_id"] = "baseline"
    r.update(ct.SBERT_BASELINE)
    return r


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== SBERT({ct.SBERT_TUNE_NAME}) 튜닝 시작 (device={device}) ===", flush=True)

    train_df = base_common.load_split_csv(base_common.TRAIN_CSV)
    val_df = base_common.load_split_csv(base_common.VAL_CSV)
    texts_train, y_train = base_common.get_texts_and_labels(train_df)
    texts_val, y_val = base_common.get_texts_and_labels(val_df)

    results = _load_existing()
    have_ids = {r["config_id"] for r in results}
    b = ct.SBERT_BASELINE

    if "baseline" not in have_ids:
        results.append(load_baseline_result())
        have_ids.add("baseline")
        _save(results)

    def maybe_run(config_id, max_length, pooling, lr, head):
        if config_id in have_ids:
            print(f"  [skip run] {config_id} 이미 완료됨", flush=True)
            return
        results.append(run_config(config_id, max_length, pooling, lr, head,
                                   texts_train, texts_val, y_train, y_val, device))
        have_ids.add(config_id)
        _save(results)

    for ml in ct.SBERT_MAX_LENGTH_CANDIDATES:
        maybe_run(f"max_length_{ml}", ml, b["pooling"], b["lr"], b["head"])

    for pool in ct.SBERT_POOLING_CANDIDATES:
        maybe_run(f"pooling_{pool}", b["max_length"], pool, b["lr"], b["head"])

    for lr in ct.SBERT_LR_CANDIDATES:
        maybe_run(f"lr_{lr}", b["max_length"], b["pooling"], lr, b["head"])

    for head in ct.SBERT_HEAD_CANDIDATES:
        maybe_run(f"head_{head}", b["max_length"], b["pooling"], b["lr"], head)

    def best_of(ids):
        candidates = [r for r in results if r["config_id"] in ids]
        return max(candidates, key=lambda r: r["val_macro_f1"])

    best_ml = best_of(["baseline"] + [f"max_length_{ml}" for ml in ct.SBERT_MAX_LENGTH_CANDIDATES])["max_length"]
    best_pool = best_of(["baseline"] + [f"pooling_{p}" for p in ct.SBERT_POOLING_CANDIDATES])["pooling"]
    best_lr = best_of(["baseline"] + [f"lr_{lr}" for lr in ct.SBERT_LR_CANDIDATES])["lr"]
    best_head = best_of(["baseline"] + [f"head_{h}" for h in ct.SBERT_HEAD_CANDIDATES])["head"]
    print(f"=== 그룹별 최선값: max_length={best_ml} pooling={best_pool} lr={best_lr} head={best_head} ===", flush=True)

    already_tested = any(
        r["max_length"] == best_ml and r["pooling"] == best_pool
        and r["lr"] == best_lr and r["head"] == best_head
        for r in results
    )
    if already_tested:
        print("=== 최선 조합이 이미 baseline/개별 변인 중 하나와 동일 - 추가 실행 생략 ===", flush=True)
    else:
        maybe_run("best_combo", best_ml, best_pool, best_lr, best_head)

    print(f"=== SBERT 튜닝 완료. {len(results)}개 조합 -> {RESULT_PATH} ===", flush=True)


if __name__ == "__main__":
    main()
