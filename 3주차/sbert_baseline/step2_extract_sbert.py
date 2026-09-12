"""
STEP2 - SBERT 후보 3종으로 train/val 문장 벡터 추출 (GPU, 무거운 단계).

원격 GPU 서버에서 실행하는 것을 전제로 한다. 실행 전 준비물:
  - 이 폴더(sbert_baseline/) 전체
  - <프로젝트 루트>/전처리 데이터/train.csv, val.csv  (경로는 common.py 참고)
  - pip install sentence-transformers (torch는 mission3 conda 환경에 이미 있음)

실행:
    python step2_extract_sbert.py

이미 만들어진 <모델>_<split>.npy가 있으면 그 조합은 건너뛴다(재실행 시
이어하기 가능). 모델별로 SentenceTransformer를 새로 로드하므로 GPU 메모리는
한 번에 모델 1개분만 사용한다.

산출물:
  - sbert_cache/<name>_train.npy, <name>_val.npy   (name in common.SBERT_MODELS)
  - train_results/sbert_raw_stats.json 의 "encode" 섹션
    (모델별 소요시간, 벡터 shape, max_length=256 초과 truncation 건수)
"""

import json
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import common
from common import CACHE_DIR, MAX_LENGTH, ENCODE_BATCH_SIZE, RAW_STATS_PATH, SBERT_MODELS


def _load_raw_stats():
    if RAW_STATS_PATH.exists():
        with open(RAW_STATS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_raw_stats(stats):
    with open(RAW_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def _count_truncated(model, texts):
    """max_length(256) 초과로 실제 잘리는 문장 수를 센다.
    forward pass 없이 tokenizer만 한 번 더 돌리는 거라 가벼움."""
    enc = model.tokenizer(texts, truncation=False)
    lengths = [len(ids) for ids in enc["input_ids"]]
    return sum(1 for L in lengths if L > MAX_LENGTH)


def encode_split(model, name, split_name, texts, stats):
    out_path = CACHE_DIR / f"{name}_{split_name}.npy"
    if out_path.exists():
        print(f"  [skip] {out_path.name} 이미 존재함", flush=True)
        return

    print(f"  encoding {name}/{split_name} ({len(texts)}건)...", flush=True)
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0

    truncated = _count_truncated(model, texts)

    np.save(out_path, embeddings)
    print(
        f"  -> saved {out_path.name} shape={embeddings.shape} "
        f"elapsed={elapsed:.1f}s truncated({MAX_LENGTH}초과)={truncated}",
        flush=True,
    )

    stats.setdefault(name, {})[f"encode_time_{split_name}_s"] = round(elapsed, 1)
    stats[name][f"shape_{split_name}"] = list(embeddings.shape)
    stats[name][f"truncated_{split_name}"] = truncated
    stats[name]["max_length"] = MAX_LENGTH
    stats[name]["hf_id"] = SBERT_MODELS[name]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== STEP2: SBERT 벡터 추출 (device={device}) ===", flush=True)

    train_df = common.load_split_csv(common.TRAIN_CSV)
    val_df = common.load_split_csv(common.VAL_CSV)
    train_texts, _ = common.get_texts_and_labels(train_df)
    val_texts, _ = common.get_texts_and_labels(val_df)
    print(f"train={len(train_texts)}건 val={len(val_texts)}건 로드 완료", flush=True)

    stats = _load_raw_stats()

    for name, hf_id in SBERT_MODELS.items():
        train_done = (CACHE_DIR / f"{name}_train.npy").exists()
        val_done = (CACHE_DIR / f"{name}_val.npy").exists()
        if train_done and val_done:
            print(f"=== {name} ({hf_id}): train/val 캐시 모두 존재, 모델 로드 생략 ===", flush=True)
            continue

        print(f"=== {name} ({hf_id}) 로딩 ===", flush=True)
        model = SentenceTransformer(hf_id, device=device)
        model.max_seq_length = MAX_LENGTH  # encode()가 이 속성을 실제 truncation 길이로 씀

        encode_split(model, name, "train", train_texts, stats)
        encode_split(model, name, "val", val_texts, stats)

        _save_raw_stats(stats)  # 모델 하나 끝날 때마다 저장 (중단돼도 앞 모델 결과는 보존)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("=== STEP2 완료 ===", flush=True)
    print(f"raw stats -> {RAW_STATS_PATH}", flush=True)


if __name__ == "__main__":
    main()
