"""
PLM(klue-roberta) 파라미터 튜닝.

baseline(max_length=256, lr=2e-5)에서 그룹별로 변인 하나씩만 바꿔서 재학습한다:
  - max_length: 384, 512
  - lr:         1e-5, 5e-5

1주차 train_utils.py의 build_trainer()를 그대로 재사용해서, baseline(1주차
step5_klue-roberta.json)과 완전히 동일한 파이프라인·하이퍼파라미터(batch=16,
epoch=3)로 비교한다 - max_length/lr 외에는 아무것도 안 바뀜.

전부 끝나면 그룹별 최선값을 자동으로 골라 "best_combo" 1개를 추가로 실행한다.

실행: python tune_klue_roberta.py   (원격 GPU, 회당 60~120분 예상 - 오래 걸림)
이미 끝난 config_id는 건너뛰므로 중단 후 재실행 가능 (재학습 처음부터 다시 X).
산출물: tuning_results/tune_klue-roberta_results.json
"""

import json
import sys
import time

import common_tuning as ct
from common_tuning import base_common, TARGET_SYMPTOMS

sys.path.insert(0, str(ct.MISSION1_WEEK1_CODE_DIR))
import train_utils  # 1주차 mission3_code/train_utils.py 재사용

RESULT_PATH = ct.TUNING_RESULTS_DIR / "tune_klue-roberta_results.json"


def _load_existing():
    if RESULT_PATH.exists():
        with open(RESULT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save(results):
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_dfs():
    train_df = base_common.load_split_csv(base_common.TRAIN_CSV).copy()
    val_df = base_common.load_split_csv(base_common.VAL_CSV).copy()
    # train_utils.SymptomDataset은 text/labels(9차원 리스트) 컬럼을 기대함 (1주차와 동일 형태)
    train_df["labels"] = train_df[TARGET_SYMPTOMS].values.tolist()
    val_df["labels"] = val_df[TARGET_SYMPTOMS].values.tolist()
    return train_df, val_df


def run_config(config_id, lr, max_length, train_df, val_df):
    print(f"=== config {config_id}: lr={lr} max_length={max_length} (batch={ct.PLM_BATCH_SIZE}, epochs={ct.PLM_EPOCHS}) ===", flush=True)
    output_dir = ct.TUNING_CKPT_DIR / f"klue-roberta_{config_id}"

    t0 = time.time()
    trainer, tokenizer, model = train_utils.build_trainer(
        ct.PLM_TUNE_HF_ID, train_df, val_df, str(output_dir),
        lr=lr, max_length=max_length,
        batch_size=ct.PLM_BATCH_SIZE, epochs=ct.PLM_EPOCHS,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()
    elapsed = time.time() - t0

    result = {
        "config_id": config_id, "name": ct.PLM_TUNE_NAME, "hf_id": ct.PLM_TUNE_HF_ID,
        "lr": lr, "max_length": max_length,
        "train_time_s": round(elapsed, 1),
        "val_macro_f1": eval_metrics.get("eval_macro_f1"),
    }
    for sym in TARGET_SYMPTOMS:
        result[f"f1_{sym}"] = eval_metrics.get(f"eval_f1_{sym}")
    print(f"  val_macro_f1={result['val_macro_f1']:.4f} (train_time={result['train_time_s']}s)", flush=True)
    return result


def load_baseline_result():
    with open(ct.PLM_BASELINE_RESULT_JSON, encoding="utf-8") as f:
        r = json.load(f)
    r = dict(r)
    r["config_id"] = "baseline"
    r.update(ct.PLM_BASELINE)
    return r


def main():
    print(f"=== PLM({ct.PLM_TUNE_NAME}) 튜닝 시작 ===", flush=True)
    train_df, val_df = load_dfs()

    results = _load_existing()
    have_ids = {r["config_id"] for r in results}
    b = ct.PLM_BASELINE

    if "baseline" not in have_ids:
        results.append(load_baseline_result())
        have_ids.add("baseline")
        _save(results)

    def maybe_run(config_id, lr, max_length):
        if config_id in have_ids:
            print(f"  [skip run] {config_id} 이미 완료됨", flush=True)
            return
        results.append(run_config(config_id, lr, max_length, train_df, val_df))
        have_ids.add(config_id)
        _save(results)

    for ml in ct.PLM_MAX_LENGTH_CANDIDATES:
        maybe_run(f"max_length_{ml}", b["lr"], ml)

    for lr in ct.PLM_LR_CANDIDATES:
        maybe_run(f"lr_{lr}", lr, b["max_length"])

    def best_of(ids):
        candidates = [r for r in results if r["config_id"] in ids]
        return max(candidates, key=lambda r: r["val_macro_f1"])

    best_ml = best_of(["baseline"] + [f"max_length_{ml}" for ml in ct.PLM_MAX_LENGTH_CANDIDATES])["max_length"]
    best_lr = best_of(["baseline"] + [f"lr_{lr}" for lr in ct.PLM_LR_CANDIDATES])["lr"]
    print(f"=== 그룹별 최선값: max_length={best_ml} lr={best_lr} ===", flush=True)

    already_tested = any(r["max_length"] == best_ml and r["lr"] == best_lr for r in results)
    if already_tested:
        print("=== 최선 조합이 이미 존재 - 추가 실행 생략 ===", flush=True)
    else:
        maybe_run("best_combo", best_lr, best_ml)

    print(f"=== PLM 튜닝 완료. {len(results)}개 조합 -> {RESULT_PATH} ===", flush=True)


if __name__ == "__main__":
    main()
