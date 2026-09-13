"""
3주차 파라미터 튜닝 공통 상수/유틸.

sbert_baseline/common.py를 그대로 재사용하고(경로·라벨순서·baseline 하이퍼파라미터
전부 거기 있는 값 그대로 씀), 튜닝 전용 경로·그리드만 여기 추가한다.
baseline 결과 폴더(train_results)는 건드리지 않고, 튜닝 결과는 전부
tuning_results/에 따로 쌓는다.
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent          # 3주차/tuning
WEEK3_DIR = _THIS_DIR.parent                          # 3주차
PROJECT_ROOT = WEEK3_DIR.parent

SBERT_BASELINE_DIR = WEEK3_DIR / "sbert_baseline"
sys.path.insert(0, str(SBERT_BASELINE_DIR))
import common as base_common  # noqa: E402  (sbert_baseline/common.py 재사용)

MISSION1_WEEK1_CODE_DIR = PROJECT_ROOT / "1주차" / "mission3_code"

TARGET_SYMPTOMS = base_common.TARGET_SYMPTOMS

TUNING_CACHE_DIR = WEEK3_DIR / "tuning_cache"
TUNING_RESULTS_DIR = WEEK3_DIR / "tuning_results"
TUNING_CKPT_DIR = _THIS_DIR / "checkpoints"
TUNING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TUNING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TUNING_CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SBERT(ko-sroberta) 튜닝 대상 + 그리드
# ---------------------------------------------------------------------------
SBERT_TUNE_NAME = "ko-sroberta"
SBERT_TUNE_HF_ID = base_common.SBERT_MODELS[SBERT_TUNE_NAME]

SBERT_BASELINE = {
    "max_length": base_common.MAX_LENGTH,      # 256
    "pooling": "mean",
    "lr": base_common.CLASSIFIER_LR,           # 1e-3
    "head": "linear",
}
SBERT_MAX_LENGTH_CANDIDATES = [384, 512]
SBERT_POOLING_CANDIDATES = ["max", "cls"]
SBERT_LR_CANDIDATES = [5e-4, 5e-3]
SBERT_HEAD_CANDIDATES = ["mlp"]

# baseline(256/mean) 임베딩은 이미 sbert_baseline 단계에서 만들어져 있으니 재사용
SBERT_BASELINE_CACHE = {
    "train": base_common.CACHE_DIR / f"{SBERT_TUNE_NAME}_train.npy",
    "val": base_common.CACHE_DIR / f"{SBERT_TUNE_NAME}_val.npy",
}
SBERT_BASELINE_RESULT_JSON = base_common.RESULTS_DIR / f"sbert_{SBERT_TUNE_NAME}.json"

# ---------------------------------------------------------------------------
# PLM(klue-roberta) 튜닝 대상 + 그리드
# ---------------------------------------------------------------------------
PLM_TUNE_NAME = "klue-roberta"
PLM_TUNE_HF_ID = "klue/roberta-base"

PLM_BASELINE = {
    "max_length": 256,
    "lr": 2e-5,
}
PLM_MAX_LENGTH_CANDIDATES = [384, 512]
PLM_LR_CANDIDATES = [1e-5, 5e-5]
PLM_BATCH_SIZE = 16   # 1주차 train_utils.py DEFAULT_BATCH_SIZE와 동일 (변인 통제)
PLM_EPOCHS = 3        # 1주차 train_utils.py DEFAULT_EPOCHS와 동일 (변인 통제)

PLM_BASELINE_RESULT_JSON = (
    PROJECT_ROOT / "1주차" / "mission3_code" / "train_results" / "step5_klue-roberta.json"
)
