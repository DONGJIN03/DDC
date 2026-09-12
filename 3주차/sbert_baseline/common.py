"""
3주차 SBERT baseline 공통 상수/유틸.

STEP1/2/3/4가 전부 이 모듈을 import해서 쓴다. 경로나 모델 목록을 바꿔야 하면
이 파일 하나만 고치면 된다 (다른 파일은 여기 정의된 상수/딕셔너리를 참조만 함).

이 파일은 로컬(ipynb)과 원격(GPU 서버) 양쪽에 동일하게 두고 쓴다. torch/
sentence-transformers는 여기서 import하지 않는다 - STEP1/4(로컬, 가벼운 작업)는
그 패키지들이 없어도 동작해야 하기 때문. 무거운 라이브러리는 STEP2/3에서만 import.
"""

import random
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
# 이 파일 위치: <프로젝트 루트>/3주차/sbert_baseline/common.py
_THIS_DIR = Path(__file__).resolve().parent
WEEK3_DIR = _THIS_DIR.parent
PROJECT_ROOT = WEEK3_DIR.parent

DATA_DIR = PROJECT_ROOT / "전처리 데이터"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"

CACHE_DIR = WEEK3_DIR / "sbert_cache"
RESULTS_DIR = WEEK3_DIR / "train_results"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1주차/2주차 기존 baseline 결과 (STEP4 통합 비교표용)
WEEK1_RESULTS_DIR = PROJECT_ROOT / "1주차" / "mission3_code" / "train_results"
WEEK2_RESULTS_DIR = PROJECT_ROOT / "2주차" / "stepD_results"

RAW_STATS_PATH = RESULTS_DIR / "sbert_raw_stats.json"
SPLIT_CHECK_PATH = RESULTS_DIR / "train_val_split_check.txt"
SUMMARY_PATH = RESULTS_DIR / "sbert_summary.txt"

# 코드를 원격 GPU 서버로 옮길 때: 위 경로들은 전부 "이 파일 기준 상대 경로"라
# 폴더 구조(3주차/sbert_baseline, 그 옆에 sbert_cache·train_results, 그 위
# 두 단계 위에 전처리 데이터/)만 그대로 유지해서 옮기면 코드 수정 없이 바로 동작한다.
# 절대경로를 하드코딩한 곳은 없음.

# ---------------------------------------------------------------------------
# 라벨 순서 (1주차 data_utils.py의 TARGET_SYMPTOMS와 완전히 동일 - 하드코딩 유지)
# ---------------------------------------------------------------------------
TARGET_SYMPTOMS = [
    "고열", "구토", "두통", "복통", "어지러움",
    "열상", "오심", "전신쇠약", "호흡곤란",
]

# ---------------------------------------------------------------------------
# SBERT 후보 3종 (3주차_실험계획.md PHASE 0 표와 동일)
# ---------------------------------------------------------------------------
SBERT_MODELS = {
    "ko-sroberta": "jhgan/ko-sroberta-multitask",
    "kr-sbert": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    "ko-sbert-sts": "jhgan/ko-sbert-sts",
}

# ---------------------------------------------------------------------------
# 실험 고정값
# ---------------------------------------------------------------------------
MAX_LENGTH = 256          # SBERT encode 시 명시 (기본값 128 사용 금지)
ENCODE_BATCH_SIZE = 64
THRESHOLD = 0.5           # 9개 전부 고정, dev 미분리
SEED = 42

# 분류기(Linear 헤드) 하이퍼파라미터 - 얼린 인코더라 PLM full fine-tuning(lr=2e-5)보다
# 큰 lr을 씀 (헤드만 학습하므로). 값을 바꾸면 이 상수만 고치면 됨.
CLASSIFIER_LR = 1e-3
CLASSIFIER_EPOCHS = 30
CLASSIFIER_BATCH_SIZE = 256
CLASSIFIER_OPTIMIZER = "Adam"

# 1주차 klue-roberta 결과 (같은 RoBERTa 뼈대를 full fine-tuning한 것) - 비교 기준값.
# 1주차/mission3_code/train_results/step5_klue-roberta.json 에서 그대로 가져온 값.
KLUE_ROBERTA_HF_ID = "klue/roberta-base"
KLUE_ROBERTA_VAL_MACRO_F1 = 0.6023357094710049

# ---------------------------------------------------------------------------
# sentencepiece 안전장치 (1주차 이슈 04 대응, 방어적으로만 포함)
# ---------------------------------------------------------------------------
# 1주차에서 한글 경로 + sentencepiece 조합으로 kobert(use_fast=False, 커스텀
# 토크나이저)에서 "Illegal byte sequence" 에러가 난 적이 있음 (train_utils.py에서
# monkeypatch로 해결). 이번 SBERT 3종은 전부 klue-roberta 계열 fast tokenizer
# (tokenizer.json)라 이 버그 경로를 안 탈 가능성이 높지만, 혹시 몰라 같은 패치를
# 방어적으로 적용해둔다. 문제 없으면 그냥 조용히 아무 일도 안 함.
def _patch_sentencepiece_unicode_safe():
    try:
        import sentencepiece as _spm
    except ImportError:
        return

    if getattr(_spm.SentencePieceProcessor, "_week3_patched", False):
        return

    _original_load = _spm.SentencePieceProcessor.Load

    def _unicode_safe_load(self, model_file=None, model_proto=None, **kwargs):
        if model_file is not None and model_proto is None:
            with open(model_file, "rb") as f:
                model_proto = f.read()
            return self.LoadFromSerializedProto(model_proto)
        return _original_load(self, model_file=model_file, model_proto=model_proto, **kwargs)

    _spm.SentencePieceProcessor.Load = _unicode_safe_load
    _spm.SentencePieceProcessor._week3_patched = True


_patch_sentencepiece_unicode_safe()


# ---------------------------------------------------------------------------
# 공통 함수
# ---------------------------------------------------------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_split_csv(path):
    """train.csv/val.csv를 로드하고 라벨 컬럼 순서가 TARGET_SYMPTOMS와
    동일한지 검증한다. 다르면 바로 에러를 내서 조용히 잘못된 순서로
    진행되는 걸 막는다."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    label_cols = list(df.columns[1:])
    if label_cols != TARGET_SYMPTOMS:
        raise ValueError(
            f"{path} 라벨 컬럼 순서가 TARGET_SYMPTOMS와 다름.\n"
            f"  파일 순서: {label_cols}\n  기대 순서: {TARGET_SYMPTOMS}"
        )
    return df


def get_texts_and_labels(df):
    texts = df["text"].astype(str).tolist()
    labels = df[TARGET_SYMPTOMS].values.astype(int)
    return texts, labels
