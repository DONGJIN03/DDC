"""
SBERT 튜닝 -> PLM(klue-roberta) 튜닝을 순서대로 실행하는 러너.

원격 GPU에서 이 파일 하나만 실행하면 둘 다 끝난다. tune_sbert.py, tune_klue_roberta.py
둘 다 이미 끝난 config는 건너뛰는 로직이 있어서, 중간에 끊겨도 이 파일을 다시
실행하면 이어서 진행된다.

실행:
    python run_tuning_remote.py
    (로그를 파일로도 남기려면) python run_tuning_remote.py > tuning_log.txt 2>&1

주의: SBERT 쪽은 몇 분이면 끝나지만, PLM(klue-roberta) 쪽은 baseline 기준
회당 63분 - max_length를 늘리면 더 걸릴 수 있어서, 4~5개 config 합쳐 총
5~8시간 정도 예상.
"""

import sys
import time

import tune_sbert
import tune_klue_roberta


def main():
    t0 = time.time()

    print("############################################", flush=True)
    print("# SBERT(ko-sroberta) 튜닝 시작", flush=True)
    print("############################################", flush=True)
    try:
        tune_sbert.main()
    except Exception as e:
        print(f"[중단] SBERT 튜닝에서 에러 발생: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
    print(f"SBERT 튜닝 완료 (누적 {time.time() - t0:.1f}s)\n", flush=True)

    print("############################################", flush=True)
    print("# PLM(klue-roberta) 튜닝 시작 - 오래 걸림 (수 시간 예상)", flush=True)
    print("############################################", flush=True)
    try:
        tune_klue_roberta.main()
    except Exception as e:
        print(f"[중단] PLM 튜닝에서 에러 발생: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
    print(f"PLM 튜닝 완료 (총 {time.time() - t0:.1f}s)\n", flush=True)

    print("=== 전체 튜닝 파이프라인 완료 ===", flush=True)
    print(
        "tuning_results/ 안의 tune_sbert_results.json, tune_klue-roberta_results.json을 "
        "로컬로 옮기면 됩니다.",
        flush=True,
    )


if __name__ == "__main__":
    main()
