# GitHub 저장소 교체 가이드 (수리평가 반영 최종본)

**중요: 수리 특화 평가자가 공개 저장소를 직접 확인하여, supplement에 positive control이 없음(구버전)을 blocking으로 지적했습니다. 이번 교체는 제출 전 필수입니다.**

## 교체·추가 대상

| 경로 | 내용 |
|---|---|
| `supplements/manuscript.pdf` / `.tex` | Eq.(3) one-step 표기 분리, Eq.(4)→exact line integral + Taylor 분리(신규 Eq.5), Methods 보강 4건, CV_null 각주 (12쪽) |
| `supplements/supplement.pdf` / `.tex` | **S6 positive control 섹션 + Table SVII** (평가자 지적의 핵심), 민감도 문장, governs→predicts (4쪽) |
| `supplements/figs/fig1_pipeline.pdf` | baseline 라벨 (기존) |
| `code/make_pipeline.py` | 위 그림 스크립트 |
| `data/results_positive_control_*.csv` | **신규 4개**: channels_only, genuine_channel, summary, sensitivity |
| `README.md` | supplements/ 폴더명 + positive-control 데이터 행 추가 |

## 방법 (웹)
저장소 → Add file ▸ Upload files → 이 패키지의 `supplements/`, `code/`, `data/`, `README.md`를 통째로 드래그앤드롭 → Commit.
(data/의 기존 CSV는 그대로 유지되고 신규 4개만 추가됩니다.)

## 교체 확인
- `supplements/supplement.pdf`에 **"S6. SYNTHETIC POSITIVE CONTROL"** 섹션과 Table SVII이 보이면 최신본.
- `data/`에 `results_positive_control_summary.csv`가 보이면 완료.
