# GitHub 저장소 교체 가이드 (최종 제출본 반영)

이 패키지는 **최초 업로드 이후 변경된 파일만** 담고 있습니다.
데이터(`data/`), 나머지 코드, 나머지 그림은 변경이 없으므로 건드릴 필요 없습니다.

## 교체 대상 (7개 파일)

| 저장소 경로 | 변경 사유 |
|---|---|
| `supplements/manuscript.pdf` | 최종 제출본 (blocking 3건 수정, 초록 342→242단어, date 제거 등) |
| `supplements/manuscript.tex` | 위와 동일 |
| `supplements/supplement.pdf` | date 제거 |
| `supplements/supplement.tex` | date 제거 |
| `supplements/figs/fig1_pipeline.pdf` | Fig 4: artifact-controls 박스에 baseline 반영 + 오버플로우 수정 |
| `code/make_pipeline.py` | Fig 4 라벨 변경 |
| `README.md` | 폴더명 `manuscript/` → `supplements/` 반영 |

## 웹에서 교체하는 방법 (가장 간편)

1. GitHub에서 저장소 열기 → **Add file ▸ Upload files**
2. 이 패키지의 `supplements/`, `code/`, `README.md`를 **통째로 드래그앤드롭**
   - 같은 경로의 기존 파일은 자동으로 새 버전으로 덮어써집니다(커밋 히스토리에 남음).
3. 커밋 메시지 예: `Update to final submission version (post-review fixes)`
4. **Commit changes**

## git 명령어로 교체하는 방법

```bash
# 로컬 클론 폴더에서, 이 패키지의 내용물을 덮어쓴 뒤:
git add -A
git commit -m "Update to final submission version (post-review fixes)"
git push
```

## 교체 후 확인

- 저장소의 `supplements/manuscript.pdf` 첫 페이지에 "(Dated: ...)" 줄이 **없으면** 최신본입니다.
- 원고의 [46] 참조가 이 저장소를 가리키므로, **APS 투고 직전에 이 교체를 완료**해 두는 것이 좋습니다.

## (선택) Zenodo DOI

제출 전에 GitHub에서 Release(예: `v1.0`)를 만들고 Zenodo와 연동하면 고정 DOI가 발급됩니다.
DOI를 받으시면 알려주세요 — Data availability 문장에 함께 넣어드립니다.
