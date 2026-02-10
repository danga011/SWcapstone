# Final Dataset
본 데이터셋은 산업용 표면 결함 탐지를 위한 이미지 데이터로 구성되어 있습니다.


## 데이터 구조

데이터는 3단계 계층 구조로 정리되어 있습니다:

```
dataset_type / defect_type / label
```

| 계층 | 설명 | 값 |
|------|------|-----|
| dataset_type | 데이터셋 종류 | Kolektor, MVTec, NEU |
| defect_type | 결함 유형 | 데이터셋별로 상이 |
| label | 레이블 | normal, anomaly |


---


## Dataset Type 상세

### Kolektor
- **설명**: 산업용 금속 표면 결함 데이터
- **특징**: 세부 결함 타입 정보 미제공
- **결함 유형**: surface_defect (단일 타입으로 통합)


### MVTec
- **설명**: 이상 탐지 벤치마크 데이터셋
- **선별 기준**: 금속 및 표면 결함 특성이 강한 카테고리만 선별
- **결함 유형**:
  - grid
  - metal_nut
  - tile


### NEU
- **설명**: 금속 표면 결함 데이터셋
- **특징**: 결함 유형이 명확하게 구분되어 있음
- **결함 유형**: Crazing, Inclusion, Patches, Pitted, Rolled, Scratches


---


## Defect Type 정리 기준

| Dataset | Defect Types | 비고 |
|---------|--------------|------|
| Kolektor | surface_defect | 원본 데이터에 세부 타입 정보 없음 |
| MVTec | grid, metal_nut, tile | 표면 결함 특성이 강한 타입만 선별 사용 |
| NEU | Crazing, Inclusion, Patches, Pitted, Rolled, Scratches | 원본 제공 결함 이름 사용 |


---


## Label 설명

### normal
- 정상 이미지
- 용도: 정상 샘플 학습용

### anomaly
- 이상 이미지
- 용도: 검증, 평가, 시각화


---


## 중요 사항

> **NEU 데이터셋**
> 
> NEU 데이터셋은 anomaly-only 데이터셋입니다! 정상 이미지가 포함되어 있지 않으며, 디렉토리 구조상 `anomaly/` 폴더만 존재합니다.


> **Train/Validation/Test 분리**
> 
> 현재 데이터셋은 train/val/test로 사전 분리되어 있지 않습니다. MLOps 파이프라인에서 실험 목적에 맞게 자유롭게 split을 구성할 수 있도록 하기 위함이며, 이후 필요시 split 기준에 맞춰 재구성 가능합니다.