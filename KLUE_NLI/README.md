# Dacon : 한국어 문장 관계 분류 경진대회

Solution for 한국어 문장 관계 분류 경진대회 by **gistarrr**

## Content
- [Competition Abstract](#competition-abstract)
- [Environment](#environment)
- [Archive Contents](#archive-contents)
- [Solution](#solution)
  * [Install Requirements](#install-requirements)
  * [Argument](#argument)
  * [Running Code(Train & Inference)](#running-code-train---inference-)
  * [Hard-voting Ensemble](#hard-voting-ensemble)
- [Final result](#-----)


## Competition Abstract

- premise 문장을 참고해 hypothesis 문장이 참인지(Entailment), 거짓인지(Contradiction), 혹은 참/거짓 여부를 알 수 없는 문장인지(Neutral)를 판별해야 합니다.
- 데이터셋 통계:
  - train.csv: 총 24998개
  - test_data.csv: 총 1666개 (정답 라벨 blind = 100으로 임의 표현)
- 자세한 내용은 [데이콘 대회 홈페이지](https://dacon.io/competitions/official/235875/overview/description) 참고

## Environment

- Colaboratory Pro

## Archive Contents

- KLUE_NLI : 구현 코드와 모델 checkpoint 및 모델 결과를 포함하는 디렉토리

```
KLUE_NLI/
├── input/
│   ├── aeda.ipynb
│   ├── test_data.csv
│   ├── train_aeda.csv
│   ├── train_data.csv
│   ├── train_rtt.csv
│   ├── validation_aeda.csv
│   ├── validation_rtt.csv
│   └── validation.csv
├── arguments.py
├── data_collator.py
├── data.py
├── inference.py
├── model.py
├── train.py
├── trainer.py
├── running.ipynb
├── crawling_papago_rtt.ipynb
└── ensemble_voting.ipynb
```

- `inputs/` : 모델 학습에 필요한 데이터
- `aeda.ipynb` : 데이터 Augumentation 기법 중 하나인 Aeda 기법 적용
- `data_collator.py` : 모델에 맞는 Data Collator 정의
- `data.py` : 데이터 불러오기
- `train.py` : 단일 모델 및 k-fold 모델 학습
- `trainer.py` : 모델에 맞는 Custom Trainer 정의
- `model.py` : Custom Model
- `inference.py` : 단일 모델 및 k-fold 모델 결과 생성
- `running.ipynb` : 코랩 환경에서 실행할 수 있는 노트북 파일
- `crawling_papago_rtt.ipynb` : RTT(round-trip translation)를 위한 파파고 번역기를 이용하는 셀레니움 코드
- `ensemble_voting.ipynb` : 여러 모델을 앙상블

## Solution

### Install Requirements

```
pip install transformers datasets wandb python-dotenv
```

### Argument

|      argument       | description | default                                       |
| :-----------------: | :--------   | :------------------------------------------- |
|model_name_or_path| Pretrained Model 선택| klue/roberta-large|
| data_name | 학습 데이터 파일 Path | train_data.csv |
| save_path | 학습한 모델 저장 폴더 | ./checkpoints/roberta-large |
| output_name | 최종 csv 파일 이름 | ./output/roberta-large      |
| aeda | Aeda 데이터 파일 사용 유무  | False     |
| train_rtt | train_rtt 데이터 사용 유무 | False  |
| valid_rtt | valid_rtt 데이터 사용 유무 | False   |
| k_fold  | k-fold 수 | 5 |
| use_lstm | Roberta + LSTM 모델 사용 유무 | False |
| use_SIC | Roberta + Self-Explainable 모델 사용 유무 | False |
| lamb | Self-Explainable loss 계수 | 1.0 |
| use_rdrop | R-drop loss 사용 유무 | False |
| alpha | R-drop loss 계수 | 1.0 |
| report_to | 시각화 툴 이름 | wandb |
| dotenv_path | WandB API Key 파일 Path | ./wandb.env |
| project_name | WandB project 이름 | KLUE-NLI |

### Train & Inference

- **Model 1** : Validation RTT data + Roberta-Large + R-drop
- **Model 2** : Validation RTT data + Roberta-LSTM + R-drop
- **Model 3** : Validation RTT data + Self-Explainable + R-drop
- **Model 4** : Aeda data + Self-Explainable
- 위 4 모델을 각각 5 k-fold를 진행하여 각 Fold별 soft voting을 이용하여 결과물 생성
- 실행 코드는 `running.ipynb` 참고

### Hard-voting Ensemble

- 위 4가지 모델로부터 생성된 csv 파일을 Hard voting 하여 최종 결과물 생성
- `ensemble_voting.ipynb`를 통해 생성

## Final result

|         | ACC | RANK |
| :-----: |:----: | :--: |
| Public  | 89.5 |  14  |
| Private | 89.315 |  10   |
