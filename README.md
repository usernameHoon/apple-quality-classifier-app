# 🍎 사과 품질 예측 애플리케이션

이 프로젝트는 이미지를 기반으로 사과의 품질(예: 특, 상, 보통)을 예측하는 AI 기반 웹 애플리케이션입니다. 
프론트엔드는 React로, 모델 서버는 사전 학습된 MobilenetV2 모델을 사용하는 Python 기반으로 구성되어 있습니다.

---

## ✨ 주요 기능
- 웹 인터페이스를 통해 사과 이미지를 업로드
- 실시간 품질 예측 결과 제공 (예: "특", "상", "보통")
- 경량 딥러닝 모델을 사용하여 빠른 추론 가능
- 사용하기 쉽고 반응형 UI 제공

---

## 📁 프로젝트 구조
```bash
apple-quality-classifier-app/
├── frontend/               # React + TailwindCSS 기반 프론트엔드
│   └── apple-quality-predictor-ui/
│       ├── src/            # React 컴포넌트 및 로직
│       ├── public/         # 정적 파일
│       └── package.json    # 프로젝트 설정
├── model-server/          # Python FastAPI 기반 모델 서버
    ├── app/
    │   ├── predict.py      # 예측 로직
    │   └── server.py       # API 서버 실행
    ├── model/
    │   └── mobilenetv2_model.pt  # 학습된 PyTorch 모델 파일
    └── scripts/
        ├── train.py        # 모델 학습 스크립트
        └── evaluate.py     # 모델 평가 스크립트
```

---

## 🚀 시작하기

### 1. 프론트엔드 실행
```bash
cd frontend/apple-quality-predictor-ui
npm install
npm start
```

### 2. 모델 서버 실행
```bash
cd model-server
pip install -r requirements.txt  # 또는 FastAPI, torch 등 수동 설치
python app/server.py             # 모델 API 서버 실행
```

> ⚠️ 프론트엔드와 모델 서버는 서로 통신 가능한 포트에서 실행되어야 합니다.

---

## 💡 실행 화면
![사과 품질 예측 사이트 - Chrome 2025-06-17 15-51-10](https://github.com/user-attachments/assets/3e6596de-4a3a-4478-a3fe-5e52b60907c2)


---

## 🧱 모델 정보
- **아키텍처**: MobilenetV2
- **프레임워크**: PyTorch
- **모델 파일**: `mobilenetv2_model.pt`

---

## 📖 스크립트
- `train.py`: 사과 품질 분류 모델 학습
- `evaluate.py`: 테스트 데이터셋 기반 성능 평가
