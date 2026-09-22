# 말씨 지도 AI (Malssi Map AI) 🗺️🎙️

사용자의 목소리를 분석하여 어떤 지역 방언(사투리)의 특징이 가장 많이 담겨있는지 알려주는 AI 웹 서비스입니다. 음성을 업로드하거나 직접 마이크로 녹음하면, 모델이 텍스트로 변환(STT)해주고 지역별 사투리 유사도 리포트를 제공합니다.

## ✨ 주요 기능
- **직접 녹음 및 파일 업로드**: 브라우저 마이크를 통한 실시간 녹음 및 오디오 파일 업로드 지원
- **Speech-to-Text (STT)**: OpenAI Whisper 모델을 활용한 음성 인식 및 원문 추출
- **방언 분석 분류기**: Librosa를 통한 오디오 특징(MFCC) 추출 및 Scikit-learn (Random Forest) 베이스라인 모델을 통한 실시간 확률 예측
- **시각화 리포트**: Recharts와 Tailwind CSS를 활용한 직관적인 분석 결과 시각화

## 🛠 기술 스택
- **Frontend**: Next.js (App Router), React, Tailwind CSS, Recharts, Lucide-react
- **Backend**: FastAPI, Python
- **AI / ML**: OpenAI Whisper, Scikit-learn, Librosa, NumPy

## 📂 프로젝트 구조
```text
.
├── frontend/             # Next.js 프론트엔드 코드
│   ├── src/app/          # 메인 UI 및 API 연동 (page.tsx)
│   └── ...
├── backend/              # FastAPI 백엔드 코드
│   ├── main.py           # API 서버 및 추론 로직
│   ├── ml/               # 머신러닝 파이프라인 (데이터 생성, 전처리, 학습)
│   ├── data/             # 오디오 데이터 셋 저장 경로 (git-ignored)
│   └── models/           # 학습된 모델 파일 저장 경로 (git-ignored)
└── README.md
```

## 🚀 실행 방법 (로컬 환경)

### 1. 백엔드 (FastAPI) 실행
```bash
cd backend
# 가상환경 활성화 (MacOS/Linux)
source .venv/bin/activate
# 필수 패키지 설치
pip install -r requirements.txt
# 서버 실행 (기본 포트: 8000)
uvicorn main:app --reload
```
서버 실행 시 초기 1회 Whisper 모델(`tiny`)이 다운로드 및 로드됩니다.

### 2. 프론트엔드 (Next.js) 실행
새로운 터미널 창을 엽니다.
```bash
cd frontend
# 패키지 설치
npm install
# 개발 서버 실행 (기본 포트: 3000)
npm run dev
```
이후 브라우저에서 `http://localhost:3000`에 접속하여 서비스를 이용할 수 있습니다.

## 🧠 머신러닝 파이프라인 가이드 (Phase 1 & 2)
실제 데이터로 모델을 훈련시키고 싶다면 `backend/ml/`의 스크립트들을 사용하세요.
1. `backend/data/raw` 폴더에 `.wav` 파일과 `metadata.csv`를 준비합니다. (테스트를 원하시면 `python ml/01_generate_dummy.py`를 실행하여 가짜 오디오를 만드세요)
2. `python ml/02_preprocess.py` 를 실행하여 MFCC를 추출하고 데이터셋을 분할합니다.
3. `python ml/03_train_model.py` 를 실행하면 Random Forest 모델이 훈련되어 `models/dialect_rf_model.pkl`에 저장됩니다. 백엔드를 재시작하면 새로운 모델이 즉시 적용됩니다!
