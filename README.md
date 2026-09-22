# 말씨 지도 AI (Malssi Map AI) 🗺️🎙️

사용자의 목소리를 분석하여 어떤 지역 방언(사투리)의 특징이 가장 많이 담겨있는지 알려주는 AI 웹 서비스입니다. 
단순한 억양 분석을 넘어, STT(Speech-to-Text)로 추출한 텍스트 문맥과 오디오 스펙트로그램을 동시에 결합(Multimodal Fusion)하여 전국 6대 지역(경상도, 전라도, 충청도, 강원도, 제주도, 표준어)의 사투리 농도를 예측합니다. 
분석 후에는 재미있는 '사투리 칭호 뱃지'가 부여되며, 최근 분석 기록들이 실시간으로 함께 공유됩니다!

## ✨ 주요 기능
- **직접 녹음 및 파일 업로드**: 브라우저 마이크를 통한 실시간 녹음 및 오디오 파일 업로드 지원
- **멀티모달 사투리 분석 딥러닝**:
  - 오디오 파트: Librosa를 통한 Mel-Spectrogram 추출 및 PyTorch CNN 딥러닝 이미지 분류
  - 텍스트 파트: OpenAI Whisper 모델 기반 STT(음성 인식) 및 Character Embedding
  - 두 파트의 특징(Feature)을 결합(`torch.cat`)하여 최종 6개 지역(경상, 전라, 충청, 강원, 제주, 표준어) 확률 도출
- **UI 게임화 (Gamification)**: 사투리 농도(확률)에 따라 다이내믹하고 재치 있는 칭호 뱃지(예: `🔥 불타는 찐 갱상도 네이티브`, `🍊 뀰맛 나는 제주도 네이티브`) 부여 및 미려한 Glassmorphism 디자인 적용
- **히스토리 공유 API**: SQLite 기반의 가벼운 데이터베이스를 연동하여 최근 분석된 다른 유저들의 말씨 기록을 카드 형태로 실시간 조회

## 🛠 기술 스택
- **Frontend**: Next.js (App Router), React, Tailwind CSS, Recharts, Lucide-react
- **Backend**: FastAPI, Python, SQLAlchemy, SQLite
- **AI / ML**: PyTorch, OpenAI Whisper, Librosa, NumPy

## 📂 프로젝트 구조
```text
.
├── frontend/             # Next.js 프론트엔드 코드
│   ├── src/app/          # 메인 UI, 칭호 게임화 로직 및 히스토리 조회 (page.tsx)
│   └── ...
├── backend/              # FastAPI 백엔드 코드
│   ├── main.py           # API 서버, 멀티모달 모델 인퍼런스, DB 연동
│   ├── models.py         # SQLAlchemy DB 모델 정의
│   ├── database.py       # SQLite 커넥션 설정
│   ├── ml/               # 머신러닝 파이프라인 (더미생성 -> 전처리 -> 모델학습)
│   ├── data/             # 오디오 데이터 셋 저장 경로 (git-ignored)
│   └── models/           # 학습된 .pth 딥러닝 모델 저장 경로 (git-ignored)
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
서버 실행 시 초기 1회 Whisper 모델(`tiny`)이 다운로드 되며, SQLite DB(`malssi.db`)가 자동 생성됩니다.

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

## 🧠 머신러닝 파이프라인 가이드 (6-Class Multimodal)
실제 사투리 데이터로 딥러닝 모델을 처음부터 다시 훈련시키고 싶다면 `backend/ml/`의 스크립트들을 사용하세요.

1. **데이터 준비**: `backend/data/raw` 폴더에 `.wav` 파일과 `metadata.csv`를 준비합니다. (테스트를 원하시면 `python ml/01_generate_dummy.py`를 실행하여 6개 지역 가짜 오디오를 만드세요.)
2. **전처리 (Preprocess)**: `python ml/06_preprocess_multimodal.py` 를 실행하여 텍스트 토큰화(Vocab) 및 오디오 텐서화를 수행합니다.
3. **학습 (Train)**: `python ml/07_train_multimodal.py` 를 실행하면 CNN + Text Embedding 퓨전 딥러닝 모델이 훈련되어 `models/dialect_multimodal_model.pth`에 저장됩니다. 
4. 백엔드(FastAPI) 서버가 재시작되면 새로 학습된 모델이 즉시 적용됩니다!
