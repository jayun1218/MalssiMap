import os
import shutil
import tempfile
import whisper
import joblib
import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

# FastAPI 앱 초기화
app = FastAPI(
    title="말씨 지도 (Malssi Map) API",
    description="음성 및 텍스트를 기반으로 지역 방언을 분석하는 API",
    version="0.1.0"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper 모델 로드
print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")
print("Whisper model loaded!")

# Random Forest 베이스라인 모델 로드 시도
rf_model = None
model_path = os.path.join("models", "dialect_rf_model.pkl")
if os.path.exists(model_path):
    try:
        rf_model = joblib.load(model_path)
        print(f"Machine Learning model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading ML model: {e}")
else:
    print("No ML model found. Will use dummy data for dialect probabilities.")

class AnalysisResponse(BaseModel):
    transcript: str
    confidence: float
    result: Dict[str, float]

@app.get("/")
def read_root():
    return {"message": "Welcome to Malssi Map API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

def extract_mfcc_from_file(file_path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        # 1. Whisper를 이용한 STT 처리
        result = whisper_model.transcribe(temp_file_path, language="ko")
        transcript = result.get("text", "").strip()
        
        # 2. 방언 분류 (머신러닝 모델 사용 또는 더미 데이터)
        if rf_model is not None:
            # 특징 추출
            mfcc_features = extract_mfcc_from_file(temp_file_path)
            # 예측 (확률)
            probs = rf_model.predict_proba([mfcc_features])[0]
            classes = rf_model.classes_
            
            # 클래스 이름을 키로, 확률을 값으로 매핑
            final_result = {cls_name: float(prob) for cls_name, prob in zip(classes, probs)}
            
            # 누락된 지역이 있다면 0.0 처리
            for region in ["gyeongsang", "jeolla", "chungcheong", "standard"]:
                if region not in final_result:
                    final_result[region] = 0.0
        else:
            final_result = {
                "gyeongsang": 0.82,
                "jeolla": 0.08,
                "chungcheong": 0.06,
                "standard": 0.04
            }
        
        return AnalysisResponse(
            transcript=transcript,
            confidence=0.87,
            result=final_result
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
