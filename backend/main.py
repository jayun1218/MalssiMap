import os
import shutil
import tempfile
import whisper
import numpy as np
import librosa
import torch
import torch.nn as nn
import json
from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from sqlalchemy.orm import Session
from datetime import datetime

# DB 설정 가져오기
import models
from database import engine, get_db

# 테이블 자동 생성
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="말씨 지도 (Malssi Map) API",
    description="음성 및 텍스트를 기반으로 지역 방언을 분석하는 API",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")
print("Whisper model loaded!")

# Multimodal Fusion 모델 정의
class MultimodalFusion(nn.Module):
    def __init__(self, vocab_size, num_classes=4):
        super(MultimodalFusion, self).__init__()
        self.audio_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.audio_fc = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.text_emb = nn.Embedding(vocab_size, 16, padding_idx=0)
        self.text_fc = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, audio, text):
        a = self.audio_net(audio)
        a = a.view(a.size(0), -1)
        a_feat = self.audio_fc(a)
        
        t = self.text_emb(text)
        t = torch.mean(t, dim=1)
        t_feat = self.text_fc(t)
        
        fused = torch.cat((a_feat, t_feat), dim=1)
        out = self.classifier(fused)
        return out

# Vocab 로드
vocab_path = os.path.join("models", "vocab.json")
char2idx = {}
if os.path.exists(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        char2idx = json.load(f)
vocab_size = len(char2idx) + 1 if char2idx else 100

# 모델 로드
mm_model = None
model_path = os.path.join("models", "dialect_multimodal_model.pth")
if os.path.exists(model_path):
    try:
        mm_model = MultimodalFusion(vocab_size=vocab_size, num_classes=4)
        mm_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        mm_model.eval()
        print(f"✅ Multimodal Fusion model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Error loading Multimodal model: {e}")
        mm_model = None
else:
    print("⚠️ No Multimodal model found. Will use dummy data.")

class AnalysisResponse(BaseModel):
    transcript: str
    confidence: float
    result: Dict[str, float]

class HistoryResponse(BaseModel):
    id: int
    transcript: str
    gyeongsang: float
    jeolla: float
    chungcheong: float
    standard: float
    created_at: datetime

    class Config:
        from_attributes = True

def extract_mel_spectrogram_for_inference(file_path, sr=22050, n_mels=128, max_len=130):
    y, sr = librosa.load(file_path, sr=sr)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    
    if melspec_db.shape[1] > max_len:
        melspec_db = melspec_db[:, :max_len]
    else:
        pad_width = max_len - melspec_db.shape[1]
        melspec_db = np.pad(melspec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    melspec_db = np.expand_dims(melspec_db, axis=0) # Channel
    melspec_db = np.expand_dims(melspec_db, axis=0) # Batch
    return torch.tensor(melspec_db, dtype=torch.float32)

def encode_text_for_inference(text, char2idx_map, max_len=20):
    encoded = [char2idx_map.get(ch, 0) for ch in text]
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded = encoded + [0] * (max_len - len(encoded))
    return torch.tensor([encoded], dtype=torch.long) # Add Batch dimension

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        # 1. STT (Whisper)
        result = whisper_model.transcribe(temp_file_path, language="ko")
        transcript = result.get("text", "").strip()
        if not transcript:
            transcript = "소리 없음"
        
        # 2. Multimodal 추론
        if mm_model is not None:
            audio_tensor = extract_mel_spectrogram_for_inference(temp_file_path)
            text_tensor = encode_text_for_inference(transcript, char2idx)
            
            with torch.no_grad():
                logits = mm_model(audio_tensor, text_tensor)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()
            
            classes = ["gyeongsang", "jeolla", "chungcheong", "standard"]
            final_result = {cls_name: float(prob) for cls_name, prob in zip(classes, probs)}
        else:
            final_result = {
                "gyeongsang": 0.82,
                "jeolla": 0.08,
                "chungcheong": 0.06,
                "standard": 0.04
            }
            
        # 3. 데이터베이스에 기록 저장
        db_history = models.AnalysisHistory(
            transcript=transcript,
            gyeongsang=final_result.get("gyeongsang", 0.0),
            jeolla=final_result.get("jeolla", 0.0),
            chungcheong=final_result.get("chungcheong", 0.0),
            standard=final_result.get("standard", 0.0)
        )
        db.add(db_history)
        db.commit()
        db.refresh(db_history)
        
        return AnalysisResponse(
            transcript=transcript,
            confidence=0.87,
            result=final_result
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/api/history", response_model=List[HistoryResponse])
def get_history(db: Session = Depends(get_db)):
    # 최신 기록 10개 반환
    histories = db.query(models.AnalysisHistory).order_by(models.AnalysisHistory.created_at.desc()).limit(10).all()
    return histories
