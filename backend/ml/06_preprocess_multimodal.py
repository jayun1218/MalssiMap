import os
import csv
import librosa
import numpy as np
import torch
import json
from sklearn.model_selection import GroupShuffleSplit

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_mel_spectrogram(file_path, sr=22050, n_mels=128, max_len=130):
    y, sr = librosa.load(file_path, sr=sr)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    
    if melspec_db.shape[1] > max_len:
        melspec_db = melspec_db[:, :max_len]
    else:
        pad_width = max_len - melspec_db.shape[1]
        melspec_db = np.pad(melspec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return melspec_db

def build_vocab(transcripts):
    """간단한 글자(Character) 단위 단어장 생성"""
    chars = set()
    for text in transcripts:
        for ch in text:
            chars.add(ch)
    # 0은 패딩용 (PAD)
    char2idx = {ch: i+1 for i, ch in enumerate(sorted(list(chars)))}
    return char2idx

def text_to_tensor(text, char2idx, max_len=20):
    """텍스트를 정수 인덱스 리스트로 변환 및 패딩"""
    encoded = [char2idx.get(ch, 0) for ch in text]
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded = encoded + [0] * (max_len - len(encoded))
    return encoded

def main():
    print("🚀 [Multimodal 1/2] 오디오 & 텍스트 동시 전처리 시작...")
    
    raw_data_dir = os.path.join("..", "data", "raw")
    processed_data_dir = os.path.join("..", "data", "processed")
    ensure_dir(processed_data_dir)
    
    csv_filepath = os.path.join(raw_data_dir, "metadata.csv")
    
    if not os.path.exists(csv_filepath):
        print("❌ metadata.csv 파일을 찾을 수 없습니다. 01_generate_dummy.py를 먼저 실행하세요.")
        return

    label_map = {"gyeongsang": 0, "jeolla": 1, "chungcheong": 2, "standard": 3}
    
    audio_features = []
    texts = []
    labels = []
    groups = []

    # 1. 파일 읽기
    with open(csv_filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            speaker_id = row["speaker_id"]
            region = row["region"]
            transcript = row.get("transcript", "")
            
            file_path = os.path.join(raw_data_dir, filename)
            mel_spec = extract_mel_spectrogram(file_path)
            
            audio_features.append(mel_spec)
            texts.append(transcript)
            labels.append(label_map[region])
            groups.append(speaker_id)
            
    # 2. 텍스트 전처리 (Vocab 생성)
    char2idx = build_vocab(texts)
    # Vocab 저장 (API 서버에서 추론할 때 써야 하므로)
    vocab_path = os.path.join("..", "models", "vocab.json")
    ensure_dir(os.path.dirname(vocab_path))
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(char2idx, f, ensure_ascii=False)
        
    text_features = [text_to_tensor(t, char2idx) for t in texts]

    # NumPy 변환
    X_audio = np.array(audio_features)
    X_audio = np.expand_dims(X_audio, axis=1) # Channel 추가
    X_text = np.array(text_features)
    y = np.array(labels)
    groups = np.array(groups)
    
    # 3. 화자 독립 분할
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_audio, y, groups))
    
    # 4. PyTorch Tensor 변환 및 저장
    torch.save(torch.tensor(X_audio[train_idx], dtype=torch.float32), os.path.join(processed_data_dir, "X_audio_train.pt"))
    torch.save(torch.tensor(X_audio[test_idx], dtype=torch.float32), os.path.join(processed_data_dir, "X_audio_test.pt"))
    
    torch.save(torch.tensor(X_text[train_idx], dtype=torch.long), os.path.join(processed_data_dir, "X_text_train.pt"))
    torch.save(torch.tensor(X_text[test_idx], dtype=torch.long), os.path.join(processed_data_dir, "X_text_test.pt"))
    
    torch.save(torch.tensor(y[train_idx], dtype=torch.long), os.path.join(processed_data_dir, "y_train_mm.pt"))
    torch.save(torch.tensor(y[test_idx], dtype=torch.long), os.path.join(processed_data_dir, "y_test_mm.pt"))
    
    print(f"✅ 오디오 및 텍스트 텐서 전처리 완료!")
    print(f"Vocab 크기: {len(char2idx)} (저장됨: {vocab_path})")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
