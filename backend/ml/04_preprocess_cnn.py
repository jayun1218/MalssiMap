import os
import csv
import librosa
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_mel_spectrogram(file_path, sr=22050, n_mels=128, max_len=130):
    """오디오에서 Mel-Spectrogram을 추출하고 고정된 길이(max_len)로 패딩/자르기"""
    y, sr = librosa.load(file_path, sr=sr)
    # Mel-Spectrogram 추출
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    # dB 스케일 변환 (이미지처럼 다루기 위해)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    
    # 시간축 길이 맞추기 (max_len 프레임)
    if melspec_db.shape[1] > max_len:
        melspec_db = melspec_db[:, :max_len]
    else:
        pad_width = max_len - melspec_db.shape[1]
        melspec_db = np.pad(melspec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    return melspec_db

def main():
    print("🚀 [CNN 1/2] Mel-Spectrogram 추출 및 텐서 변환 시작...")
    
    raw_data_dir = os.path.join("..", "data", "raw")
    processed_data_dir = os.path.join("..", "data", "processed")
    ensure_dir(processed_data_dir)
    
    csv_filepath = os.path.join(raw_data_dir, "metadata.csv")
    
    if not os.path.exists(csv_filepath):
        print("❌ metadata.csv 파일을 찾을 수 없습니다. 01_generate_dummy.py를 먼저 실행하세요.")
        return

    # 라벨을 정수로 매핑
    label_map = {
        "gyeongsang": 0,
        "jeolla": 1,
        "chungcheong": 2,
        "standard": 3
    }
    
    features = []
    labels = []
    groups = []

    print("오디오 파일에서 Mel-Spectrogram 추출 중...")
    with open(csv_filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            speaker_id = row["speaker_id"]
            region = row["region"]
            
            file_path = os.path.join(raw_data_dir, filename)
            
            mel_spec = extract_mel_spectrogram(file_path)
            
            features.append(mel_spec)
            labels.append(label_map[region])
            groups.append(speaker_id)
            
    X = np.array(features)
    y = np.array(labels)
    groups = np.array(groups)
    
    # 채널(Channel) 차원 추가 (PyTorch CNN은 [Batch, Channel, Height, Width] 형태 요구)
    # 현재 X는 [Batch, n_mels, max_len]이므로 [Batch, 1, n_mels, max_len]으로 변환
    X = np.expand_dims(X, axis=1)
    
    # 화자 독립 분할
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # NumPy 배열을 PyTorch Tensor로 변환
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # 저장
    torch.save(X_train_tensor, os.path.join(processed_data_dir, "X_train_cnn.pt"))
    torch.save(X_test_tensor, os.path.join(processed_data_dir, "X_test_cnn.pt"))
    torch.save(y_train_tensor, os.path.join(processed_data_dir, "y_train_cnn.pt"))
    torch.save(y_test_tensor, os.path.join(processed_data_dir, "y_test_cnn.pt"))
    
    print(f"✅ 전처리 완료! 텐서 Shape: {X_train_tensor.shape}")
    print(f"저장 위치: {os.path.abspath(processed_data_dir)}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
