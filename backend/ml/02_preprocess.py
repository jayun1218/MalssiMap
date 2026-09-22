import os
import csv
import librosa
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_mfcc(file_path, sr=22050, n_mfcc=13):
    """오디오 파일에서 MFCC 특징 벡터의 평균을 추출합니다."""
    # 음성 파일 로드
    y, sr = librosa.load(file_path, sr=sr)
    # MFCC 추출
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # 시간축 기준 평균 계산 (단일 벡터로 축소)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

def main():
    print("🚀 [2/3] 데이터 전처리 및 화자 독립(Speaker Split) 데이터셋 분할 시작...")
    
    raw_data_dir = os.path.join("..", "data", "raw")
    processed_data_dir = os.path.join("..", "data", "processed")
    ensure_dir(processed_data_dir)
    
    csv_filepath = os.path.join(raw_data_dir, "metadata.csv")
    
    if not os.path.exists(csv_filepath):
        print("❌ metadata.csv 파일을 찾을 수 없습니다. 01_generate_dummy.py를 먼저 실행하세요.")
        return

    features = []
    labels = []
    groups = [] # 화자 ID 보관 (GroupShuffleSplit 용)

    # 1. 데이터 로드 및 특징 추출
    print("오디오 파일에서 MFCC 추출 중...")
    with open(csv_filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            speaker_id = row["speaker_id"]
            region = row["region"]
            
            file_path = os.path.join(raw_data_dir, filename)
            
            # MFCC 추출
            mfcc_vector = extract_mfcc(file_path)
            
            features.append(mfcc_vector)
            labels.append(region)
            groups.append(speaker_id)
            
    X = np.array(features)
    y = np.array(labels)
    groups = np.array(groups)
    
    # 2. 화자 독립(Speaker Independent) 데이터셋 분할
    # 같은 화자(speaker_id)의 데이터가 Train과 Test에 섞이지 않도록 분할합니다.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 3. 전처리된 numpy 배열 저장
    np.save(os.path.join(processed_data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_data_dir, "y_test.npy"), y_test)
    
    print(f"✅ 전처리 완료! (Train: {len(X_train)}개, Test: {len(X_test)}개)")
    print(f"저장 위치: {os.path.abspath(processed_data_dir)}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
