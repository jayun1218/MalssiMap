import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    print("🚀 [3/3] 베이스라인 모델(Random Forest) 학습 시작...")
    
    processed_data_dir = os.path.join("..", "data", "processed")
    models_dir = os.path.join("..", "models")
    ensure_dir(models_dir)
    
    # 1. 전처리된 데이터 로드
    try:
        X_train = np.load(os.path.join(processed_data_dir, "X_train.npy"))
        X_test = np.load(os.path.join(processed_data_dir, "X_test.npy"))
        y_train = np.load(os.path.join(processed_data_dir, "y_train.npy"))
        y_test = np.load(os.path.join(processed_data_dir, "y_test.npy"))
    except FileNotFoundError:
        print("❌ 전처리된 npy 파일을 찾을 수 없습니다. 02_preprocess.py를 먼저 실행하세요.")
        return

    # 2. Random Forest 모델 초기화 및 학습
    print("모델 학습 중...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 3. 모델 평가
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n📊 모델 평가 결과:")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 4. 모델 저장
    model_path = os.path.join(models_dir, "dialect_rf_model.pkl")
    joblib.dump(rf_model, model_path)
    
    print(f"✅ 베이스라인 모델 학습 완료 및 저장됨: {os.path.abspath(model_path)}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
