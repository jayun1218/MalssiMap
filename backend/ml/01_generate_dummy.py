import os
import csv
import numpy as np
import soundfile as sf

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_dummy_audio(filepath, region, duration=2.0, sr=22050):
    """
    각 지역별로 미세하게 다른 주파수 대역의 사인파(Sine wave)와 노이즈를 섞어
    가상의(더미) 방언 음성 데이터를 생성합니다.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 지역별 더미 특징 (기본 주파수 변경)
    freq_map = {
        "gyeongsang": 440.0,    # A4
        "jeolla": 330.0,        # E4
        "chungcheong": 261.63,  # C4
        "standard": 523.25      # C5
    }
    base_freq = freq_map.get(region, 440.0)
    
    # 사인파 + 약간의 랜덤 노이즈
    audio = 0.5 * np.sin(2 * np.pi * base_freq * t)
    noise = np.random.normal(0, 0.1, len(t))
    audio_data = audio + noise
    
    # 16-bit PCM WAV 형식으로 저장
    sf.write(filepath, audio_data, sr)

def main():
    print("🚀 [1/3] 더미 데이터 생성 시작...")
    
    raw_data_dir = os.path.join("..", "data", "raw")
    ensure_dir(raw_data_dir)
    
    regions = ["gyeongsang", "jeolla", "chungcheong", "standard"]
    num_samples_per_region = 10 # 지역당 10개 (총 40개 샘플)
    
    csv_filepath = os.path.join(raw_data_dir, "metadata.csv")
    
    with open(csv_filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "speaker_id", "region"])
        
        sample_count = 0
        for region in regions:
            for i in range(num_samples_per_region):
                filename = f"{region}_{i:03d}.wav"
                filepath = os.path.join(raw_data_dir, filename)
                
                # 화자 독립성을 테스트하기 위해 화자 ID를 2개씩 묶음 (총 20명 화자)
                speaker_id = f"speaker_{region}_{i // 2:02d}"
                
                generate_dummy_audio(filepath, region)
                writer.writerow([filename, speaker_id, region])
                sample_count += 1
                
    print(f"✅ 더미 음성 데이터 {sample_count}개 및 metadata.csv 생성 완료!")
    print(f"저장 위치: {os.path.abspath(raw_data_dir)}")

if __name__ == "__main__":
    # backend 디렉토리에서 실행된다고 가정하고 경로 조정 (또는 현재 스크립트 위치 기준)
    # 현재 파일의 위치(backend/ml)를 기준으로 상위 폴더(backend) 설정
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
