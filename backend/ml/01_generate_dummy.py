import os
import csv
import numpy as np
import soundfile as sf
import random

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_dummy_audio(filepath, region, duration=2.0, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq_map = {
        "gyeongsang": 440.0,
        "jeolla": 330.0,
        "chungcheong": 261.63,
        "standard": 523.25
    }
    base_freq = freq_map.get(region, 440.0)
    audio = 0.5 * np.sin(2 * np.pi * base_freq * t)
    noise = np.random.normal(0, 0.1, len(t))
    audio_data = audio + noise
    sf.write(filepath, audio_data, sr)

def get_dummy_text(region):
    texts = {
        "gyeongsang": ["밥 묵었나", "머라카노", "와 그라노", "진짜가"],
        "jeolla": ["아따 워메", "참말로 그라제", "거시기 하구만", "아니랑께"],
        "chungcheong": ["아녀유", "그려유", "뭐하슈", "괜찮아유"],
        "standard": ["안녕하세요", "밥 먹었어요", "정말인가요", "아닙니다"]
    }
    return random.choice(texts[region])

def main():
    print("🚀 [1/3] 더미 데이터(오디오+텍스트) 생성 시작...")
    
    raw_data_dir = os.path.join("..", "data", "raw")
    ensure_dir(raw_data_dir)
    
    regions = ["gyeongsang", "jeolla", "chungcheong", "standard"]
    num_samples_per_region = 10
    
    csv_filepath = os.path.join(raw_data_dir, "metadata.csv")
    
    with open(csv_filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "speaker_id", "region", "transcript"])
        
        sample_count = 0
        for region in regions:
            for i in range(num_samples_per_region):
                filename = f"{region}_{i:03d}.wav"
                filepath = os.path.join(raw_data_dir, filename)
                speaker_id = f"speaker_{region}_{i // 2:02d}"
                transcript = get_dummy_text(region)
                
                generate_dummy_audio(filepath, region)
                writer.writerow([filename, speaker_id, region, transcript])
                sample_count += 1
                
    print(f"✅ 더미 데이터 {sample_count}개 및 metadata.csv(transcript 포함) 생성 완료!")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
