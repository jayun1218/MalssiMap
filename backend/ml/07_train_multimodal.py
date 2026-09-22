import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class MultimodalFusion(nn.Module):
    def __init__(self, vocab_size, num_classes=4):
        super(MultimodalFusion, self).__init__()
        
        # 1. Audio Branch (CNN)
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
        
        # 2. Text Branch (Embedding + Average Pooling)
        # 패딩 인덱스는 0이므로 padding_idx=0 설정
        self.text_emb = nn.Embedding(vocab_size, 16, padding_idx=0)
        self.text_fc = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        # 3. Fusion Classifier (Audio 32 + Text 16 = 48)
        self.classifier = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, audio, text):
        # Audio 처리
        a = self.audio_net(audio)
        a = a.view(a.size(0), -1)
        a_feat = self.audio_fc(a)
        
        # Text 처리
        t = self.text_emb(text) # [Batch, SeqLen, 16]
        # 시퀀스 길이 방향으로 평균 (Average Pooling)
        t = torch.mean(t, dim=1) # [Batch, 16]
        t_feat = self.text_fc(t)
        
        # Fusion
        fused = torch.cat((a_feat, t_feat), dim=1)
        out = self.classifier(fused)
        return out

def main():
    print("🚀 [Multimodal 2/2] 퓨전 모델 학습 시작...")
    
    processed_data_dir = os.path.join("..", "data", "processed")
    models_dir = os.path.join("..", "models")
    vocab_path = os.path.join(models_dir, "vocab.json")
    ensure_dir(models_dir)
    
    # Vocab Size 로드
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab) + 1 # 패딩 0 포함
    
    # 텐서 로드
    try:
        X_audio_train = torch.load(os.path.join(processed_data_dir, "X_audio_train.pt"), weights_only=True)
        X_audio_test = torch.load(os.path.join(processed_data_dir, "X_audio_test.pt"), weights_only=True)
        X_text_train = torch.load(os.path.join(processed_data_dir, "X_text_train.pt"), weights_only=True)
        X_text_test = torch.load(os.path.join(processed_data_dir, "X_text_test.pt"), weights_only=True)
        y_train = torch.load(os.path.join(processed_data_dir, "y_train_mm.pt"), weights_only=True)
        y_test = torch.load(os.path.join(processed_data_dir, "y_test_mm.pt"), weights_only=True)
    except FileNotFoundError:
        print("❌ 텐서 파일을 찾을 수 없습니다. 06_preprocess_multimodal.py를 먼저 실행하세요.")
        return

    # DataLoader 생성
    train_dataset = TensorDataset(X_audio_train, X_text_train, y_train)
    test_dataset = TensorDataset(X_audio_test, X_text_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalFusion(vocab_size=vocab_size, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    print(f"디바이스: {device}, 에포크: {epochs}")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for audios, texts, labels in train_loader:
            audios, texts, labels = audios.to(device), texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(audios, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * audios.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
    # 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for audios, texts, labels in test_loader:
            audios, texts, labels = audios.to(device), texts.to(device), labels.to(device)
            outputs = model(audios, texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    print(f"\n📊 멀티모달 모델 평가 결과 Accuracy: {acc:.2f}%")
    
    # 모델 저장
    model_path = os.path.join(models_dir, "dialect_multimodal_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ 멀티모달 딥러닝 모델 저장 완료: {os.path.abspath(model_path)}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
