import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# CNN 모델 정의
class DialectCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(DialectCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # 시간과 주파수 축을 1x1로 압축
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x

def main():
    print("🚀 [CNN 2/2] PyTorch CNN 모델 학습 시작...")
    
    processed_data_dir = os.path.join("..", "data", "processed")
    models_dir = os.path.join("..", "models")
    ensure_dir(models_dir)
    
    # 1. 텐서 데이터 로드
    try:
        X_train = torch.load(os.path.join(processed_data_dir, "X_train_cnn.pt"), weights_only=True)
        X_test = torch.load(os.path.join(processed_data_dir, "X_test_cnn.pt"), weights_only=True)
        y_train = torch.load(os.path.join(processed_data_dir, "y_train_cnn.pt"), weights_only=True)
        y_test = torch.load(os.path.join(processed_data_dir, "y_test_cnn.pt"), weights_only=True)
    except FileNotFoundError:
        print("❌ 텐서 파일을 찾을 수 없습니다. 04_preprocess_cnn.py를 먼저 실행하세요.")
        return

    # 2. DataLoader 생성
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 3. 모델, 손실함수, 옵티마이저 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DialectCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 학습 루프
    epochs = 10
    print(f"디바이스: {device}, 에포크: {epochs}")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
    # 5. 모델 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    print(f"\n📊 CNN 모델 평가 결과 Accuracy: {acc:.2f}%")
    
    # 6. 모델 저장 (가중치만 저장하는 것이 권장됨)
    model_path = os.path.join(models_dir, "dialect_cnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ 딥러닝 모델 저장 완료: {os.path.abspath(model_path)}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
