import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm

# 라벨 재매핑 클래스
class RemapLabelDataset(Dataset):
    def __init__(self, dataset, label_map):
        self.dataset = dataset
        self.label_map = label_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        class_name = self.dataset.classes[label]
        new_label = self.label_map[class_name]
        return image, new_label

def train():
    # 디렉토리 경로
    train_dir = './dataset/train'
    valid_dir = './dataset/valid'
    model_path = './model/mobilenetv2_model.pt'

    # 라벨 이름 → 숫자 라벨 매핑
    label_map = {
        'apple_fuji_L': 0,  # 특/상
        'apple_fuji_M': 1,  # 보통
        'apple_fuji_S': 2   # 보통 이하
    }

    # 데이터 전처리
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_valid = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 원본 데이터셋
    raw_train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    raw_valid_dataset = datasets.ImageFolder(valid_dir, transform=transform_valid)

    # 라벨 재매핑 적용
    train_dataset = RemapLabelDataset(raw_train_dataset, label_map)
    valid_dataset = RemapLabelDataset(raw_valid_dataset, label_map)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2)

    num_classes = 3

    # 모델 정의
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_losses = []
    val_accuracies = []

    for epoch in range(10):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/10", unit="batch")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

        # 검증
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        val_accuracies.append(acc)
        print(f"Validation Accuracy: {acc:.2%}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✅ 모델 state_dict 저장 완료: {model_path}")

    # 학습 과정 시각화
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, marker='o', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig('./model/training_plot.png')
    plt.show()

if __name__ == '__main__':
    train()
