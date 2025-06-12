import torch
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader

# 라벨 재매핑용 래퍼
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, confusion_matrix

# 클래스 정의
class_names = ["Large", "Medium", "Small"]
num_classes = len(class_names)

# 라벨 매핑 (클래스 디렉토리명 기준)
label_map = {
    "apple_fuji_L": 0,  # Large
    "apple_fuji_M": 1,  # Medium
    "apple_fuji_S": 2,  # Small
}


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


# 데이터 경로 및 전처리
test_dir = "./dataset/test"
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
raw_test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_dataset = RemapLabelDataset(raw_test_dataset, label_map)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 로드
weights = MobileNet_V2_Weights.IMAGENET1K_V1
model = mobilenet_v2(weights=weights)
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load("./model/mobilenetv2_model.pt", map_location="cpu"))
model.eval()

# 예측 수행
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# 평가 결과 출력
print("✅ [Test Set Evaluation]")
print(classification_report(all_labels, all_preds, target_names=class_names))
print("🔎 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
