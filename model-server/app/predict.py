import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import sys
import os

# 클래스 라벨 정의 (라벨 인덱스: 0 → Large, 1 → Medium, 2 → Small)
class_names = ["Large", "Medium", "Small"]

# 전처리 정의 (학습과 동일해야 함)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


# 모델 로드 함수
def load_model(model_path="./model/mobilenetv2_model.pt"):
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


# 이미지 파일 경로 기반 예측
def predict(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = class_names[pred.item()]
    confidence = conf.item()
    return label, confidence


# PIL.Image 객체 기반 예측
def predict_image(image, model):
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return class_names[pred.item()], conf.item()


# CLI 실행을 위한 main()
def main():
    if len(sys.argv) < 2:
        print("❗ 사용법: python predict.py 이미지경로")
        return

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❗ 이미지 파일을 찾을 수 없습니다: {image_path}")
        return

    model = load_model()
    label, confidence = predict(image_path, model)
    print(f"✅ 예측 결과: {label} ({confidence * 100:.2f}%)")


if __name__ == "__main__":
    main()
