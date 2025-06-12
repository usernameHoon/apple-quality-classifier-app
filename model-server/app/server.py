from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.predict import load_model, predict_image
from io import BytesIO
from PIL import Image, UnidentifiedImageError

app = FastAPI()

# CORS 설정 (React 등 프론트엔드 연동 시)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서버 시작 시 모델 1회 로드
model = load_model()

# 라벨 → 한글 품질 등급 매핑
label_to_korean = {
    "Large": "특", 
    "Medium": "보통", 
    "Small": "보통 이하"
}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 허용 확장자 검사
    allowed_extensions = (".jpg", ".jpeg", ".png")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400, detail="이미지 파일(jpg, jpeg, png)만 업로드 가능합니다."
        )

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="유효한 이미지 파일이 아닙니다.")

    # 예측 수행
    label_en, confidence = predict_image(image, model)
    label_ko = label_to_korean.get(label_en, "알 수 없음")

    return {
        "label": label_en,  # 영문 라벨 (ex: Large)
        "label_korean": label_ko,  # 한글 등급 (ex: 특)
        "confidence": round(confidence, 4),
    }
