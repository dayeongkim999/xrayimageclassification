# 라이브러리 로드
from urllib.request import urlopen
from PIL import Image
import torch
import timm

# 이미지 로드
img = Image.open(
    urlopen("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png")
)

# 1) 모델 불러오기 (pretrained=True 권장)
model = timm.create_model('densenet121.ra_in1k', pretrained=True)
model.eval()

# 2) 모델에 맞는 transform 자동 생성
cfg = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**cfg, is_training=False)

# 3) 예측
tensor = transform(img).unsqueeze(0)   # (1, 3, 288, 288)
output = model(tensor)

prob, idx = torch.topk(output.softmax(dim=1), 5)
print(prob, idx)

# densenet
# 라이브러리 로드
import torch, timm, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ----------
# 1. 데이터셋 구성
# ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 30
IMG_SIZE = 224
LR = 1e-4
NUM_EPOCHS = 100

transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # X-ray: 1ch → 3ch
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # Horizontal/Vertical flipping: Yes
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),

    # Rotation Range 0–360°, Zoom 5%, Shift 5%
    transforms.RandomAffine(
        degrees=360,          # 회전 범위 ≈ 0~360도
        translate=(0.05, 0.05), # 가로/세로 최대 5% 평행이동
        scale=(0.95, 1.05),     # 5% 내외 줌
    ),

    # Re-scaling 1/255
    transforms.ToTensor(),     # 이미지를 [0,1] 범위 (1/255)로 바꿔줌
])

transform_eval = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),   # 1/255
])

train_data = datasets.ImageFolder("datasets/chest_xray/chest_xray/train", transform=transform_train)
val_data   = datasets.ImageFolder("datasets/chest_xray/chest_xray/val",   transform=transform_eval)
test_data  = datasets.ImageFolder("datasets/chest_xray/chest_xray/test",  transform=transform_eval)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

# ----------
# 2. DenseNet121 모델 생성
# ----------
model = timm.create_model("densenet121.ra_in1k", pretrained=True, num_classes=2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.NAdam(model.parameters(), lr=LR)

scaler = GradScaler(enabled=True)  # AMP 자동

for epoch in range(100):  # 논문은 100 Epoch
    model.train()
    train_loss, train_correct, total = 0,0,0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/100", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        with autocast():  # ⬅ Mixed Precision (빠름 + VRAM 절약)
            preds = model(imgs)
            loss = criterion(preds, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_correct += (preds.argmax(1)==labels).sum().item()
        total += labels.size(0)

    train_acc = train_correct/total

    # ────────────────────── Validation ──────────────────────
    model.eval()
    val_loss, val_correct, val_total = 0,0,0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, labels)
            val_loss += loss.item()
            val_correct += (preds.argmax(1)==labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    # 📌 로그 출력
    print(f"[Epoch {epoch+1:03d}] "
          f"Train Loss={train_loss/len(train_loader):.4f} | "
          f"Val Loss={val_loss/len(val_loader):.4f} | "
          f"Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}\n")

    # ────────────────────── Early Stop ──────────────────────
    IF_ES = True  # 끄려면 False
    if IF_ES:
        if epoch>5 and val_loss > prev_loss: patience -=1
        else: patience = 3
        if patience == 0: print("🛑 Early stop"); break
        prev_loss = val_loss

MODEL_NAME = "densenet121_xray"

torch.save(model.state_dict(), f"{MODEL_NAME}.pth")
print("📦 Model Saved →", MODEL_NAME+".pth")
