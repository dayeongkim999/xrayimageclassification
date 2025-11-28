import torch, timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

# ================================
# 1) GPU / 모델 로드
# ================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "densenet121_xray.pth"

model = timm.create_model("densenet121.ra_in1k", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_NAME, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("🚀 Model Loaded Successfully")

# ================================
# 2) Test Dataset 로드
# ================================

# 평가용 transform (train augmentation 절대 쓰면 안됨)
transform_eval = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# dataset 경로 반드시 존재해야 함
test_path = "datasets/chest_xray/chest_xray/test"

test_data = datasets.ImageFolder(test_path, transform=transform_eval)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"🗂 Test Data Loaded: {len(test_data)} images")

# ================================
# 3) Inference + 성능 평가
# ================================
preds = []
probs = []
labels_true = []

with torch.inference_mode():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        
        output = model(imgs)
        preds.extend(output.argmax(1).cpu().numpy())                      # class 예측
        probs.extend(torch.softmax(output, dim=1)[:,1].cpu().numpy())    # pneumonia 확률
        labels_true.extend(labels.numpy())

# classification report / confusion matrix 저장
report = classification_report(labels_true, preds, target_names=["NORMAL","PNEUMONIA"])
cm = confusion_matrix(labels_true, preds)

with open("evaluation_report.txt", "w") as f:
    f.write("=== DenseNet121 Chest X-ray Evaluation ===\n\n")
    f.write(report + "\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm) + "\n")

print("📄 Report Saved → evaluation_report.txt")

# ================================
# 4) ROC Curve + AUC 저장
# ================================
fpr, tpr, _ = roc_curve(labels_true, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color="red")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve — DenseNet121")
plt.legend()
plt.savefig("roc_curve.png", dpi=300)
plt.show()

print(f"🔥 ROC AUC = {roc_auc:.4f} (Saved as roc_curve.png)")

# ================================
# 5) PR Curve 저장
# ================================
precision, recall, _ = precision_recall_curve(labels_true, probs)

plt.figure(figsize=(6,6))
plt.plot(recall, precision, color="blue")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — DenseNet121")
plt.savefig("pr_curve.png", dpi=300)
plt.show()

print("📊 PR Curve saved → pr_curve.png")
print("🎉 전체 평가 완료!")

