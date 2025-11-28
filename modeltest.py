import torch, timm

MODEL_NAME = "densenet121_xray.pth"

model = timm.create_model("densenet121.ra_in1k", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_NAME, map_location="cuda"))
model.to("cuda")
model.eval()

print("🚀 Model Loaded Successfully")

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

model.eval()
preds_list = []
labels_list = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to("cuda")
        output = model(imgs).argmax(1).cpu()

        preds_list.extend(output.numpy())
        labels_list.extend(labels.numpy())

report = classification_report(labels_list, preds_list, target_names=["NORMAL","PNEUMONIA"])
cm = confusion_matrix(labels_list, preds_list)

with open("evaluation_report.txt", "w") as f:
    f.write("=== DenseNet121 ChestXray Evaluation ===\n\n")
    f.write(report + "\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm) + "\n")

print("📄 결과 저장 완료 → evaluation_report.txt")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# 이진분류는 확률 기반 Score 필요 → softmax로 확률 반환
model.eval()
probs = []
labels_true = []

with torch.inference_mode():
    for imgs, labels in test_loader:
        imgs = imgs.to("cuda")

        output = model(imgs)              # (batch,2)
        prob = torch.softmax(output, dim=1)[:,1].cpu()   # pneumonia 확률만 추출
        probs.extend(prob.numpy())
        labels_true.extend(labels.numpy())

# ROC Curve
fpr, tpr, thresholds = roc_curve(labels_true, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color="red")
plt.plot([0,1], [0,1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve — DenseNet121 Chest X-ray")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png", dpi=300)
plt.show()

print(f"🔥 ROC AUC Score = {roc_auc:.4f}")

precision, recall, _ = precision_recall_curve(labels_true, probs)

plt.figure(figsize=(6,6))
plt.plot(recall, precision, color="blue")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — DenseNet121")
plt.savefig("pr_curve.png", dpi=300)
plt.show()
