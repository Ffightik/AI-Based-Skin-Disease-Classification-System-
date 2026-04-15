# eval_and_plot.py
import torch, torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import os, numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

MODEL_PATH = "skin_detector_v2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Fill these with sample paths from your val set
val_dir = "skin_detector_dataset/val"
true_labels=[]
preds=[]
paths=[]
for cls_idx, cls in enumerate(["non_skin","skin"]):
    folder = os.path.join(val_dir, cls)
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(x)
            prob = F.softmax(out, dim=1)[0].cpu().numpy()
        pred = int(prob[1] > 0.5)
        true_labels.append(cls_idx)
        preds.append(pred)
        paths.append(p)

print(classification_report(true_labels, preds, target_names=["non_skin","skin"]))
cm = confusion_matrix(true_labels, preds)
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks([0,1],["non_skin","skin"])
plt.yticks([0,1],["non_skin","skin"])
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Confusion matrix")
plt.show()
