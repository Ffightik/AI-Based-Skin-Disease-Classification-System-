import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import timm
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥 Using device:", device)

# === Настройки ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(os.path.dirname(BASE_DIR), "2.-Suspicious-irritated-mole-found-not-to-be-melanoma(200W).png")
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "melanoma_best_v3.pth")


print("🔍 Model path:", MODEL_PATH)
print("🔍 Image path:", IMAGE_PATH)
print("📁 Current working directory:", os.getcwd())

LABELS = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Загрузка модели ===
model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=len(LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === Преобразование изображения ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

image = Image.open(IMAGE_PATH).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

# === Визуализация вероятностей классов ===
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    pred_index = np.argmax(probs)
    confidence = probs[pred_index] * 100

print(f"\n🧠 Predicted class: {LABELS[pred_index]} ({confidence:.2f}%)\n")
print("📊 Probabilities:")
for i, p in enumerate(probs):
    print(f" - {LABELS[i]}: {p * 100:.2f}%")

# === Визуализация Grad-CAM ===
def gradcam(model, image_tensor, target_layer='blocks.6'):
    features = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    target = dict(model.named_modules())[target_layer]
    target.register_forward_hook(forward_hook)
    target.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, pred_class] = 1
    output.backward(gradient=one_hot)

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    grad_cam = torch.relu((weights * features).sum(dim=1)).squeeze().detach().cpu().numpy()

    grad_cam = cv2.resize(grad_cam, (224, 224))
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
    return grad_cam

# Получаем карту внимания
cam = gradcam(model, input_tensor)

# === Визуализация ===
plt.figure(figsize=(12, 6))

# 1️⃣ Исходное изображение
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# 2️⃣ Grad-CAM тепловая карта
plt.subplot(1, 3, 2)
plt.imshow(cam, cmap='jet')
plt.title("Grad-CAM Heatmap")
plt.axis("off")

# 3️⃣ Наложение на оригинал
plt.subplot(1, 3, 3)
overlay = np.array(image.resize((224, 224))) / 255
overlay = 0.6 * overlay + 0.4 * plt.cm.jet(cam)[..., :3]
plt.imshow(overlay)
plt.title("Model Attention Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()

# === Анализ внутренних признаков (feature map) ===
def visualize_feature_maps(model, image_tensor, layer_name='blocks.3'):
    """Показывает, что извлекает модель внутри."""
    activation = {}
    def hook_fn(module, input, output):
        activation[layer_name] = output.detach()

    layer = dict(model.named_modules())[layer_name]
    layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(image_tensor)

    fmap = activation[layer_name][0].cpu()
    num_features = min(12, fmap.shape[0])  # Покажем первые 12 карт

    plt.figure(figsize=(12, 6))
    for i in range(num_features):
        plt.subplot(3, 4, i + 1)
        plt.imshow(fmap[i], cmap='viridis')
        plt.axis("off")
    plt.suptitle(f"Feature Maps from Layer: {layer_name}")
    plt.tight_layout()
    plt.show()

# Вызов функции
visualize_feature_maps(model, input_tensor)
