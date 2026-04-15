import torch
from torchvision import transforms
from PIL import Image
import timm
import os
from dotenv import load_dotenv
import os
import openai

from openai import OpenAI
from dotenv import load_dotenv
import os

# Загружаем ключ из .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Проверяем
if not api_key:
    raise ValueError("❌ API key not found! Make sure your .env file contains OPENAI_API_KEY.")

# Создаём клиента
client = OpenAI(api_key=api_key)



# ======= PATHS =======
DATA_DIR = r"C:\Users\nechy\Desktop\skinCancer\dataset\archive\all_images"
MODEL_PATH = r"C:\Users\nechy\Desktop\skinCancer\melanoma_best_v3.pth"

# ======= DEVICE =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======= CLASSES =======
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# ======= TRANSFORMS =======
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======= LOAD MODEL (EfficientNet-B4 from timm) =======
model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=len(classes))

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    print("✅ Model loaded successfully on", device)
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# ======= LABEL DESCRIPTIONS =======
label_descriptions = {
    "nv": "Melanocytic nevi — benign mole. Usually harmless, but monitor for changes.",
    "mel": "Melanoma — malignant skin cancer. Requires immediate medical attention.",
    "bkl": "Benign keratosis-like lesion — non-cancerous thickened skin (e.g., seborrheic keratosis).",
    "bcc": "Basal cell carcinoma — a slow-growing type of skin cancer that requires treatment.",
    "akiec": "Actinic keratosis / carcinoma in situ — precancerous lesion. Should be evaluated by a dermatologist.",
    "df": "Dermatofibroma — benign fibrous skin nodule. Harmless.",
    "vasc": "Vascular lesion — benign vascular growth (e.g., hemangioma)."
}

# ======= PREDICTION FUNCTION =======
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_label = classes[predicted.item()]
        probability = torch.softmax(outputs, dim=1)[0][predicted.item()].item() * 100

    description = label_descriptions.get(predicted_label, "No description available for this class.")

    print(f"\nPrediction for '{os.path.basename(image_path)}': {predicted_label}")
    print(f"{description}")
    print(f"Model confidence: {probability:.2f}%")

    return predicted_label, probability, description

# ======= TEST IMAGE =======
test_image = os.path.join(DATA_DIR, "ISIC_0024491.jpg")

if os.path.exists(test_image):
    predict_image(test_image)
else:
    print(f"❌ Test image not found: {test_image}")

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful AI."},
        {"role": "user", "content": "Explain this diagnosis: melanoma"}
    ]
)
print(response.choices[0].message.content)




