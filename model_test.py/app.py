import os
import torch
import timm
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# PATHS
# ======================
SKIN_DETECTOR_PATH = r"C:\Users\nechy\Desktop\skinCancer\skin_detector_v2.pth"
MELANOMA_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "melanoma_best_v3.pth")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

MEL_LABELS = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']


# ======================
# TRANSFORMS
# ======================
skin_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

mel_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ======================
# MODEL LOADING
# ======================
def load_skin_detector(path):
    print("Loading SkinDetector:", path)
    model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=2)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


def load_melanoma_model(path):
    print("Loading Melanoma Model:", path)
    model = timm.create_model("tf_efficientnet_b4", pretrained=False, num_classes=len(MEL_LABELS))
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


skin_detector = load_skin_detector(SKIN_DETECTOR_PATH)
melanoma_model = load_melanoma_model(MELANOMA_MODEL_PATH)


# ======================
# SIMPLE CROP ON CLICK (OLD WORKING LOGIC)
# ======================
CROP_SIZE = 200

def extract_crop_user(pil_image, click):
    if not click:
        return None, "Click on the mole."

    img = np.array(pil_image)
    h, w, _ = img.shape

    x = int(click[0]["x"])
    y = int(click[0]["y"])

    half = CROP_SIZE // 2

    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half)
    y2 = min(h, y + half)

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return None, "Crop out of image bounds."

    return crop, None

def crop_on_click(image, evt: gr.SelectData):
    if image is None:
        return None, ""

    clicks = [{"x": evt.index[0], "y": evt.index[1]}]
    crop, err = extract_crop_user(image, clicks)

    if err:
        return None, err

    crop_pil = Image.fromarray(crop)

    return crop_pil, "Is this crop correct?"




# ======================
# GRAD-CAM
# ======================
def find_target_layer(model):
    for name, layer in reversed(list(model.named_modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    raise RuntimeError("No conv layer found.")


def compute_gradcam(model, input_tensor, class_idx):
    model.zero_grad()
    target = find_target_layer(model)

    activ = {}
    grads = {}

    def forward_hook(m, i, o):
        activ["value"] = o

    def backward_hook(m, gi, go):
        grads["value"] = go[0]

    h1 = target.register_forward_hook(forward_hook)
    h2 = target.register_backward_hook(backward_hook)

    out = model(input_tensor)
    one_hot = torch.zeros_like(out)
    one_hot[0, class_idx] = 1
    out.backward(gradient=one_hot)

    A = activ["value"]
    G = grads["value"]

    weights = G.mean(dim=(2, 3), keepdim=True)
    cam = (weights * A).sum(dim=1)[0]
    cam = torch.relu(cam).detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    h1.remove(); h2.remove()
    return cam


# ======================
# GPT
# ======================
LABEL_MAP = {
    "akiec": "Actinic keratosis / Bowen’s disease (early skin carcinoma)",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma (benign skin nodule)",
    "nv": "Melanocytic nevus (common mole)",
    "mel": "Melanoma (skin cancer)",
    "vasc": "Vascular lesion"
}

def get_gpt_text(label):
    if not client:
        return "GPT explanation unavailable."

    diagnosis = LABEL_MAP.get(label, label)
    prompt = (
        f"Provide a short, medically accurate explanation of '{diagnosis}'. "
        f"Explain what it is, whether it is benign or malignant, and what a dermatologist recommends. "
        f"Keep the tone clear, calm, and non-alarming."
    )

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except:
        return "GPT error."


# ======================
# FULL ANALYSIS (OLD WORKING VERSION)
# ======================
def analyze_click(image, evt: gr.SelectData):
    if image is None:
        return "Upload image first.", None, None

    clicks = [{"x": evt.index[0], "y": evt.index[1]}]
    crop, err = extract_crop_user(image, clicks)
    if err:
        return err, None, None

    crop_pil = Image.fromarray(crop)

    s_in = skin_transform(crop_pil).unsqueeze(0).to(DEVICE)
    sd = torch.softmax(skin_detector(s_in), dim=1)[0][1].item()

    if sd < 0.35:
        return "This does not appear to be skin.", None, None

    m_in = mel_transform(crop_pil).unsqueeze(0).to(DEVICE)
    logits = melanoma_model(m_in)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    idx = int(np.argmax(probs))
    label = MEL_LABELS[idx]
    conf = probs[idx]

    cam = compute_gradcam(melanoma_model, m_in, idx)
    cam_uint8 = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (crop.shape[1], crop.shape[0]))

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(
        crop_rgb, 0.55,
        cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), 0.45, 0
    )

    overlay_pil = Image.fromarray(overlay)
    explanation = get_gpt_text(label)

    result_html = (
        f"<b>Prediction:</b> {label.upper()}<br>"
        f"<b>Confidence:</b> {conf * 100:.1f}%<br>"
        f"<b>Skin probability:</b> {sd * 100:.1f}%"
    )

    return result_html, overlay_pil, explanation

def analyze_with_saved_click(image, clicks):
    if image is None or clicks is None:
        return "No crop selected.", None, None

    x, y = clicks[0]["x"], clicks[0]["y"]

    class DummyEvt:
        def __init__(self, x, y):
            self.index = (x, y)

    evt = DummyEvt(x, y)
    return analyze_click(image, evt)






# ==============================
#  UI (CSS)
# ==============================
custom_css = """
/* ===== GLOBAL FORCE LIGHT ===== */
:root, body, html,
.gradio-container,
#root, #app {
    background: #EEEFFD !important; /* Light blue background */
    --background-fill-primary: #EEEFFD !important;
    --color-text: #000 !important;
    color: #000 !important;
    font-family: "Inter", sans-serif !important;
}

/* ===== Blue Card Style ===== */
.card, .gr-box, .gr-panel, .gr-block, .gr-group {
    background: rgba(91, 101, 220, 0.18) !important; /* soft blue */
    border-radius: 20px !important;
    padding: 22px !important;
    border: 1px solid rgba(91, 101, 220, 0.35) !important;
    backdrop-filter: blur(4px) !important;
    box-shadow: 0 6px 22px rgba(0,0,0,0.15) !important;
    transition: 0.28s ease-in-out;
}

.card:hover {
    box-shadow: 0 10px 26px rgba(0,0,0,0.20) !important;
    transform: translateY(-2px);
}

/* ===== Title ===== */
#main-title {
    font-size: 50px;
    font-weight: 900 !important;   
    text-align: center;
    color: #1A2A6C;
    letter-spacing: 0.5px;         
}


/* Subtitle */
#subtitle {
    color: #000;
    text-align: center;
    opacity: 0.7;
}

/* Red warning */
#warning {
    margin-top: 8px;
    text-align: center;
    color: #D10000;
    font-weight: bold;
}

/* ===== Instructions Box ===== */
#instructions {
    background: rgba(91, 101, 220, 0.18) !important;
    border: 1px solid rgba(91, 101, 220, 0.35) !important;
    padding: 20px;
    width: 520px;
    border-radius: 18px;
    margin: 20px auto;
    color: #000 !important;
    text-align: center;
    box-shadow: 0 6px 22px rgba(0,0,0,0.12) !important;
}
#instructions b {
    color: #5B65DC;
    font-size: 18px;
}

/* ===== Text result area ===== */
#result-text, .gr-html {
    color: #000 !important;
    font-size: 18px;
    line-height: 1.45;
    font-weight: 600;
}

/* ===== Explanation text box ===== */
textarea {
    background: rgba(91, 101, 220, 0.10) !important;
    border: 1px solid rgba(91, 101, 220, 0.40) !important;
    color: #000 !important;
    border-radius: 14px !important;
    padding: 12px !important;
    font-size: 16px !important;
}

/* ===== Image styling ===== */
.gr-image {
    border-radius: 18px !important;
    overflow: hidden !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
    border: 1px solid rgba(91, 101, 220, 0.30) !important;
}

/* ===== Buttons ===== */
button, .gr-button {
    background: #5B65DC !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 12px 20px !important;
    border: none !important;
    transition: 0.2s ease;
    font-weight: 600;
}

button:hover, .gr-button:hover {
    background: #4A55C7 !important;
    transform: translateY(-3px);
}
"""

# ==============================
# GRADIO INTERFACE
# ==============================

click_state = gr.State()
with gr.Blocks(title="Skin Cancer Detector", css=custom_css) as demo:

    # ---- Headers ----
    gr.HTML("<div id='main-title'>Skin Cancer Detector</div>")
    gr.HTML("<div id='subtitle'>AI-based screening tool</div>")
    gr.HTML("<div id='warning'>This is NOT a medical diagnosis. Consult a dermatologist.</div>")

    # ---- Instructions ----
    gr.HTML("""
    <div id="instructions">
        <b>How to take a correct photo:</b><br>
        • Take a close-up (macro) photo<br>
        • Use good lighting or flash<br>
        • Keep the camera steady<br>
        • Ensure only skin is in the frame<br>
    </div>
    """)


    gr.HTML(
        "<div style='text-align:center; font-weight:600; font-size:17px; color:#1A2A6C; margin-top:18px; margin-bottom:10px;'>"
        "Please upload a photo, then click on the mole you would like to analyze."
        "</div>"
    )
    # ---- Layout ----
    with gr.Row():

        # Upload Section
        with gr.Column(scale=1):
            img_in = gr.Image(
                type="pil",
                label="Upload Photo",
                elem_classes="card"
            )


        # Output Section
        with gr.Column(scale=1):
            out_text = gr.HTML(elem_id="result-text", elem_classes="card fade-in")
            out_cam = gr.Image(label="Attention Map (Grad-CAM)", elem_classes="card fade-in")
            out_gpt = gr.Textbox(label="Explanation", lines=6, elem_classes="card fade-in")
    crop_preview = gr.Image(label="Cropped area", elem_classes="card")
    confirm_text = gr.Markdown(elem_classes="card")

    with gr.Row():
        btn_yes = gr.Button("✅ Yes, analyze")
        btn_no = gr.Button("❌ No, recrop")


    # ---- Click handler ----
    img_in.select(
        crop_on_click,
        inputs=[img_in],
        outputs=[crop_preview, confirm_text]
    )
    img_in.select(
        analyze_click,
        inputs=[img_in],
        outputs=[out_text, out_cam, out_gpt]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)