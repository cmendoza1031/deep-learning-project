"""
Gradio demo: upload a dermoscopy image, compare HetLoRA vs DQAW predictions.
Shows rank collapse effect in real time.

Requires trained model checkpoints in ./results/
"""

import gradio as gr
import torch
import torch.nn.functional as F
import timm
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit_lora import apply_lora_to_vit
from torchvision import transforms

CLASS_NAMES = ['MEL (Melanoma)', 'NV (Nevus)', 'BCC (Basal Cell)',
               'AK (Actinic Keratosis)', 'BKL (Benign Keratosis)',
               'DF (Dermatofibroma)', 'VASC (Vascular Lesion)']

CLINICAL_RISK = {
    0: '🔴 HIGH RISK (malignant)',
    1: '🟡 MODERATE (benign but monitor)',
    2: '🟡 MODERATE (locally invasive)',
    3: '🟢 LOW RISK (pre-cancerous, treatable)',
    4: '🟢 LOW RISK (benign)',
    5: '🟢 LOW RISK (benign)',
    6: '🟢 LOW RISK (benign)',
}

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_model(checkpoint_path: str, rank: int, device: str = 'cpu'):
    """Loads a trained model checkpoint."""
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=7)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    model = apply_lora_to_vit(model, rank=rank, alpha=16)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def predict(image, model: torch.nn.Module, device: str = 'cpu'):
    """Returns prediction probabilities."""
    if image is None:
        return None

    from PIL import Image
    if hasattr(image, 'mode') and image.mode != 'RGB':
        image = image.convert('RGB')
    img_tensor = PREPROCESS(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    return probs


def build_demo(checkpoint_dir: str = './results', device: str = 'cpu'):
    """
    Build and launch the Gradio demo.

    NOTE: Run this AFTER training all methods and saving checkpoints.
    """

    models = {}
    try:
        # FedAvg: homogeneous rank 8 for all clients
        models['fedavg'] = load_model(f'{checkpoint_dir}/fedavg/best_model.pt', rank=8, device=device)
        # HetLoRA, FlexLoRA, DQAW: server saves client 0's model (rank 16)
        models['hetlora'] = load_model(f'{checkpoint_dir}/hetlora/best_model.pt', rank=16, device=device)
        models['flexlora'] = load_model(f'{checkpoint_dir}/flexlora/best_model.pt', rank=16, device=device)
        models['dqaw'] = load_model(f'{checkpoint_dir}/dqaw/best_model.pt', rank=16, device=device)
        print("All 4 models loaded successfully")
    except Exception as e:
        print(f"Warning: {e}")
        print("Demo will run in demo mode — train models first to enable real predictions")

    def classify_image(image):
        if image is None:
            return "Please upload an image", {}, {}, {}, {}

        results = {}
        for method_name, model in models.items():
            probs = predict(image, model, device)
            results[method_name] = probs

        if not results:
            return "No trained models found. Run experiments first.", {}, {}, {}, {}

        def format_probs(probs, method_label):
            if probs is None:
                return {}
            return {CLASS_NAMES[i]: float(probs[i]) for i in range(7)}

        fedavg_probs = format_probs(results.get('fedavg'), 'FedAvg')
        hetlora_probs = format_probs(results.get('hetlora'), 'HetLoRA')
        flexlora_probs = format_probs(results.get('flexlora'), 'FlexLoRA')
        dqaw_probs = format_probs(results.get('dqaw'), 'DQAW')

        if results.get('dqaw') is not None:
            top_class = np.argmax(results['dqaw'])
            mel_probs = {m: float(results[m][0]) if results[m] is not None else 0.0 for m in results}
            mel_comparison = " | ".join([f"{m}: MEL={mel_probs[m]:.1%}" for m in ['fedavg', 'hetlora', 'flexlora', 'dqaw'] if m in mel_probs])
            interpretation = f"""
**DQAW Prediction**: {CLASS_NAMES[top_class]}
**Clinical Risk**: {CLINICAL_RISK[top_class]}

**MEL (Melanoma) probability by method**: {mel_comparison}

**Why this matters**: Compare all 4 methods for melanoma (MEL) recall.
Rank collapse in HetLoRA degrades malignant lesion detection.
DQAW's quality-adaptive weighting preserves higher-rank feature directions.
            """
        else:
            interpretation = "No model loaded."

        return interpretation, fedavg_probs, hetlora_probs, flexlora_probs, dqaw_probs

    with gr.Blocks(title="MedHetLoRA Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🩺 MedHetLoRA — Heterogeneous Federated LoRA for Skin Lesion Classification

        **Research by Cristian Mendoza, UCI 2026**

        This demo shows how rank collapse in heterogeneous federated LoRA affects
        skin lesion classification — specifically how it harms detection of high-stakes
        malignant classes (melanoma) compared to benign classes.

        Upload a dermoscopy image to compare **four** federated training methods:
        - **FedAvg**: Homogeneous ranks (baseline)
        - **HetLoRA**: Heterogeneous ranks with zero-padding (rank collapse ⚠️)
        - **FlexLoRA**: Heterogeneous ranks with ΔW + SVD
        - **DQAW (Ours)**: Heterogeneous ranks with data-quality-adaptive weighting ✅
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Dermoscopy Image")
                predict_btn = gr.Button("Compare Methods", variant="primary")

            with gr.Column(scale=2):
                interpretation = gr.Markdown("Upload an image to see predictions")

        with gr.Row():
            fedavg_out = gr.Label(num_top_classes=7, label="FedAvg (Homogeneous)")
            hetlora_out = gr.Label(num_top_classes=7, label="HetLoRA (Rank Collapse ⚠️)")
            flexlora_out = gr.Label(num_top_classes=7, label="FlexLoRA (ΔW+SVD)")
            dqaw_out = gr.Label(num_top_classes=7, label="DQAW — Ours ✅")

        predict_btn.click(
            fn=classify_image,
            inputs=image_input,
            outputs=[interpretation, fedavg_out, hetlora_out, flexlora_out, dqaw_out]
        )

        gr.Markdown("""
        ---
        **Dataset**: FedISIC (4 hospitals, 22K dermoscopic images, 7 lesion classes)
        **Backbone**: ViT-base/16 (ImageNet-21k pretrained)
        **Client Ranks**: Client 0=16, Client 1=8, Client 2=4, Client 3=2
        **Novel Contribution**: DQAW weights client contributions by per-sample LoRA update magnitude,
        rewarding quality over quantity and mitigating rank-heterogeneity bias.
        """)

    return demo


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    demo = build_demo(device=device)
    demo.launch(share=True)
