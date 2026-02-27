import time
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from captum.attr import LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x).logits

image_path = sys.argv[1] if len(sys.argv) > 1 else "normal_1.jpeg"
print(f"Analyse de l'image : {image_path}")
image = Image.open(image_path).convert("RGB")

model_name = "Aunsiels/resnet-pneumonia-detection"
processor = AutoImageProcessor.from_pretrained(model_name)
hf_model = AutoModelForImageClassification.from_pretrained(model_name)

wrapped_model = ModelWrapper(hf_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapped_model.to(device)
wrapped_model.eval()

inputs = processor(images=image, return_tensors="pt")
input_tensor = inputs["pixel_values"].to(device)
input_tensor.requires_grad = True

# Warm-up
logits = wrapped_model(input_tensor)

start_infer = time.time()
logits = wrapped_model(input_tensor)
predicted_class_idx = logits.argmax(-1).item()
end_infer = time.time()

print(f"Temps d'inférence : {end_infer - start_infer:.4f} secondes")
print(f"Classe prédite : {hf_model.config.id2label[predicted_class_idx]}")

target_layer = wrapped_model.model.resnet.encoder.stages[-1].layers[-1]

start_xai = time.time()
layer_gradcam = LayerGradCam(wrapped_model, target_layer)
attributions_gradcam = layer_gradcam.attribute(input_tensor, target=predicted_class_idx)
end_xai = time.time()

print(f"Temps d'explicabilité (Grad-CAM) : {end_xai - start_xai:.4f} secondes")

upsampled_attr = LayerAttribution.interpolate(attributions_gradcam, input_tensor.shape[2:])
original_img_np = np.array(image.resize(input_tensor.shape[2:][::-1]))

attr_gradcam_np = upsampled_attr.squeeze().cpu().detach().numpy()
attr_gradcam_np = np.expand_dims(attr_gradcam_np, axis=2)

fig, axis = viz.visualize_image_attr(
    attr_gradcam_np,
    original_img_np,
    method="blended_heat_map",
    sign="positive",
    show_colorbar=True,
    title=f"Grad-CAM - Pred: {hf_model.config.id2label[predicted_class_idx]}"
)

output_filename = f"gradcam_{image_path.split('.')[0]}.png"
fig.savefig(output_filename, bbox_inches='tight')
print(f"Visualisation sauvegardée dans {output_filename}")
