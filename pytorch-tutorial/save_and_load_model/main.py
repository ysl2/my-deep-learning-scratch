# ===============================
# === SAVE AND LOAD THE MODEL ===
# ===============================

import torch
import torch.onnx as onnx
import torchvision.models as models


# ===
# === Saving and Loading Model Weights
# ===

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()


# ===
# === Saving and Loading Models with Shapes
# ===

torch.save(model, 'model.pth')
model = torch.load('model.pth')


# ===
# === Exporting Model to ONNX
# ===

input_image = torch.zeros((1, 3, 224, 224))
onnx.export(model, input_image, 'model.onnx')
