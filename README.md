# TAMOG-Net
TAMOG-Net is a lightweight disease recognition network designed for high-resolution images acquired by low-UAVs. By incorporating the OSSF and RepDepthwiseConv modules, it enhances multi-scale feature extraction while maintaining low computational cost, enabling accurate classification of crop health conditions and common diseases.

# Document content

samples/: Test images of tea leaves and rice. There are 4 categories of tea leaves with 20 images per category, and 5 categories of rice with 20 images per category.
tamog_net.py: Complete TAMOG-Net network code
weights/: Pre-trained weights for tea and rice seed 3407.
evaluate_samples.py: Test samples/ and output per-image correct and incorrect results.

# Operating environment

- Python 3.9+
- PyTorch 2.7.1
- torchvision 0.22.1
- Pillow

# Image preprocessing

from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Weight Loading and Inference

import torch

state_dict = torch.load(
    "weights/tamog_net_rice_seed3407.pth",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(state_dict, strict=True)
model.eval()

with torch.inference_mode():
    logits = model(preprocess(image).unsqueeze(0))
    index = logits.argmax(dim=1).item()
The tea weight uses a 4-classification model, with the file tamog_net_tea_seed3407.pth; the rice weight uses a 5-classification model, with the file tamog_net_rice_seed3407.pth.

# Tag Mapping

The model output index must be interpreted in the following order:
	tea_labels = ["Basilepta fulvipes", "Tea anthracnose", "Heat stress", "Healthy",]
	rice_labels = ["Rice bacterial blight", "Rice leaf roller", "Rice stem borer","Healthy", "Rice leaf blast",]
	predicted_label = rice_labels[index]
The index of folder names sorted alphabetically is not necessarily equal to the model output index. When calculating test set metrics, label mapping should be performed using the model output order above.
