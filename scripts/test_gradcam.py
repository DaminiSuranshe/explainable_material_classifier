"""
End-to-end Grad-CAM test for trained small-data model.
"""

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from src.models.transfer_learning_model import TransferLearningModel
from src.data_management.category_manager import CategoryManager


def generate_gradcam(model, image_tensor, class_idx):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, inp, out):
        activations.append(out)

    handle_fwd = model.features.register_forward_hook(forward_hook)
    handle_bwd = model.features.register_backward_hook(backward_hook)

    output = model(image_tensor)
    score = output["logits"][:, class_idx]
    score.backward()

    grads = gradients[0]
    acts = activations[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1)

    cam = torch.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    handle_fwd.remove()
    handle_bwd.remove()

    return cam


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cat_mgr = CategoryManager("config/categories.json")
    num_classes = cat_mgr.num_classes()[1]

    model = TransferLearningModel(num_classes=num_classes)
    model.load_state_dict(
        torch.load("models/saved_models/small_data_v1.0.pth", map_location=device)
    )
    model.to(device)

    img_path = "test_image.jpg"  # replace with your image
    raw = cv2.imread(img_path)
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    img_tensor = preprocess(Image.fromarray(raw)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)["logits"]
        pred_class = logits.argmax(dim=1).item()

    cam = generate_gradcam(model, img_tensor, pred_class)
    cam = cv2.resize(cam, (raw.shape[1], raw.shape[0]))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(raw, 0.6, heatmap, 0.4, 0)

    cv2.imwrite("gradcam_overlay.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("✅ Grad-CAM saved as gradcam_overlay.jpg")


if __name__ == "__main__":
    main()
