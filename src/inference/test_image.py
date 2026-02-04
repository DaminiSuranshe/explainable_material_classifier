import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from PIL import Image

from src.data_management.category_manager import CategoryManager
from src.preprocessing.heavy_augmentation import HeavyAugmentation


MODEL_PATH = "models/saved_models/small_data_v1.0.pth"
IMAGE_PATH = "dataset/splits/test/Textile Material/Natural Fabric/natural1.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(num_classes: int) -> nn.Module:
    model = mobilenet_v2(weights=None)

    # Minimal classifier — weights will be loaded selectively
    model.classifier = nn.Sequential(
        nn.Linear(1280, num_classes)
    )

    return model


def main():
    category_manager = CategoryManager("config/categories.json")
    num_classes = category_manager.num_subcategories

    model = build_model(num_classes)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # 🔑 THIS IS THE KEY LINE
    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()

    transform = HeavyAugmentation(
        image_size=(224, 224),
        is_training=False,
    )

    image = Image.open(IMAGE_PATH).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)

    top_probs, top_ids = torch.topk(probs, k=3)

    print("\nPrediction results:")
    for p, idx in zip(top_probs[0], top_ids[0]):
        label = category_manager.decode_subcategory(idx.item())
        print(f"  {label}: {p.item():.3f}")


if __name__ == "__main__":
    main()
