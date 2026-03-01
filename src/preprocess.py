import torch
import torchvision.transforms as T
from PIL import Image

IMG_SIZE = 224  # 你 test_single_ckp 里用的是 224

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

def load_image_to_tensor(file) -> torch.Tensor:
    img = Image.open(file).convert("RGB")
    return _transform(img)
