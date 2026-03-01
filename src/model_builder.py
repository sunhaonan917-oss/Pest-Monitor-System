import os
import sys
import torch

# 让 Python 能找到项目根目录（里面有 methods/）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from methods.backbone import model_dict
from methods.meta_FDMixup_model import MetaFDMixup


def build_model(n_way: int = 5, n_support: int = 5, backbone_name: str = "ResNet10_EMA"):
    model_func = model_dict[backbone_name]
    model = MetaFDMixup(model_func, n_way=n_way, n_support=n_support)
    return model


def load_checkpoint(model, ckpt_path: str, device: torch.device):
    """
    兼容 .tar：里面通常是 {'epoch':..., 'state': model.state_dict()}
    返回：missing_keys, unexpected_keys
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state" in ckpt:
        state_dict = ckpt["state"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError("checkpoint 格式不认识（不是 dict）")

    # 多卡前缀
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.to(device)
    if hasattr(model, "support_label"):
        model.support_label = model.support_label.to(device)

    model.eval()
    return missing, unexpected
