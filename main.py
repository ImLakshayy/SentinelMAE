import torch
from utils.config_loader import load_config
from models.videomae_finetune import build_model

if __name__ == "__main__":
    config = load_config()

    device = torch.device(config["project"]["device"])

    print("Loading model...")
    model = build_model(config)
    model.to(device)

    print("Model loaded successfully!")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable params: {trainable}")
    print(f"Total params: {total}")