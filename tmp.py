import torch

ckpt = torch.load("/home/vnedungadi/CropFM/experiments/pretrain/contrastive_run/model_checkpoints/last.ckpt", map_location="cpu", weights_only=False)  # whatever the file is
state_dict = ckpt.get("state_dict", ckpt)

backbone_state = {}
for k, v in state_dict.items():
    if k.startswith("model.tokenizer.") or k.startswith("model.encoder."):
        backbone_state[k[len("model."):]] = v  # strip "model." prefix

torch.save(backbone_state, "/home/vnedungadi/CropFM/experiments/pretrain/contrastive_run/model_checkpoints/last.pt")