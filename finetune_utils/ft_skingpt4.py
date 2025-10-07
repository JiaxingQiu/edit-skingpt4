# Dataset wrapper → returns {"image", "text_input"} expected by model.forward
import torch
from torch.utils.data import Dataset, DataLoader
# suppress warnings
import warnings
warnings.filterwarnings("ignore")




class MIDASFTSkGPT4Dataset(Dataset):
    def __init__(self, base_dataset, vis_processor, target):
        """
        Args:
            base_dataset: the base dataset
            vis_processor: the visual processor
            target: the target label to train on: "y3" (malignant/benign/other) or "y16" or "y16_description"
        """
        self.base = base_dataset
        self.vis = vis_processor
        self.target = target

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        img = item['image']                               # PIL Image
        target_text = item['y'][self.target]                   # e.g., "malignant" or a description
        img_tensor = self.vis(img)                        # preprocess to tensor CHW
        return {"image": img_tensor, "text_input": target_text}

    def collate_fn(self, batch):
        images = torch.stack([b["image"] for b in batch], dim=0)
        texts = [b["text_input"] for b in batch]
        return {"image": images, "text_input": texts}
    
    def get_loader(self, batch_size, shuffle, num_workers):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)

# example usage
# # vis_processor is already initialized from run/init.py
# target = 'y3'
# train_ds_ft = MIDASFTSkGPT4Dataset(train_dataset, vis_processor, target)
# val_ds_ft   = MIDASFTSkGPT4Dataset(val_dataset,   vis_processor, target)

# train_loader = train_ds_ft.get_loader(batch_size=2, shuffle=True, num_workers=2)
# val_loader   = val_ds_ft.get_loader(batch_size=2, shuffle=False, num_workers=2)

# device = next(model.parameters()).device
# sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())


import os
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

def _get_device(model):
    return next(model.parameters()).device

def run_epoch(loader, model, optimizer=None, scaler=None, train=True):
    device = _get_device(model)
    model.train(mode=train)
    total_loss, n = 0.0, 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        texts  = batch["text_input"]
        samples = {"image": images, "text_input": texts}

        if train:
            optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            out = model(samples)
            loss = out["loss"]

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * images.size(0)
        n += images.size(0)

    return total_loss / max(n, 1)


# LLaMA weights: frozen (not updated). The model sets all llm_model params to requires_grad=False.
# Vision encoder + layer norm: frozen by default (freeze_vit=True).
# Q-Former + query tokens: frozen by default (freeze_qformer=True).
# Updated weights: primarily the projection layer llama_proj (mapping Q-Former outputs to LLaMA hidden size).
def finetune(model, train_loader, val_loader, n_epochs=1, retrain=False,
             lr=1e-4, weight_decay=0.0, ckpt_path="./model_skingpt4/weights/finetune_llama.pth"):
    if retrain or not os.path.exists(ckpt_path):
        optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
        scaler = GradScaler(enabled=True)
        best_val = float("inf")

        try:
            for epoch in range(n_epochs):
                train_loss = run_epoch(train_loader, model, optimizer, scaler, train=True)
                with torch.no_grad():
                    val_loss = run_epoch(val_loader, model, train=False)
                print(f"epoch {epoch+1}/{n_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"model": model.state_dict()}, ckpt_path)
        except KeyboardInterrupt:
            torch.save({"model": model.state_dict()}, ckpt_path)
            print(f"\nKeyboardInterrupt: saved checkpoint to {ckpt_path}")
        finally:
            model.eval()
    else:
        state = torch.load(ckpt_path, map_location="cpu")
        state = state.get("model", state)
        _ = model.load_state_dict(state, strict=False)
        model.eval()

    return model



# example usage
# model = finetune(model, train_loader, val_loader, n_epochs=1, retrain=False,
#         lr=1e-4, weight_decay=0.0, ckpt_path="./model_skingpt4/weights/finetune_llama.pth")