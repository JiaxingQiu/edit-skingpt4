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



# IO dataset: returns {"image", "prompt_input", "answer_output"}
class MIDASFTSkGPTIODataset(Dataset):
    def __init__(self, base_dataset, vis_processor, prompt_key="text_demo", answer_keys=("text_outcome", "text_lesion")):
        """
        Args:
            base_dataset: underlying dataset yielding dicts with PIL 'image' and a 'y' dict of text fields
            vis_processor: visual preprocessor to tensorize images
            prompt_key: which 'y' key to use as prompt (e.g., 'text_demo')
            answer_keys: tuple of 'y' keys to concatenate for answer (e.g., ('text_outcome','text_lesion'))
        """
        self.base = base_dataset
        self.vis = vis_processor
        self.prompt_key = prompt_key
        self.answer_keys = answer_keys

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        img = item["image"]
        img_tensor = self.vis(img)

        y = item.get("y", {}) or {}
        prompt = (y.get(self.prompt_key, "") or "").strip()
        parts = []
        for k in self.answer_keys:
            v = (y.get(k, "") or "").strip()
            if v:
                parts.append(v)
        answer = " ".join(parts)

        return {"image": img_tensor, "prompt_input": prompt, "answer_output": answer}

    def collate_fn(self, batch):
        images = torch.stack([b["image"] for b in batch], dim=0)
        prompts = [b["prompt_input"] for b in batch]
        answers = [b["answer_output"] for b in batch]
        return {"image": images, "prompt_input": prompts, "answer_output": answers}

    def get_loader(self, batch_size, shuffle, num_workers):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)

