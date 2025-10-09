import os, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

from model_skingpt4 import *
from data_utils import *
from finetune_utils import *
from eval_utils import *

# --- init model ---
pretrained_cfg_path = f"skin{model_type}_eval_llama2_13bchat"
cfg = init_cfg(0, pretrained_cfg_path)
cfg.model_cfg.freeze_vit = not unfreeze_vit
cfg.model_cfg.freeze_qformer = not unfreeze_qformer
print(cfg.model_cfg)
model, vis_processor, chat = init_chat(cfg)
sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())

# --- model name (key for weights and results saving)---
model_name = f"ft_skin{model_type}_{eval_target}" # universal model name
unfreeze_suffix = "" # default unfreeze is the alignment layer
if unfreeze_vit:
    unfreeze_suffix += unfreeze_suffix + "_vit"
if unfreeze_qformer:
    unfreeze_suffix += unfreeze_suffix + "_qformer"
model_name += unfreeze_suffix
print("-" * 50)
print(f"model name: {model_name}")
print(f"model type: {type(model).__name__}")
print("-" * 50)

# --- saving/loading paths ---
pt_ckpt_path = model.ckpt
print(f"pretrained weights ckpt path: {pt_ckpt_path}")
ft_ckpt_path = f"./model_skingpt4/weights/{model_name}.pth"
print(f"finetuned weights ckpt path: {ft_ckpt_path}")
res_dir = f"./finetune/results/{model_name}"
print(f"results directory: {res_dir}")
print("-" * 50)

# --- init data ---
df = process_tabular("./data")
train_df, val_df, test_df = split_df(df)
train_dataset = MIDASDataset(train_df, "./data")
val_dataset = MIDASDataset(val_df, "./data")
test_dataset = MIDASDataset(test_df, "./data")