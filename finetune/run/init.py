import os, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

from model_skingpt4 import *
from data_utils import *
from finetune_utils import *
from eval_utils import *

# --- init model ---
pretrained_cfg_path = f"skin{model_type}_eval_llama2_13bchat"
model, vis_processor, chat = init_chat(0, pretrained_cfg_path)
sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())
print(type(model).__name__) 
pt_ckpt_path = model.ckpt
print(f"pretrained weights ckpt path: {pt_ckpt_path}")

# --- init data ---
df = process_tabular("./data")
train_df, val_df, test_df = split_df(df)
train_dataset = MIDASDataset(train_df, "./data")
val_dataset = MIDASDataset(val_df, "./data")
test_dataset = MIDASDataset(test_df, "./data")