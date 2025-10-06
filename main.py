import os, torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()


# --- init model ---
from model_skingpt4 import *
model, vis_processor, chat = init_chat()
sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())



# --- load data ---
from data_utils import *
df = process_tabular("./data")
train_df, val_df, test_df = split_df(df)
train_dataset = MIDASDataset(train_df, "./data")
val_dataset = MIDASDataset(val_df, "./data")
test_dataset = MIDASDataset(test_df, "./data")



# --- finetune ---
from finetune_utils import *
# vis_processor is already initialized from run/init.py
target = 'y3'
train_ds_ft = MIDASFTSkGPT4Dataset(train_dataset, vis_processor, target)
val_ds_ft   = MIDASFTSkGPT4Dataset(val_dataset,   vis_processor, target)
train_loader = train_ds_ft.get_loader(batch_size=2, shuffle=True, num_workers=2)
val_loader   = val_ds_ft.get_loader(batch_size=2, shuffle=False, num_workers=2)

model = load_model_weights(model, "./model_skingpt4/weights/finetune_llama.pth")
model = finetune(model, train_loader, val_loader, n_epochs=50, retrain=True,
        lr=1e-4, weight_decay=0.0, ckpt_path="./model_skingpt4/weights/finetune_llama.pth")



# --- eval ---
from eval_utils import *
import os, torch

target = "y3"
question = "Is the lesion malignant or benign, or other?"
res_dir = f"./results/ft_skingpt4_{target}"
os.makedirs(res_dir, exist_ok=True)
for split_name, ds in [("test", test_dataset), ("train", train_dataset), ("val", val_dataset)]:
    res = eval_ft_skingpt4(model, ds, temperature=0.01, target=target, question=question)
    torch.save(res, f"{res_dir}/eval_{split_name}.pth")