ft_ckpt_path = f"./model_skingpt4/weights/ft_skin{model_type}_{target}.pth"
if retrain:
    print(f"finetuned weights will be saved to ckpt path: {ft_ckpt_path}")
else:
    print(f"finetuned weights will be loaded from ckpt path: {ft_ckpt_path}")

if model_type == "gpt_io":
    train_ds_ft = MIDASFTSkGPTIODataset(train_dataset, vis_processor, prompt_keys=prompt_keys, answer_keys=answer_keys)
    val_ds_ft   = MIDASFTSkGPTIODataset(val_dataset,   vis_processor, prompt_keys=prompt_keys, answer_keys=answer_keys)
elif model_type == "gpt4":
    train_ds_ft = MIDASFTSkGPT4Dataset(train_dataset, vis_processor, target)
    val_ds_ft   = MIDASFTSkGPT4Dataset(val_dataset,   vis_processor, target)
else:
    raise ValueError(f"Unsupported pretrained model type: {model_type}")

train_loader = train_ds_ft.get_loader(batch_size=2, shuffle=True, num_workers=2)
val_loader   = val_ds_ft.get_loader(batch_size=2, shuffle=False, num_workers=2)
model.finetune(train_loader, val_loader, n_epochs=10, retrain=retrain, lr=1e-4, weight_decay=0.5, ckpt_path=ft_ckpt_path)

