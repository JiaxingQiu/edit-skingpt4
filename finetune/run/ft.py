if args.model_type == "gpt_io":
    train_ds_ft = MIDASFTSkGPTIODataset(train_dataset, vis_processor, prompt_keys=args.prompt_keys, answer_keys=args.answer_keys)
    val_ds_ft   = MIDASFTSkGPTIODataset(val_dataset,   vis_processor, prompt_keys=args.prompt_keys, answer_keys=args.answer_keys)
elif args.model_type == "gpt4":
    train_ds_ft = MIDASFTSkGPT4Dataset(train_dataset, vis_processor, args.eval_target)
    val_ds_ft   = MIDASFTSkGPT4Dataset(val_dataset,   vis_processor, args.eval_target)
else:
    raise ValueError(f"Unsupported pretrained model type: {args.model_type}")

train_loader = train_ds_ft.get_loader(batch_size=2, shuffle=True, num_workers=2)
val_loader   = val_ds_ft.get_loader(batch_size=2, shuffle=False, num_workers=2)
model.finetune(train_loader, val_loader, n_epochs=args.n_epochs, retrain=args.retrain, lr=args.init_lr, ckpt_path=args.ft_ckpt_path)

