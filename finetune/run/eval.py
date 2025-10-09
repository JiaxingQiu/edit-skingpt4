# --- print one example ---
i = 16
image = test_dataset[i]['image']
temperature = 0.01
ending_question = "Is the lesion malignant or benign, or unknown?"
print(f"ground truth: {test_dataset[i]['y'][target]}")
print("-" * 50)
print("Pretrained model")
model = load_model_weights(model, pt_ckpt_path)
resp = chat_with_image(chat, image, ending_question, temperature=temperature)
print(resp)
print("-" * 50)
print(f"Finetuned model ({model_type}_{target})")
model = load_model_weights(model, ft_ckpt_path)
if model_type == "gpt_io":
    prompt = ""
    for k in prompt_keys:
        prompt += f"{test_dataset[i]['y'][k]}. "
    question = prompt + ending_question
else:
    question = ending_question
resp = chat_with_image(chat, image, question, temperature=temperature)
print(resp)    


# --- eval ---
ending_question = "Is the lesion malignant or benign, or other?"
res_dir = f"./finetune/results/ft_skin{model_type}_{eval_target}"
os.makedirs(res_dir, exist_ok=True)
for split_name, ds in [("test", test_dataset), ("train", train_dataset), ("val", val_dataset)]:
    if model_type == "gpt_io":
        res = eval_ft_skingpt4(chat, ds, temperature=0.05, target=eval_target, question=ending_question, prompt_keys=prompt_keys)
    else:
        res = eval_ft_skingpt4(chat, ds, temperature=0.05, target=eval_target, question=ending_question, prompt_keys=None)
    torch.save(res, f"{res_dir}/eval_{split_name}.pth")

