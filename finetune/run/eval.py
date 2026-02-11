import random
# --- print one example ---
exp_ds = train_dataset #test_dataset
exp_dl = train_ds_ft
N = min(20, len(exp_ds))
idxs = random.sample(range(len(exp_ds)), N)
for i in idxs: # sample 20 between 1 and len(exp_ds)
    print("-" * 50)
    image = exp_ds[i]['image']
    ending_question = chat.model.conv_question
    print(f"ground truth: {exp_ds[i]['y'][args.eval_target]}")
    print(f"answer: {exp_dl[i]['answer_output']}")
    # prepare question (local_q)
    if args.model_type == "gpt_io":
        prompt = ""
        for k in args.prompt_keys:
            prompt += f"{exp_ds[i]['y'][k]} "
        question = prompt + ending_question
    else:
        question = ending_question
    print("Pretrained model")
    model = load_model_weights(model, args.pt_ckpt_path)
    resp = chat_with_image(chat, image, question, temperature=args.temperature, remove_system=args.remove_system)
    print(resp)
    print(f"Finetuned model ({model_name})")
    model = load_model_weights(model, args.ft_ckpt_path)
    resp = chat_with_image(chat, image, question, temperature=args.temperature, remove_system=args.remove_system, print_prompt=True)
    print(resp)    


# --- eval ---
import json
import numpy as np

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)

os.makedirs(args.res_dir, exist_ok=True)
for split_name, ds in [("test", test_dataset), ("train", train_dataset), ("val", val_dataset)]:
    if args.model_type == "gpt_io":
        res = eval_ft_skingpt4(chat, ds, temperature=args.temperature, remove_system=args.remove_system, 
                               target=args.eval_target, prompt_keys=args.prompt_keys) 
    else:# gpt4: prompt_keys=None
        res = eval_ft_skingpt4(chat, ds, temperature=args.temperature, remove_system=args.remove_system,
                               target=args.eval_target, prompt_keys=None) 
    with open(f"{args.res_dir}/eval_{split_name}.json", "w") as f:
        json.dump(res, f, default=_json_default, indent=2)

