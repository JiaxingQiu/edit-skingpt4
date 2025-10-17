import argparse
from pathlib import Path

def run(args):
    print("-" * 50)
    print(f"model_type    : {args.model_type}")
    print(f"retrain       : {args.retrain}")
    print(f"continue_train: {args.continue_train}")
    print(f"n_epochs      : {args.n_epochs}")
    print(f"init_lr       : {args.init_lr}")
    print(f"eval_target   : {args.eval_target}")
    print(f"prompt_keys   : {args.prompt_keys}")
    print(f"answer_keys   : {args.answer_keys}")
    print(f"unfreeze_vit  : {args.unfreeze_vit}")
    print(f"unfreeze_qformer: {args.unfreeze_qformer}")
    print("-" * 50)
    print(f"no_conv_training: {args.no_conv_training}")
    print(f"keep_system   : {args.keep_system}")
    print(f"temperature   : {args.temperature}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["gpt4", "gpt_io"], required=True)
    parser.add_argument("--unfreeze_vit", action="store_true", help="Unfreeze the vision encoder")
    parser.add_argument("--unfreeze_qformer", action="store_true", help="Unfreeze the Q-Former")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model")
    parser.add_argument("--continue_train", action="store_true", help="Continue training from the last checkpoint")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--init_lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--prompt_keys", nargs="+", default=["text_demo"], help="For gpt_io: e.g., text_demo text_lesion")
    # parser.add_argument("--answer_keys", nargs="+", default=["text_y3"], help="For gpt_io: e.g., text_outcome text_lesion")
    parser.add_argument("--answer_keys", nargs="+", default=["y3"], help="For gpt_io: e.g., text_outcome text_lesion")
    parser.add_argument("--no_conv_training", action="store_true", help="No conversation template for training")
    # for eval 
    parser.add_argument("--eval_target", default = "y3", help="Single y-field (e.g., y3, y16, text_full) as eval target")
    parser.add_argument("--keep_system", action="store_true", help="Keep system prompt for eval") # for gpt4
    parser.add_argument("--temperature", type=float, default=0.001, help="Temperature for eval")
    args = parser.parse_args()
    run(args)

    # engineer args
    args.freeze_vit = not args.unfreeze_vit
    args.freeze_qformer = not args.unfreeze_qformer
    args.use_conv_training = not args.no_conv_training
    args.remove_system = not args.keep_system

    # project directory (root of main)
    ROOT = Path(__file__).resolve().parent      
    RUN  = ROOT / "finetune/run"
    RESULTS = ROOT / "finetune/results" # # make sure a results/ folder exists 
    RESULTS.mkdir(exist_ok=True)    


    globals()["args"] = args
    for script in ["init.py", "ft.py", "eval.py"]:
        exec((RUN / script).read_text(), globals())
    
