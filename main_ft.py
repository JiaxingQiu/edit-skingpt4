import argparse
from pathlib import Path

def run(args):
    print(f"model_type    : {args.model_type}")
    print(f"retrain       : {args.retrain}")
    print(f"target        : {args.target}")
    print(f"prompt_keys   : {args.prompt_keys}")
    print(f"answer_keys   : {args.answer_keys}")
    print(f"eval_target   : {args.eval_target}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["gpt4", "gpt_io"], required=True)
    parser.add_argument("--retrain", action="store_true", help="Retrain the model")
    parser.add_argument("--target", help="Finetune generated text output")
    parser.add_argument("--prompt_keys", nargs="+", default=["text_demo"], help="For gpt_io: e.g., text_demo text_lesion")
    parser.add_argument("--answer_keys", nargs="+", default=["text_outcome"], help="For gpt_io: e.g., text_outcome text_lesion")
    parser.add_argument("--eval_target", default = "y3", help="Single y-field (e.g., y3, y16, text_full) as eval target")
    args = parser.parse_args()
    run(args)

    # project directory (root of main)
    ROOT = Path(__file__).resolve().parent      
    RUN  = ROOT / "finetune/run"
    RESULTS = ROOT / "finetune/results" # # make sure a results/ folder exists 
    RESULTS.mkdir(exist_ok=True)    


    globals()["args"] = args
    globals().update(vars(args)) 
    for script in ["init.py", "ft.py", "eval.py"]:
        exec((RUN / script).read_text(), globals())
    
