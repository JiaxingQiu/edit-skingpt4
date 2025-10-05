if base_model not in locals():
    base_model = 'llama' 


# Initialize model with the correct config path
if base_model == 'llama':
    config_path = "eval_configs/skingpt4_eval_llama2_13bchat.yaml"  # llama
elif base_model == 'vicuna':
    config_path = 'eval_configs/skingpt4_eval_vicuna.yaml' # vicuna
else:
    raise ValueError(f"Invalid base model: {base_model}")

gpu_id = 0  # Update if using different GPU

# Create args object
class Args:
    def __init__(self):
        self.cfg_path = config_path
        self.gpu_id = gpu_id
        self.options = None

args = Args()
cfg = Config(args)

# Initialize model
print('Initializing Chat')
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

def chat_with_image(image, question, num_beams=1, temperature=0.01):
    # Initialize conversation
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(image, chat_state, img_list)
    chat.ask(question, chat_state)
    response = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=300,
        max_length=2000
    )[0]
    return response