import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

MODEL_NAME = "Qwen/Qwen3-0.6B"

# 1. 4bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 2. 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 3. 加载4bit模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# 4. 准备QLoRA训练
model = prepare_model_for_kbit_training(model)

# 5. LoRA配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)

# 6. 给模型添加LoRA
model = get_peft_model(model, lora_config)

# 7. 查看真正需要训练的参数
model.print_trainable_parameters()