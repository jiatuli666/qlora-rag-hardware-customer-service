import os
import json
import time
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer


# ============================================================
# 1. 基础配置
# ============================================================

MODEL_NAME = "Qwen/Qwen3-0.6B"

#TRAIN_FILE = "data/train_v2.json"
#TRAIN_FILE = "data/train_v3.json"
#TRAIN_FILE = "data/train_v4.json"
#TRAIN_FILE = "data/train_v5.json"
#TRAIN_FILE = "data/train_v6.json"
TRAIN_FILE = "data/train_v7.json"
VAL_FILE = "data/val.json"

#OUTPUT_DIR = "outputs/qwen3-0.6b-qlora-v2"
#OUTPUT_DIR = "outputs/qwen3-0.6b-qlora-v3"
#OUTPUT_DIR = "outputs/qwen3-0.6b-qlora-v4"
#OUTPUT_DIR = "outputs/qwen3-0.6b-qlora-v5"
#OUTPUT_DIR = "outputs/qwen3-0.6b-qlora-v6"
OUTPUT_DIR = "outputs/qwen3-0.6b-qlora-v7"
MAX_LENGTH = 256


# ============================================================
# 2. 检查运行环境
# ============================================================

print("=" * 60)
print("检查运行环境")
print("=" * 60)

print("PyTorch版本：", torch.__version__)
print("CUDA是否可用：", torch.cuda.is_available())

if torch.cuda.is_available():

    gpu_name = torch.cuda.get_device_name(0)

    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print("GPU：", gpu_name)
    print("显存：", f"{gpu_memory:.1f} GB")

else:
    print("警告：没有检测到CUDA，将无法进行GPU QLoRA训练。")
    raise RuntimeError("CUDA不可用")


# ============================================================
# 3. 加载数据
# ============================================================

print()
print("=" * 60)
print("加载训练数据")
print("=" * 60)

train_dataset = load_dataset(
    "json",
    data_files=TRAIN_FILE,
    split="train"
)

val_dataset = load_dataset(
    "json",
    data_files=VAL_FILE,
    split="train"
)

print("训练数据：", len(train_dataset))
print("验证数据：", len(val_dataset))


# ============================================================
# 4. 加载Tokenizer
# ============================================================

print()
print("=" * 60)
print("加载Tokenizer")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# Qwen系列通常有pad_token
# 如果没有，则使用eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer加载完成")


# ============================================================
# 5. 配置4bit量化
# ============================================================

print()
print("=" * 60)
print("配置4bit NF4量化")
print("=" * 60)

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    # RTX 3060使用FP16
    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True,

)

print("4bit NF4量化配置完成")


# ============================================================
# 6. 加载基础模型
# ============================================================

print()
print("=" * 60)
print("加载Qwen3-0.6B基础模型")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto",
    torch_dtype=torch.float16
)

print("基础模型加载完成")


# ============================================================
# 7. QLoRA准备
# ============================================================

print()
print("=" * 60)
print("准备QLoRA训练")
print("=" * 60)

model = prepare_model_for_kbit_training(model)

print("k-bit训练准备完成")


# ============================================================
# 8. LoRA配置
# ============================================================

print()
print("=" * 60)
print("配置LoRA")
print("=" * 60)

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
        "o_proj",
    ],

)

print("LoRA配置完成")

print()
print("LoRA参数：")
print("r =", lora_config.r)
print("alpha =", lora_config.lora_alpha)
print("dropout =", lora_config.lora_dropout)


# ============================================================
# 9. 开启梯度检查点
# ============================================================

print()
print("=" * 60)
print("开启Gradient Checkpointing")
print("=" * 60)

model.gradient_checkpointing_enable()

print("Gradient Checkpointing已开启")


# ============================================================
# 10. 训练参数
# ============================================================

print()
print("=" * 60)
print("配置训练参数")
print("=" * 60)

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=4,

    # --------------------------------------------------------
    # Epoch
    # --------------------------------------------------------

    num_train_epochs=3,

    # --------------------------------------------------------
    # Learning Rate
    # --------------------------------------------------------

    learning_rate=1e-4,

    lr_scheduler_type="cosine",


    # --------------------------------------------------------
    # FP16
    # --------------------------------------------------------

    fp16=False,

    bf16=False,

    # --------------------------------------------------------
    # 优化器
    # --------------------------------------------------------

    optim="paged_adamw_8bit",

    # --------------------------------------------------------
    # 日志
    # --------------------------------------------------------

    logging_steps=5,

    logging_first_step=True,

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_strategy="epoch",

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    save_strategy="epoch",

    save_total_limit=2,

    # --------------------------------------------------------
    # 其他
    # --------------------------------------------------------

    report_to="none",

    remove_unused_columns=False,

    gradient_checkpointing=True,

)

print("训练参数配置完成")


# ============================================================
# 11. 创建SFT Trainer
# ============================================================

print()
print("=" * 60)
print("创建SFT Trainer")
print("=" * 60)

trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    peft_config=lora_config,


)

print("SFT Trainer创建完成")


# ============================================================
# 12. 打印可训练参数
# ============================================================

print()
print("=" * 60)
print("检查LoRA参数")
print("=" * 60)

try:
    trainer.model.print_trainable_parameters()
except Exception:
    pass


# ============================================================
# 13. 开始训练
# ============================================================

print()
print("=" * 60)
print("开始QLoRA正式训练")
print("=" * 60)

start_time = time.time()

trainer.train()

end_time = time.time()

training_time = end_time - start_time


# ============================================================
# 14. 保存LoRA Adapter
# ============================================================

print()
print("=" * 60)
print("保存LoRA Adapter")
print("=" * 60)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("LoRA模型保存完成：")
print(OUTPUT_DIR)


# ============================================================
# 15. GPU显存统计
# ============================================================

print()
print("=" * 60)
print("GPU显存统计")
print("=" * 60)

if torch.cuda.is_available():

    current_memory = torch.cuda.memory_allocated() / 1024**3

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    print(
        "当前显存占用：",
        f"{current_memory:.2f} GB"
    )

    print(
        "峰值显存占用：",
        f"{peak_memory:.2f} GB"
    )


# ============================================================
# 16. 输出训练结果
# ============================================================

print()
print("=" * 60)
print("QLoRA训练完成")
print("=" * 60)

print(
    "训练耗时：",
    f"{training_time / 60:.2f} 分钟"
)

print(
    "模型保存位置：",
    OUTPUT_DIR
)

print()
print("下一步可以使用LoRA模型进行推理。")
print("=" * 60)