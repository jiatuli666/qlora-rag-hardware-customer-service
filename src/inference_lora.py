import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel


# ============================================================
# 1. 模型路径
# ============================================================

BASE_MODEL = "Qwen/Qwen3-0.6B"

LORA_MODEL = "outputs/qwen3-0.6b-qlora-test"


# ============================================================
# 2. 4bit量化配置
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


# ============================================================
# 3. 加载Tokenizer
# ============================================================

print("=" * 60)
print("加载Tokenizer")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    LORA_MODEL
)

print("Tokenizer加载完成")


# ============================================================
# 4. 加载基础模型
# ============================================================

print("\n" + "=" * 60)
print("加载基础模型")
print("=" * 60)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

print("基础模型加载完成")


# ============================================================
# 5. 加载LoRA Adapter
# ============================================================

print("\n" + "=" * 60)
print("加载LoRA Adapter")
print("=" * 60)

model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL
)

print("LoRA Adapter加载成功")


# ============================================================
# 6. 设置推理模式
# ============================================================

model.eval()


# ============================================================
# 7. 定义问题
# ============================================================

question = "PH2批头是什么？"


messages = [
    {
        "role": "system",
        "content": "你是一名专业的五金工具电商客服，回答要准确、简洁、实用。"
    },
    {
        "role": "user",
        "content": question
    }
]


# ============================================================
# 8. 构建Prompt
# ============================================================

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)


# ============================================================
# 9. Tokenize
# ============================================================

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}


# ============================================================
# 10. 模型生成
# ============================================================

print("\n" + "=" * 60)
print("开始生成回答")
print("=" * 60)

with torch.no_grad():

    outputs = model.generate(

        **inputs,

        # 最多生成128个Token
        max_new_tokens=128,

        # 控制随机性
        temperature=0.7,

        # Top-P
        top_p=0.9,

        # 防止重复
        repetition_penalty=1.05,

        # 使用EOS
        eos_token_id=tokenizer.eos_token_id,

        # PAD
        pad_token_id=tokenizer.pad_token_id
    )


# ============================================================
# 11. 只截取新生成的内容
# ============================================================

generated_tokens = outputs[
    0,
    inputs["input_ids"].shape[1]:
]

answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)


# ============================================================
# 12. 输出
# ============================================================

print("\n" + "=" * 60)
print("用户问题")
print("=" * 60)

print(question)


print("\n" + "=" * 60)
print("QLoRA微调后回答")
print("=" * 60)

print(answer)


print("\n" + "=" * 60)
print("推理完成")
print("=" * 60)