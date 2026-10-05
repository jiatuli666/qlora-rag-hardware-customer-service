import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


# =========================
# 1. 模型名称
# =========================

MODEL_NAME = "Qwen/Qwen3-0.6B"


# =========================
# 2. 4bit量化配置
# =========================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True
)


# =========================
# 3. 加载Tokenizer
# =========================

print("正在加载Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer加载成功")


# =========================
# 4. 加载模型
# =========================

print("正在加载模型...")

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto"
)

print("模型加载成功")


# =========================
# 5. 构造对话
# =========================

messages = [

    {
        "role": "system",
        "content": "你是一名专业的五金工具电商客服。"
    },

    {
        "role": "user",
        "content": "PH2批头是什么？"
    }

]


# =========================
# 6. 使用Chat Template
# =========================

text = tokenizer.apply_chat_template(

    messages,

    tokenize=False,

    add_generation_prompt=True

)


# =========================
# 7. Tokenize
# =========================

inputs = tokenizer(

    text,

    return_tensors="pt"

)


# 把输入移动到GPU

inputs = {
    key: value.to(model.device)

    for key, value in inputs.items()
}


# =========================
# 8. 模型生成
# =========================

print()
print("============================")
print("模型正在回答...")
print("============================")


with torch.no_grad():

    outputs = model.generate(

        **inputs,

        max_new_tokens=128,

        temperature=0.7,

        do_sample=True

    )


# =========================
# 9. 解码
# =========================

response = tokenizer.decode(

    outputs[0][inputs["input_ids"].shape[1]:],

    skip_special_tokens=True

)


# =========================
# 10. 输出
# =========================

print()

print("用户：PH2批头是什么？")

print()

print("模型：")

print(response)