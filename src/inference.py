import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


# =========================
# 1. 基础模型和 LoRA 路径
# =========================
MODEL_NAME = "Qwen/Qwen3-0.6B"
LORA_PATH = "outputs/qwen3-0.6b-qlora"


# =========================
# 2. 4bit 量化配置
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# =========================
# 3. 加载 tokenizer
# =========================
print("=" * 60)
print("加载 Tokenizer")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# =========================
# 4. 加载基础模型
# =========================
print("\n加载基础模型...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("基础模型加载完成")


# =========================
# 5. 加载 LoRA Adapter
# =========================
print("\n加载 LoRA Adapter...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
)

model.eval()

print("LoRA Adapter 加载完成")


# =========================
# 6. 单轮问答函数
# =========================
def ask(question):
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名专业的五金工具电商客服，"
                "回答要准确、简洁、实用。"
                "不确定的产品参数不要擅自编造。"
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    # Qwen3关闭思考模式，避免客服回答出现大量<think>
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    # 移动到模型所在设备
    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # 只取模型新增生成的部分
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


# =========================
# 7. 测试问题
# =========================
test_questions = [
    "PH2批头是什么？",
    "S2钢有什么特点？",
    "65mm和100mm批头有什么区别？",
    "电动螺丝刀可以使用这个批头吗？",
    "一盒有多少支？",
    "磁性批头能吸不锈钢螺丝吗？",
]


# =========================
# 8. 开始测试
# =========================
print("\n")
print("=" * 60)
print("开始 QLoRA 模型测试")
print("=" * 60)

for i, question in enumerate(test_questions, 1):

    print(f"\n【问题 {i}】")
    print(question)

    answer = ask(question)

    print("【模型回答】")
    print(answer)

print("\n")
print("=" * 60)
print("测试完成")
print("=" * 60)