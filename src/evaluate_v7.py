import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel


# =========================
# 1. 路径配置
# =========================

MODEL_NAME = "Qwen/Qwen3-0.6B"

LORA_PATH = r"D:\QLoRA\outputs\qwen3-0.6b-qlora-v7"

TEST_FILE = r"D:\QLoRA\data\final_test.json"

OUTPUT_FILE = r"D:\QLoRA\data\final_v7_results.json"


# =========================
# 2. 读取最终测试集
# =========================

with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print("=" * 60)
print(f"最终测试题数量：{len(test_data)}")
print("=" * 60)


# =========================
# 3. 4bit量化配置
# =========================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


# =========================
# 4. Tokenizer
# =========================

print("正在加载Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# =========================
# 5. 加载Base Model
# =========================

print("正在加载Base Model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)


# =========================
# 6. 加载V7 LoRA
# =========================

print("正在加载V7 LoRA Adapter...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

model.eval()

print("V7 QLoRA加载完成")
print("模型设备：", model.device)


# =========================
# 7. System Prompt
# =========================

SYSTEM_PROMPT = """你是一名专业的五金工具电商客服。
回答要准确、简洁、实用。
不确定的产品参数不要擅自编造。"""


# =========================
# 8. 生成答案
# =========================

def generate_answer(question):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # Chat Template
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # Tokenize
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt"
    )

    # 移动到模型设备
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    # 生成
    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 只截取新生成内容
    input_length = inputs["input_ids"].shape[-1]

    generated_tokens = outputs[0][input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# =========================
# 9. 开始测试
# =========================

results = []

for i, item in enumerate(test_data, start=1):

    question = item["question"]

    print()
    print("-" * 60)
    print(f"[{i}/{len(test_data)}]")
    print("问题：", question)

    answer = generate_answer(question)

    print("V7答案：", answer)

    results.append({
        "id": item["id"],
        "category": item["category"],
        "question": question,
        "v7_answer": answer
    })


# =========================
# 10. 保存结果
# =========================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("=" * 60)
print("V7 QLoRA测试完成")
print("结果保存：", OUTPUT_FILE)
print("=" * 60)