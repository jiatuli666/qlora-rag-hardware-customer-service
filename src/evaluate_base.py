import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =========================
# 1. 路径配置
# =========================
MODEL_NAME = "Qwen/Qwen3-0.6B"

TEST_FILE = r"D:\QLoRA\data\final_test.json"
OUTPUT_FILE = r"D:\QLoRA\data\final_base_results.json"

# =========================
# 2. 加载测试集
# =========================
with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print("=" * 60)
print(f"测试题数量：{len(test_data)}")
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
# 4. 加载Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# =========================
# 5. 加载Base Model
# =========================
print("正在加载 Base Model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

model.eval()

print("Base Model加载完成")
print("模型设备：", model.device)

# =========================
# 6. System Prompt
# =========================
SYSTEM_PROMPT = """你是一名专业的五金工具电商客服。
回答要准确、简洁、实用。
不确定的产品参数不要擅自编造。"""

# =========================
# 7. 生成答案
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

    # 先生成聊天模板文本
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # 再进行tokenize
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

    # 只截取模型新生成的部分
    input_length = inputs["input_ids"].shape[-1]

    generated_tokens = outputs[0][input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# =========================
# 8. 开始测试
# =========================
results = []

for i, item in enumerate(test_data, start=1):

    question = item["question"]

    print()
    print("-" * 60)
    print(f"[{i}/{len(test_data)}]")
    print("问题：", question)

    answer = generate_answer(question)

    print("Base答案：", answer)

    results.append({
        "id": item["id"],
        "category": item["category"],
        "question": question,
        "base_answer": answer
    })


# =========================
# 9. 保存结果
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
print("Base Model测试完成")
print("结果保存：", OUTPUT_FILE)
print("=" * 60)