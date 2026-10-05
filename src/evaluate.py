import json
import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


# ============================================================
# 1. 配置
# ============================================================

MODEL_NAME = "Qwen/Qwen3-0.6B"
#LORA_PATH = "outputs/qwen3-0.6b-qlora"
#LORA_PATH = "outputs/qwen3-0.6b-qlora-v2"
LORA_PATH = "outputs/qwen3-0.6b-qlora-v4"

TEST_PATH = "data/test.json"

#OUTPUT_DIR = "outputs/evaluation"
OUTPUT_DIR = "outputs/evaluation_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

JSON_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "evaluation_results.json"
)

TXT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "evaluation_results.txt"
)


# ============================================================
# 2. 4bit量化配置
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 3. 加载Tokenizer
# ============================================================

print("=" * 70)
print("加载 Tokenizer")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# 4. 加载基础模型
# ============================================================

print("\n加载基础模型...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

base_model.eval()

print("基础模型加载完成")


# ============================================================
# 5. 加载测试集
# ============================================================

print("\n加载测试集...")

with open(
    TEST_PATH,
    "r",
    encoding="utf-8"
) as f:
    test_data = json.load(f)

print(f"测试集数量：{len(test_data)}")


# ============================================================
# 6. 提取用户问题和标准答案
# ============================================================

def extract_question_and_answer(item):

    question = ""
    reference_answer = ""

    for message in item["messages"]:

        if message["role"] == "user":
            question = message["content"]

        elif message["role"] == "assistant":
            reference_answer = message["content"]

    return question, reference_answer


# ============================================================
# 7. 生成回答
# ============================================================

def generate_answer(model, question):

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

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

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

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================
# 8. 先测试基础模型
# ============================================================

print("\n" + "=" * 70)
print("开始测试 Base Model")
print("=" * 70)

base_results = []

for index, item in enumerate(test_data, start=1):

    question, reference = extract_question_and_answer(item)

    print(
        f"\n[{index}/{len(test_data)}] "
        f"测试问题：{question}"
    )

    answer = generate_answer(
        base_model,
        question
    )

    print(
        "Base Model：",
        answer
    )

    base_results.append({
        "question": question,
        "reference_answer": reference,
        "base_answer": answer,
    })


# ============================================================
# 9. 加载LoRA Adapter
# ============================================================

print("\n" + "=" * 70)
print("加载 QLoRA Adapter")
print("=" * 70)

qlora_model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
)

qlora_model.eval()

print("QLoRA Adapter 加载完成")


# ============================================================
# 10. 测试QLoRA模型
# ============================================================

print("\n" + "=" * 70)
print("开始测试 QLoRA Model")
print("=" * 70)

results = []

for index, item in enumerate(test_data, start=1):

    question, reference = extract_question_and_answer(item)

    print(
        f"\n[{index}/{len(test_data)}] "
        f"测试问题：{question}"
    )

    qlora_answer = generate_answer(
        qlora_model,
        question
    )

    print(
        "QLoRA Model：",
        qlora_answer
    )

    result = {
        "index": index,
        "question": question,
        "reference_answer": reference,
        "base_answer": base_results[index - 1]["base_answer"],
        "qlora_answer": qlora_answer,
    }

    results.append(result)


# ============================================================
# 11. 保存JSON
# ============================================================

with open(
    JSON_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 12. 保存TXT
# ============================================================

with open(
    TXT_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    for result in results:

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            f"问题 {result['index']}\n\n"
        )

        f.write(
            f"问题：\n"
            f"{result['question']}\n\n"
        )

        f.write(
            f"标准答案：\n"
            f"{result['reference_answer']}\n\n"
        )

        f.write(
            f"Base Model：\n"
            f"{result['base_answer']}\n\n"
        )

        f.write(
            f"QLoRA Model：\n"
            f"{result['qlora_answer']}\n\n"
        )


print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)

print(f"\nJSON结果：{JSON_OUTPUT}")
print(f"TXT结果：{TXT_OUTPUT}")