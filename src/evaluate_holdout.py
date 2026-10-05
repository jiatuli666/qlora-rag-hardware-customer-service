import json
import os
import gc
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

V1_LORA = "outputs/qwen3-0.6b-qlora"
V2_LORA = "outputs/qwen3-0.6b-qlora-v2"

TEST_PATH = "data/holdout_test.json"

OUTPUT_DIR = "outputs/holdout_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "holdout_results.json"
)

TXT_PATH = os.path.join(
    OUTPUT_DIR,
    "holdout_results.txt"
)


# ============================================================
# 2. 4bit配置
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
# 4. 加载测试集
# ============================================================

with open(TEST_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"Holdout测试集：{len(test_data)} 条")


# ============================================================
# 5. 提取问题和标准答案
# ============================================================

def extract_question(item):
    for message in item["messages"]:
        if message["role"] == "user":
            return message["content"]
    return ""


def extract_reference(item):
    for message in item["messages"]:
        if message["role"] == "assistant":
            return message["content"]
    return ""


# ============================================================
# 6. 加载基础模型
# ============================================================

def load_base_model():

    print("\n加载基础模型...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model.eval()

    print("基础模型加载完成")

    return model


# ============================================================
# 7. 加载LoRA模型
# ============================================================

def load_lora_model(lora_path):

    base_model = load_base_model()

    print(f"加载 LoRA：{lora_path}")

    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
    )

    model.eval()

    print("LoRA加载完成")

    return model


# ============================================================
# 8. 生成答案
# ============================================================

def generate_answer(model, question):

    messages = [
        {
            "role": "system",
            "content": (
                "你是一名专业的五金工具电商客服，"
                "回答要准确、简洁、实用，"
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
        return_tensors="pt",
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
# 9. 测试一个LoRA
# ============================================================

def evaluate_lora(lora_path, model_name):

    print("\n")
    print("=" * 70)
    print(f"开始测试：{model_name}")
    print("=" * 70)

    model = load_lora_model(lora_path)

    results = []

    for i, item in enumerate(test_data, 1):

        question = extract_question(item)
        reference = extract_reference(item)

        print(f"\n[{i}/{len(test_data)}]")
        print("问题：", question)

        answer = generate_answer(
            model,
            question
        )

        print("模型回答：", answer)

        results.append({
            "index": i,
            "question": question,
            "reference_answer": reference,
            "answer": answer,
        })

    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ============================================================
# 10. V1
# ============================================================

v1_results = evaluate_lora(
    V1_LORA,
    "V1 QLoRA"
)


# ============================================================
# 11. V2
# ============================================================

v2_results = evaluate_lora(
    V2_LORA,
    "V2 QLoRA"
)


# ============================================================
# 12. 合并结果
# ============================================================

final_results = []

for i in range(len(test_data)):

    final_results.append({
        "index": i + 1,
        "question": test_data[i]["messages"][1]["content"],
        "reference_answer": v1_results[i]["reference_answer"],
        "v1_answer": v1_results[i]["answer"],
        "v2_answer": v2_results[i]["answer"],
    })


# ============================================================
# 13. 保存JSON
# ============================================================

with open(
    JSON_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_results,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 14. 保存TXT
# ============================================================

with open(
    TXT_PATH,
    "w",
    encoding="utf-8"
) as f:

    for item in final_results:

        f.write("=" * 70 + "\n")
        f.write(f"问题 {item['index']}\n\n")

        f.write(
            f"问题：\n"
            f"{item['question']}\n\n"
        )

        f.write(
            f"标准答案：\n"
            f"{item['reference_answer']}\n\n"
        )

        f.write(
            f"V1 QLoRA：\n"
            f"{item['v1_answer']}\n\n"
        )

        f.write(
            f"V2 QLoRA：\n"
            f"{item['v2_answer']}\n\n"
        )


# ============================================================
# 15. 完成
# ============================================================

print("\n")
print("=" * 70)
print("Holdout评估完成")
print("=" * 70)

print(f"JSON：{JSON_PATH}")
print(f"TXT ：{TXT_PATH}")