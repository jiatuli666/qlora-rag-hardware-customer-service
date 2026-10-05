import json
import torch
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel


# ============================================================
# 路径
# ============================================================

MODEL_NAME = "Qwen/Qwen3-0.6B"

BASE_DIR = Path(__file__).resolve().parent.parent

LORA_DIR = BASE_DIR / "outputs" / "qwen3-0.6b-qlora-v5"

HOLDOUT_FILE = BASE_DIR / "data" / "holdout.json"

OUTPUT_FILE = BASE_DIR / "data" / ("holdout_v5"
                                   "_results.json")


# ============================================================
# 检查路径
# ============================================================

print("=" * 60)
print("V5 QLoRA Holdout 测试")
print("=" * 60)

print("基础模型:", MODEL_NAME)
print("V5 Adapter:", LORA_DIR)
print("测试数据:", HOLDOUT_FILE)
print("输出文件:", OUTPUT_FILE)

if not LORA_DIR.exists():
    raise FileNotFoundError(
        f"找不到 V5 LoRA Adapter:\n{LORA_DIR}"
    )

if not HOLDOUT_FILE.exists():
    raise FileNotFoundError(
        f"找不到 Holdout 数据:\n{HOLDOUT_FILE}"
    )


# ============================================================
# 4bit量化配置
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


# ============================================================
# 加载基础模型
# ============================================================

print()
print("=" * 60)
print("加载 Qwen3-0.6B 基础模型")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)


# ============================================================
# 加载 V5 LoRA Adapter
# ============================================================

print()
print("=" * 60)
print("加载 V5 LoRA Adapter")
print("=" * 60)

model = PeftModel.from_pretrained(
    base_model,
    str(LORA_DIR)
)

model.eval()

print("V5 Adapter 加载完成")


# ============================================================
# 加载 Holdout
# ============================================================

with open(
    HOLDOUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    holdout_data = json.load(f)


print()
print("=" * 60)
print("开始 V5 Holdout 测试")
print("=" * 60)

print("Holdout测试题数量:", len(holdout_data))
print()


# ============================================================
# 推理函数
# ============================================================

def generate_answer(question):

    messages = [
        {
            "role": "system",
            "content": (
                "你是一名专业的五金工具电商客服，"
                "回答要准确、简洁、实用，"
                "不确定的产品参数不要擅自编造。"
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.0
        )

    # 只截取模型新生成的内容
    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================
# 批量测试
# ============================================================

results = []

for i, item in enumerate(
    holdout_data,
    start=1
):

    question = item["question"]

    print("-" * 60)
    print(f"[{i}/{len(holdout_data)}]")
    print("问题:", question)

    answer = generate_answer(question)

    print("V5回答:", answer)

    results.append({
        "id": i,
        "question": question,
        "v5_answer": answer
    })


# ============================================================
# 保存测试结果
# ============================================================

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


# ============================================================
# 测试完成
# ============================================================

print()
print("=" * 60)
print("V5 Holdout测试完成")
print("=" * 60)

print("测试题数量:", len(results))
print("结果保存:", OUTPUT_FILE)