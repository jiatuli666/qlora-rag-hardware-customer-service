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

LORA_DIR = BASE_DIR / "outputs" / "qwen3-0.6b-qlora-v7"

OUTPUT_FILE = BASE_DIR / "data" / "v7_key_results.json"


# ============================================================
# 重点测试问题
# ============================================================

TEST_QUESTIONS = [

    "PH2批头是什么？",

    "PH2是不是代表2毫米？",

    "65mm是不是PH2的尺寸？",

    "200mm批头比65mm批头长多少？",

    "200mm和150mm批头相差多少？",

    "S2钢是不是不锈钢？",

    "S2材质为什么经常用于批头？",

    "双头批头是不是可以同时拧两颗螺丝？",

    "两用头是不是就是一根批头同时拧两颗螺丝？",

    "普通家具安装一般更适合65mm还是200mm？"
]


# ============================================================
# 检查路径
# ============================================================

print("=" * 60)
print("V7 QLoRA 重点纠错测试")
print("=" * 60)

print("基础模型:", MODEL_NAME)
print("V7 Adapter:", LORA_DIR)
print("测试题数量:", len(TEST_QUESTIONS))
print("输出文件:", OUTPUT_FILE)

if not LORA_DIR.exists():
    raise FileNotFoundError(
        f"找不到 V7 LoRA Adapter:\n{LORA_DIR}"
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
# 加载 V7 LoRA Adapter
# ============================================================

print()
print("=" * 60)
print("加载 V7 LoRA Adapter")
print("=" * 60)

model = PeftModel.from_pretrained(
    base_model,
    str(LORA_DIR)
)

model.eval()

print("V7 Adapter 加载完成")


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
# 开始测试
# ============================================================

print()
print("=" * 60)
print("开始 V7 重点纠错测试")
print("=" * 60)
print()


results = []

for i, question in enumerate(
    TEST_QUESTIONS,
    start=1
):

    print("-" * 60)
    print(f"[{i}/{len(TEST_QUESTIONS)}]")
    print("问题:", question)

    answer = generate_answer(question)

    print("V7回答:", answer)

    results.append({
        "id": i,
        "question": question,
        "v7_answer": answer
    })


# ============================================================
# 保存结果
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
# 完成
# ============================================================

print()
print("=" * 60)
print("V7重点纠错测试完成")
print("=" * 60)

print("测试题数量:", len(results))
print("结果保存:", OUTPUT_FILE)