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

OUTPUT_FILE = BASE_DIR / "data" / "v7_prompt_compare.json"


# ============================================================
# 测试问题
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
# 两套 System Prompt
# ============================================================

NORMAL_SYSTEM_PROMPT = (
    "你是一名专业的五金工具电商客服，"
    "回答要准确、简洁、实用，"
    "不确定的产品参数不要擅自编造。"
)


STRICT_SYSTEM_PROMPT = (
    "你是一名专业的五金工具电商客服。"
    "\n回答必须准确、简洁、实用。"
    "\n不确定的产品参数不要擅自编造。"
    "\n\n回答前必须遵守以下产品知识规则："
    "\n1. PH2是十字批头/螺丝槽的规格型号，不代表2毫米。"
    "\n2. 65mm、100mm、150mm、200mm表示批头长度。"
    "\n3. PH2与65mm、100mm等长度参数属于不同维度，不能混为一谈。"
    "\n4. S2是批头常见的工具钢材质，不等同于不锈钢。"
    "\n5. 双头批头是两端都有工作端的批头，不代表可以同时拧两颗螺丝。"
    "\n6. 两用头不能解释为同时拧两颗螺丝，具体含义应根据实际产品结构判断。"
    "\n7. 遇到长度计算时必须直接进行准确计算。"
)


# ============================================================
# 检查模型
# ============================================================

print("=" * 60)
print("V7 普通 Prompt vs 强约束 Prompt 对照实验")
print("=" * 60)

print("基础模型:", MODEL_NAME)
print("V7 Adapter:", LORA_DIR)
print("测试题数量:", len(TEST_QUESTIONS))

if not LORA_DIR.exists():
    raise FileNotFoundError(
        f"找不到 V7 LoRA Adapter:\n{LORA_DIR}"
    )


# ============================================================
# 4bit量化
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
print("加载 Qwen3-0.6B")
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
# 加载 V7 LoRA
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

print("V7 Adapter加载完成")


# ============================================================
# 推理函数
# ============================================================

def generate_answer(question, system_prompt):

    messages = [
        {
            "role": "system",
            "content": system_prompt
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

results = []

print()
print("=" * 60)
print("开始对照测试")
print("=" * 60)


for i, question in enumerate(
    TEST_QUESTIONS,
    start=1
):

    print()
    print("-" * 60)
    print(f"[{i}/{len(TEST_QUESTIONS)}]")
    print("问题:", question)

    # --------------------------------------------------------
    # 普通 Prompt
    # --------------------------------------------------------

    normal_answer = generate_answer(
        question,
        NORMAL_SYSTEM_PROMPT
    )

    print()
    print("【普通 Prompt】")
    print(normal_answer)

    # --------------------------------------------------------
    # 强约束 Prompt
    # --------------------------------------------------------

    strict_answer = generate_answer(
        question,
        STRICT_SYSTEM_PROMPT
    )

    print()
    print("【强约束 Prompt】")
    print(strict_answer)

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    results.append({
        "id": i,
        "question": question,
        "normal_prompt_answer": normal_answer,
        "strict_prompt_answer": strict_answer
    })


# ============================================================
# 保存 JSON
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
print("对照实验完成")
print("=" * 60)

print("测试数量:", len(results))
print("结果文件:", OUTPUT_FILE)