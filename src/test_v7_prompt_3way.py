import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel


# ============================================================
# 1. 基础配置
# ============================================================

MODEL_NAME = "Qwen/Qwen3-0.6B"

ADAPTER_PATH = r"D:\QLoRA\outputs\qwen3-0.6b-qlora-v7"

OUTPUT_FILE = r"D:\QLoRA\data\v7_prompt_3way_compare.json"


# ============================================================
# 2. 测试问题
# ============================================================

QUESTIONS = [
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
# 3. 三种 Prompt
# ============================================================

NORMAL_SYSTEM_PROMPT = """你是一名专业的五金工具电商客服，
回答要准确、简洁、实用。
不确定的产品参数不要擅自编造。"""


STRICT_SYSTEM_PROMPT = """你是一名专业的五金工具电商客服。

回答必须准确、简洁、实用。
不确定的产品参数不要擅自编造。

请遵守以下产品知识规则：

1. PH2是十字批头/螺丝槽的规格型号，不代表2毫米。
2. 65mm、100mm、150mm、200mm表示批头长度。
3. PH2与65mm、100mm等长度参数属于不同维度，不能混为一谈。
4. S2是批头常见的工具钢材质，不等同于不锈钢。
5. 双头批头是两端都有工作端的批头，不代表可以同时拧两颗螺丝。
6. 两用头不能解释为同时拧两颗螺丝，具体含义应根据实际产品结构判断。
7. 遇到长度计算时必须直接进行准确计算。

只输出最终客服答案，不要输出分析过程。"""


STRUCTURED_SYSTEM_PROMPT = """你是一名专业的五金工具电商客服。

你的任务是准确回答五金工具产品问题。

回答要求：
1. 准确
2. 简洁
3. 实用
4. 不确定的产品参数不要擅自编造

回答产品问题前，先判断问题属于哪一种产品属性：

- 型号：例如 PH1、PH2、PH3
- 长度：例如 65mm、100mm、150mm、200mm
- 材质：例如 S2
- 产品结构：例如双头批头、两用头
- 使用场景：例如家具安装、装修、维修

特别注意：

型号、长度、材质、产品结构属于不同维度，
不能将不同维度相互混淆。

例如：
PH2属于批头/螺丝槽型号，
65mm、100mm、150mm、200mm属于批头长度，
S2属于常见工具钢材质。

遇到数字计算问题：
必须直接进行准确计算。

遇到产品结构问题：
必须根据产品结构本身回答，
不要根据名称进行错误联想。

如果无法确定具体产品结构，
明确说明需要根据实际产品结构判断。

回答时只输出最终客服答案，
不要展示分析过程。"""


# ============================================================
# 4. 加载模型
# ============================================================

print("=" * 60)
print("加载 Qwen3-0.6B")
print("=" * 60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

print("\n" + "=" * 60)
print("加载 V7 LoRA Adapter")
print("=" * 60)

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

model.eval()

print("V7 Adapter加载完成")


# ============================================================
# 5. 单次生成函数
# ============================================================

def generate_answer(system_prompt, question):

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

    # 使用 apply_chat_template 生成文本
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # 再单独进行 tokenizer
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt"
    )

    # 移动到模型所在设备
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 只取新生成的部分
    input_length = inputs["input_ids"].shape[-1]

    generated_tokens = outputs[0][input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()

# ============================================================
# 6. 三组 Prompt 对照测试
# ============================================================

results = []

print("\n" + "=" * 60)
print("开始三组 Prompt 对照测试")
print("=" * 60)

for i, question in enumerate(QUESTIONS, start=1):

    print("\n" + "-" * 60)
    print(f"[{i}/10]")
    print(f"问题：{question}")

    # --------------------------------------------------------
    # 普通 Prompt
    # --------------------------------------------------------

    normal_answer = generate_answer(
        NORMAL_SYSTEM_PROMPT,
        question
    )

    print("\n【普通 Prompt】")
    print(normal_answer)

    # --------------------------------------------------------
    # 强约束 Prompt
    # --------------------------------------------------------

    strict_answer = generate_answer(
        STRICT_SYSTEM_PROMPT,
        question
    )

    print("\n【强约束 Prompt】")
    print(strict_answer)

    # --------------------------------------------------------
    # 结构化 Prompt
    # --------------------------------------------------------

    structured_answer = generate_answer(
        STRUCTURED_SYSTEM_PROMPT,
        question
    )

    print("\n【结构化 Prompt】")
    print(structured_answer)

    # --------------------------------------------------------
    # 保存结果
    # --------------------------------------------------------

    results.append({
        "id": i,
        "question": question,
        "normal_prompt_answer": normal_answer,
        "strict_prompt_answer": strict_answer,
        "structured_prompt_answer": structured_answer
    })


# ============================================================
# 7. 保存 JSON
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
# 8. 完成
# ============================================================

print("\n" + "=" * 60)
print("三组 Prompt 对照实验完成")
print("=" * 60)

print(f"测试数量：{len(QUESTIONS)}")
print(f"结果文件：{OUTPUT_FILE}")