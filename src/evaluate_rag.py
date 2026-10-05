import json
import os
import torch
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
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

INDEX_FILE = r"D:\QLoRA\knowledge_base\vector_db\hardware.index"

METADATA_FILE = r"D:\QLoRA\knowledge_base\vector_db\metadata.json"

OUTPUT_FILE = r"D:\QLoRA\data\final_rag_results.json"

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

TOP_K = 3


# =========================
# 2. 读取测试集和知识库
# =========================

with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("=" * 60)
print("最终测试题数量：", len(test_data))
print("知识库条目数量：", len(metadata))
print("=" * 60)


# =========================
# 3. 加载Embedding模型
# =========================

print("正在加载Embedding模型...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# =========================
# 4. 加载FAISS索引
# =========================

print("正在加载FAISS索引...")

index = faiss.read_index(INDEX_FILE)

print("FAISS索引加载完成")
print("向量数量：", index.ntotal)


# =========================
# 5. 加载Tokenizer
# =========================

print("正在加载Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# =========================
# 6. 量化配置
# =========================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


# =========================
# 7. 加载模型
# =========================

print("正在加载Base Model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

print("正在加载V7 LoRA...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

model.eval()

print("V7 + RAG模型加载完成")


# =========================
# 8. 检索函数
# =========================

def retrieve_knowledge(question):

    query_vector = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    query_vector = np.asarray(
        query_vector,
        dtype="float32"
    )

    scores, indices = index.search(
        query_vector,
        TOP_K
    )

    retrieved = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        item = metadata[idx]

        retrieved.append({
            "id": item.get("id", ""),
            "category": item.get("category", ""),
            "title": item.get("title", ""),
            "content": item.get("content", ""),
            "score": float(score)
        })

    return retrieved


# =========================
# 9. 生成回答
# =========================

SYSTEM_PROMPT = """你是一名专业的五金工具电商客服。

回答要求：
1. 优先依据提供的知识库回答。
2. 回答准确、简洁、实用。
3. 不确定的产品参数不要擅自编造。
4. 不要混淆批头型号、长度、材质和产品结构。
5. 如果知识库明确说明某个概念，严格按照知识库回答。
6. 如果知识库没有足够信息，请明确说明无法确定，不要猜测。
"""


def generate_answer(question, knowledge):

    knowledge_text = ""

    for i, item in enumerate(knowledge, start=1):

        knowledge_text += (
            f"\n知识{i}：\n"
            f"类别：{item['category']}\n"
            f"标题：{item['title']}\n"
            f"内容：{item['content']}\n"
        )

    user_prompt = f"""请根据以下知识库内容回答用户问题。

【知识库】
{knowledge_text}

【用户问题】
{question}

请优先使用知识库中的明确事实。
如果知识库没有足够信息，不要自行编造。
请用简洁的中文回答，控制在1到3句话。
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt"
    )

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

    input_length = inputs["input_ids"].shape[-1]

    generated_tokens = outputs[0][input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# =========================
# 10. 开始测试
# =========================

results = []

for i, item in enumerate(test_data, start=1):

    question = item["question"]

    print()
    print("-" * 60)
    print(f"[{i}/{len(test_data)}]")
    print("问题：", question)

    knowledge = retrieve_knowledge(question)

    print("检索结果：")

    for k in knowledge:
        print(
            f"  {k['id']} | "
            f"{k['title']} | "
            f"相似度：{k['score']:.4f}"
        )

    answer = generate_answer(
        question,
        knowledge
    )

    print("RAG答案：", answer)

    results.append({
        "id": item["id"],
        "category": item["category"],
        "question": question,
        "retrieved_knowledge": knowledge,
        "rag_answer": answer
    })


# =========================
# 11. 保存结果
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
print("V7 + RAG测试完成")
print("结果保存：", OUTPUT_FILE)
print("=" * 60)