import json
import os

import faiss
import torch

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel


# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = r"D:\QLoRA"

MODEL_NAME = "Qwen/Qwen3-0.6B"

ADAPTER_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "qwen3-0.6b-qlora-v7"
)

INDEX_FILE = os.path.join(
    BASE_DIR,
    "knowledge_base",
    "vector_db",
    "hardware.index"
)

METADATA_FILE = os.path.join(
    BASE_DIR,
    "knowledge_base",
    "vector_db",
    "metadata.json"
)

RESULT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "rag_v7_results.json"
)


# ============================================================
# 2. 系统提示词
# ============================================================

SYSTEM_PROMPT = """
你是一名专业的五金工具电商客服。

回答要求：
1. 优先依据提供的知识库回答。
2. 回答准确、简洁、实用。
3. 不确定的产品参数不要擅自编造。
4. 不要把批头型号、批头长度、材质等不同参数混淆。
5. 如果知识库明确说明某个概念，严格按照知识库回答。
"""


# ============================================================
# 3. 加载 Embedding 模型
# ============================================================

print("=" * 70)
print("加载 RAG + V7 QLoRA 测试系统")
print("=" * 70)

print("\n[1/5] 加载 Embedding 模型...")

embedding_model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5"
)


# ============================================================
# 4. 加载 FAISS
# ============================================================

print("\n[2/5] 加载 FAISS 索引...")

index = faiss.read_index(INDEX_FILE)

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

print(f"FAISS 向量数量：{index.ntotal}")
print(f"Metadata 数量：{len(metadata)}")


# ============================================================
# 5. 加载 Qwen3 + V7 LoRA
# ============================================================

print("\n[3/5] 加载 Qwen3-0.6B...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

print("\n加载 V7 LoRA Adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

print("V7 LoRA 加载完成")


# ============================================================
# 6. RAG 检索
# ============================================================

def retrieve(question, top_k=3):

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        item = metadata[idx]

        results.append({
            "score": float(score),
            "id": item["id"],
            "category": item["category"],
            "title": item["title"],
            "content": item["content"]
        })

    return results


# ============================================================
# 7. 生成回答
# ============================================================

def generate_answer(
    question,
    retrieved_docs
):

    # --------------------------------------------------------
    # 构造知识库上下文
    # --------------------------------------------------------

    knowledge_text = ""

    for i, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        knowledge_text += (
            f"\n【知识{i}】\n"
            f"类别：{doc['category']}\n"
            f"标题：{doc['title']}\n"
            f"内容：{doc['content']}\n"
        )

    # --------------------------------------------------------
    # 构造 Prompt
    # --------------------------------------------------------

    user_prompt = f"""
请回答下面的用户问题。

用户问题：
{question}

以下是从五金工具知识库中检索到的相关知识：
{knowledge_text}

回答要求：
- 优先依据上述知识回答。
- 不要自行改变知识中的事实。
- 如果知识库已经明确说明答案，直接根据知识回答。
- 不要编造知识库没有提供的参数。
- 回答控制在1到3句话。
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

    # --------------------------------------------------------
    # Chat Template
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 生成
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs[
        "input_ids"
    ].shape[-1]

    generated_tokens = outputs[
        0
    ][input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================
# 8. 测试问题
# ============================================================

questions = [

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
# 9. 开始测试
# ============================================================

print("\n[4/5] 开始 RAG + V7 测试")

print("=" * 70)


results = []


for i, question in enumerate(
    questions,
    start=1
):

    print("\n" + "-" * 70)

    print(f"问题 {i}：{question}")

    # --------------------------------------------------------
    # RAG 检索
    # --------------------------------------------------------

    retrieved_docs = retrieve(
        question,
        top_k=3
    )

    print("\nTop-3 检索结果：")

    for rank, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        print(
            f"Top {rank} | "
            f"Score={doc['score']:.4f} | "
            f"{doc['id']} | "
            f"{doc['title']}"
        )

    # --------------------------------------------------------
    # V7 生成
    # --------------------------------------------------------

    answer = generate_answer(
        question,
        retrieved_docs
    )

    print("\nRAG + V7回答：")
    print(answer)

    # --------------------------------------------------------
    # 保存结果
    # --------------------------------------------------------

    results.append({

        "id": i,

        "question": question,

        "retrieved": retrieved_docs,

        "answer": answer
    })


# ============================================================
# 10. 保存测试结果
# ============================================================

with open(
    RESULT_FILE,
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
# 11. 完成
# ============================================================

print("\n")
print("=" * 70)

print("RAG + V7 测试完成")

print("=" * 70)

print(f"\n测试结果已保存：")

print(RESULT_FILE)