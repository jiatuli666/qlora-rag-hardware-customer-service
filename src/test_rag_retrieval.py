import json
import os

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. 路径
# ============================================================

BASE_DIR = r"D:\QLoRA"

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


# ============================================================
# 2. Embedding 模型
# ============================================================

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

print("=" * 70)
print("加载 RAG 检索系统")
print("=" * 70)

print("\n正在加载 Embedding 模型...")

embedding_model = SentenceTransformer(MODEL_NAME)


# ============================================================
# 3. 加载 FAISS
# ============================================================

print("\n正在加载 FAISS 索引...")

index = faiss.read_index(INDEX_FILE)

print(f"FAISS 向量数量：{index.ntotal}")
print(f"向量维度：{index.d}")


# ============================================================
# 4. 加载 Metadata
# ============================================================

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

print(f"Metadata 数量：{len(metadata)}")


# ============================================================
# 5. 测试问题
# ============================================================

questions = [
    "PH2批头是什么？",
    "PH2是不是代表2毫米？",
    "65mm和PH2有什么区别？",
    "200mm比65mm长多少？",
    "200mm和150mm批头相差多少？",
    "S2钢是不是不锈钢？",
    "S2为什么适合做批头？",
    "强磁批头有什么作用？",
    "双头批头是不是可以同时拧两颗螺丝？",
    "普通家具安装应该选择65mm还是200mm？",
]


# ============================================================
# 6. 检索函数
# ============================================================

def retrieve(question, top_k=3):

    # 生成问题向量
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype("float32")

    # FAISS 检索
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

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
# 7. 开始测试
# ============================================================

print("\n")
print("=" * 70)
print("开始 Top-3 RAG 检索测试")
print("=" * 70)


for i, question in enumerate(questions, start=1):

    print("\n" + "-" * 70)

    print(f"问题 {i}")
    print(f"用户问题：{question}")

    results = retrieve(
        question,
        top_k=3
    )

    for rank, result in enumerate(results, start=1):

        print(f"\nTop {rank}")
        print(f"Score    ：{result['score']:.4f}")
        print(f"ID       ：{result['id']}")
        print(f"类别     ：{result['category']}")
        print(f"标题     ：{result['title']}")
        print(f"知识内容 ：{result['content']}")


print("\n")
print("=" * 70)
print("RAG Top-3 检索测试完成")
print("=" * 70)