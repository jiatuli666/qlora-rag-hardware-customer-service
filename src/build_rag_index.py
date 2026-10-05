import json
import os

import faiss
from sentence_transformers import SentenceTransformer


# =========================
# 1. 路径配置
# =========================

BASE_DIR = r"D:\QLoRA"

KNOWLEDGE_FILE = os.path.join(
    BASE_DIR,
    "knowledge_base",
    "raw",
    "hardware_knowledge.json"
)

VECTOR_DIR = os.path.join(
    BASE_DIR,
    "knowledge_base",
    "vector_db"
)

INDEX_FILE = os.path.join(
    VECTOR_DIR,
    "hardware.index"
)

METADATA_FILE = os.path.join(
    VECTOR_DIR,
    "metadata.json"
)


# =========================
# 2. 创建向量数据库目录
# =========================

os.makedirs(VECTOR_DIR, exist_ok=True)


# =========================
# 3. 检查知识库文件
# =========================

if not os.path.exists(KNOWLEDGE_FILE):
    raise FileNotFoundError(
        f"找不到知识库文件：\n{KNOWLEDGE_FILE}"
    )

print("=" * 60)
print("开始构建 RAG 向量索引")
print("=" * 60)

print(f"\n知识库文件：{KNOWLEDGE_FILE}")


# =========================
# 4. 读取知识库
# =========================

with open(
    KNOWLEDGE_FILE,
    "r",
    encoding="utf-8"
) as f:
    knowledge_data = json.load(f)

print(f"知识库数量：{len(knowledge_data)}")


# =========================
# 5. 加载 Embedding 模型
# =========================

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

print("\n正在加载 Embedding 模型：")
print(MODEL_NAME)

embedding_model = SentenceTransformer(MODEL_NAME)


# =========================
# 6. 构造文本
# =========================

texts = []
metadata = []

for item in knowledge_data:

    category = item.get("category", "")
    title = item.get("title", "")
    content = item.get("content", "")

    text = (
        f"类别：{category}\n"
        f"标题：{title}\n"
        f"内容：{content}"
    )

    texts.append(text)

    metadata.append({
        "id": item.get("id", ""),
        "category": category,
        "title": title,
        "content": content
    })


print(f"\n待向量化文本数量：{len(texts)}")


# =========================
# 7. 生成 Embedding
# =========================

print("\n开始生成向量...")

embeddings = embedding_model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

print(f"向量维度：{embeddings.shape}")


# =========================
# 8. 创建 FAISS 索引
# =========================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings.astype("float32"))


print("\nFAISS 索引创建完成")
print(f"向量数量：{index.ntotal}")
print(f"向量维度：{dimension}")


# =========================
# 9. 保存 FAISS 索引
# =========================

faiss.write_index(
    index,
    INDEX_FILE
)

print(f"\nFAISS 索引已保存：")
print(INDEX_FILE)


# =========================
# 10. 保存 Metadata
# =========================

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nMetadata 已保存：")
print(METADATA_FILE)


# =========================
# 11. 完成
# =========================

print("\n" + "=" * 60)
print("RAG 向量索引构建完成！")
print("=" * 60)