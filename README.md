3. 为什么同时使用 QLoRA 和 RAG？

本项目没有简单地把所有知识都交给模型进行微调，而是将：

模型能力学习 与 外部知识检索 分开处理。

3.1 QLoRA 的作用

QLoRA 主要用于学习：

五金工具客服问答任务
客服回答风格
产品问题的表达方式
简洁、准确的回答形式
常见问答模式

通过 QLoRA 对 Qwen3-0.6B 进行参数高效微调，使基础模型更适合五金工具客服场景。

3.2 RAG 的作用

RAG 主要用于解决：

产品具体规格
PH2 等型号信息
65mm / 100mm / 150mm / 200mm 长度
S2 材质相关知识
磁性及磁圈相关知识
双头批头相关知识
使用场景及规格选择

等需要依赖外部知识库的问题。

系统首先检索知识库，再将检索结果提供给大模型生成回答。

3.3 两者结合

最终形成：

QLoRA
  ↓
学习“如何回答”
  
RAG
  ↓
提供“回答什么”

        ↓

Qwen3-0.6B
        ↓
生成最终答案

这种设计能够减少单纯依赖模型参数记忆产品事实所带来的幻觉问题。

4. 技术栈
模型
Qwen3-0.6B
BAAI/bge-small-zh-v1.5
微调技术
QLoRA
LoRA
PEFT
Transformers
TRL
RAG
FAISS
BGE Embedding
Top-K 相似度检索
本地 JSON 知识库
深度学习环境
PyTorch
CUDA
NVIDIA RTX 3060 Laptop GPU
Python 3.11.9
工程工具
Git
GitHub
PowerShell
Virtual Environment
5. QLoRA 微调实验

本项目使用 QLoRA 对 Qwen3-0.6B 进行垂直领域微调。

5.1 训练数据

V7 版本训练数据：

训练数据量：406 条
训练轮数：3 epochs
训练步数：306 steps

训练数据主要覆盖：

PH2 规格
批头长度
S2 材质
磁性
双头批头
两用头
使用场景
批头磨损
防锈
螺丝匹配
电钻使用
长度选择
基础计算问题
5.2 LoRA 参数

本项目采用参数高效微调方式，仅训练 LoRA Adapter，而不是更新整个基础模型。

此前训练实验中：

Trainable Parameters：2,293,760
All Parameters：598,343,680
Trainable Ratio：0.3834%

这种方式能够显著降低微调所需的显存和计算资源。

6. QLoRA V7 训练结果

V7 版本训练结果：

指标	结果
Training Samples	406
Epochs	3
Steps	306
Train Loss	1.365
Eval Loss	1.111
Mean Token Accuracy	0.7702
GPU	RTX 3060 Laptop
训练时间	约 11.57 分钟

训练完成后生成：

outputs/
└── qwen3-0.6b-qlora-v7/

其中保存 V7 LoRA Adapter。

7. RAG 知识库

项目建立了五金工具领域本地知识库。

知识库主要覆盖：

PH2规格
长度规格
长度选择
S2材质
磁性
磁圈
双头批头
两用头
磨损
防锈
使用场景
螺丝匹配

知识库原始文件：

knowledge_base/
└── raw/
    └── hardware_knowledge.json
8. BGE 向量化

使用：

BAAI/bge-small-zh-v1.5

将知识库中的文本转换成向量。

向量维度：

512

同时对向量进行归一化处理，为后续相似度检索提供基础。

9. FAISS 向量检索

项目使用 FAISS 建立本地向量索引：

IndexFlatIP

用户输入问题后：

用户问题
   ↓
BGE Embedding
   ↓
512维向量
   ↓
FAISS
   ↓
Top-K 相似知识

最终将检索到的知识作为 Context 提供给大模型。

索引文件：

knowledge_base/
└── vector_db/
    ├── hardware.index
    └── metadata.json
10. RAG 开发测试

在开发阶段使用 10 个测试问题对知识库检索能力进行了测试。

测试结果：

Top-1 命中：10 / 10

即开发测试阶段的 Top-1 检索命中率为：

100%

注意：该结果属于开发阶段的检索测试结果，并不代表最终智能客服系统的整体准确率。

最终模型效果仍需要结合生成答案进行独立测试。

11. 独立 Final Test

为了避免使用训练数据直接评价模型，项目另外建立了独立测试集：

data/
└── final_test.json

测试集：

40 个问题

覆盖：

类别	数量
PH规格	6
长度规格	6
S2材质	6
磁性	5
双头批头	4
两用头	3
使用场景	4
磨损 / 打滑	3
计算问题	3
合计	40
12. 模型评价方法

采用人工规则评分方式。

每道题：

2分：核心事实正确，无明显错误

1分：部分正确，但信息不完整或表达存在歧义

0分：核心概念错误、事实错误或出现明显编造

总分：

40 × 2 = 80分

最终得分：

模型得分 / 80 × 100%

这种评价方式重点关注：

核心事实是否正确
是否存在概念混淆
是否出现产品知识幻觉
是否能够正确使用检索到的知识
是否能够进行正确的基础计算
13. Base Model / QLoRA / RAG 对比

项目对以下三个版本进行了对比：

Base Model
    ↓
Qwen3-0.6B

QLoRA Model
    ↓
Qwen3-0.6B + V7 LoRA

QLoRA + RAG
    ↓
Qwen3-0.6B + V7 LoRA + FAISS Knowledge Base
Base Model

主要问题：

缺乏五金工具领域知识
部分专业概念理解错误
对产品参数存在幻觉
某些基础计算可能出现错误
QLoRA Model

相比 Base Model：

客服回答风格更加稳定
五金工具领域表达更加自然
部分简单产品问题回答有所改善
能够学习训练数据中的问答模式

但独立测试仍发现部分专业概念错误。

这说明：

仅依赖 QLoRA 并不能保证模型始终正确掌握具体产品事实。

QLoRA + RAG

加入 RAG 后：

用户问题
   ↓
知识库检索
   ↓
获取相关产品知识
   ↓
Qwen3 + LoRA
   ↓
生成回答

能够进一步改善部分产品事实类问题。

实验也发现，RAG 并不是“加入知识库就一定正确”，如果：

检索结果不够准确
Prompt 约束不足
模型没有正确利用 Context
知识库本身存在表达歧义

仍然可能产生错误回答。

因此后续优化重点放在：

数据质量 + 检索质量 + Prompt + 模型生成能力

四个方面。

14. 项目中遇到的问题与解决方案
问题一：RTX 3060 显存有限

普通全参数微调对个人电脑显存要求较高。

解决方案：

采用：

QLoRA + LoRA

只训练少量 Adapter 参数，降低训练资源需求。

问题二：FP16 训练过程中出现梯度缩放问题

早期训练过程中出现：

torch.amp.grad_scaler

相关错误。

经过排查后，将 Accelerate 的 mixed precision 从：

fp16

调整为：

no

之后完成正式训练。

问题三：QLoRA 模型仍然存在专业概念错误

V7 模型虽然训练 loss 和评估指标有所改善，但独立测试中仍出现：

双头批头概念理解错误
两用头概念表达不准确
长度选择问题回答不够准确
个别计算问题出现额外错误信息

因此没有继续简单地通过增加训练轮数解决问题，而是引入 RAG。

问题四：模型缺少实时产品知识

通过建立：

五金工具知识库
        ↓
BGE Embedding
        ↓
FAISS
        ↓
Top-K Retrieval
        ↓
RAG Context

将产品知识从模型参数中分离出来，使产品信息可以通过知识库进行维护。

15. 项目目录
QLoRA/
│
├── data/
│   ├── train_v7.json
│   └── final_test.json
│
├── knowledge_base/
│   ├── raw/
│   │   └── hardware_knowledge.json
│   │
│   └── vector_db/
│       ├── hardware.index
│       └── metadata.json
│
├── src/
│   ├── train.py
│   ├── ...
│
├── outputs/
│   └── qwen3-0.6b-qlora-v7/
│
├── main.py
├── README.md
└── .gitignore

outputs/、模型权重以及虚拟环境等本地大文件不会提交到 GitHub。

16. 环境配置

推荐：

Python 3.11
CUDA
NVIDIA GPU

本项目实际训练环境：

OS：Windows 10
Python：3.11.9
PyTorch：2.6.0+cu126
Transformers：5.17.0
Datasets：5.0.1
PEFT：0.20.0
TRL：1.13.0
Accelerate：1.15.0
GPU：NVIDIA GeForce RTX 3060 Laptop GPU
17. 安装依赖

创建虚拟环境：

python -m venv qlora_env

激活：

.\qlora_env\Scripts\Activate.ps1

安装项目依赖：

pip install torch transformers datasets peft trl accelerate bitsandbytes

如果使用 RAG，还需要：

pip install sentence-transformers faiss-cpu
18. 运行流程
第一步：准备训练数据
data/train_v7.json
第二步：执行 QLoRA 微调
python src/train.py

训练完成后生成 LoRA Adapter。

第三步：建立知识库向量索引

将：

knowledge_base/raw/hardware_knowledge.json

转换为向量并建立 FAISS 索引。

最终生成：

knowledge_base/vector_db/
├── hardware.index
└── metadata.json
第四步：运行智能客服
python main.py

系统执行：

用户问题
   ↓
BGE向量化
   ↓
FAISS检索
   ↓
获取Top-K知识
   ↓
构造Prompt
   ↓
Qwen3-0.6B + LoRA
   ↓
生成最终答案
19. 项目核心亮点
亮点一：低显存环境完成大模型微调

基于 RTX 3060 Laptop GPU 完成 QLoRA 微调实验。

亮点二：QLoRA + RAG 组合架构

不是单纯训练模型，而是将：

模型微调
+
知识库检索
+
大模型生成

结合起来。

亮点三：独立测试集评价

没有直接使用训练数据作为最终评价依据，而是建立：

40题 Final Test

并采用 0 / 1 / 2 分人工评分机制评价模型。

亮点四：定位并分析模型错误

不仅关注 loss，还针对模型回答进行人工分析，发现：

专业概念混淆
知识幻觉
RAG 检索与生成之间的信息利用问题
Prompt 约束不足

并据此调整技术路线。

亮点五：完整工程化流程

项目覆盖：

数据构建
 ↓
数据质量检查
 ↓
QLoRA微调
 ↓
模型测试
 ↓
知识库构建
 ↓
Embedding
 ↓
FAISS检索
 ↓
RAG
 ↓
独立测试
 ↓
Git版本管理
 ↓
GitHub项目展示
20. 后续优化方向

当前版本仍存在进一步优化空间。

20.1 优化训练数据

增加：

Hard Negative
专业概念对比问答
容易产生幻觉的问题
多轮客服对话
产品规格判断问题

重点提高模型对：

双头
两用
PH规格
长度选择
磁圈
批头本体

等容易混淆概念的理解能力。

20.2 优化 RAG

后续可以加入：

Hybrid Search
BM25
Reranker
Query Rewrite
Metadata Filtering
Top-K 参数调优

进一步提高检索准确率。

20.3 优化 Prompt

增加更加严格的回答规则：

1. 优先使用检索到的知识
2. 知识库没有的信息不要编造
3. 涉及具体规格时必须以知识库为准
4. 不确定时明确说明
5. 避免输出与问题无关的信息
20.4 完善评测体系

后续可以将人工评分扩展为：

事实准确率
检索命中率
答案完整度
幻觉率
回答相关性

并建立自动化评测脚本。

21. 项目总结

本项目完整实践了一个垂直领域大模型应用从：

数据
 ↓
模型微调
 ↓
知识库
 ↓
向量检索
 ↓
RAG
 ↓
模型生成
 ↓
效果评估

的完整流程。

通过本项目掌握了：

QLoRA 参数高效微调
LoRA Adapter
Hugging Face Transformers
PEFT
BGE Embedding
FAISS 向量检索
RAG
Prompt Engineering
数据质量检查
模型效果评估
Git / GitHub 项目管理

项目最终形成：

Qwen3-0.6B + QLoRA + BGE + FAISS + RAG

的垂直领域智能客服系统。

22. 简历项目描述
基于 QLoRA 与 RAG 的五金工具垂直领域智能客服系统

项目技术： Python / PyTorch / Transformers / PEFT / QLoRA / BGE / FAISS / RAG / Git

针对五金工具电商客服场景，基于 Qwen3-0.6B 构建垂直领域智能问答系统，覆盖 PH2、S2 材质、批头长度、磁性及使用场景等产品知识。
使用 QLoRA 进行参数高效微调，构建 406 条领域训练数据，训练 3 epochs、306 steps，V7 模型 Train Loss 1.365、Eval Loss 1.111。
基于 BGE-small-zh-v1.5 构建 512 维文本向量，使用 FAISS IndexFlatIP 建立本地知识库，实现 Top-K 产品知识检索。
将 QLoRA 与 RAG 结合，使模型学习客服问答表达方式，同时通过外部知识库提供具体产品事实，降低单纯依赖模型参数产生产品知识幻觉的问题。
构建 40 条独立 Final Test，采用 0/1/2 分规则进行 Base Model、QLoRA Model 与 QLoRA + RAG 对比评测，并针对模型错误进行数据、检索和 Prompt 层面的分析。
在 RTX 3060 Laptop GPU 环境完成完整微调及 RAG 实验，并使用 Git/GitHub 对项目进行版本管理。