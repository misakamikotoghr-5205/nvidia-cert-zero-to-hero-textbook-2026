# 《英伟达认证工程师从零精通标准课本（2026 全栈详尽版）》

> **2026 全栈官方认证考纲对齐版 · Associate (`NCA`) & Professional (`NCP`) 双轨实战教材**
>
> 从零基础 AI 数学与 GPU/CUDA 底层原理出发，贯通大模型底座与万卡分布式训练、企业级 RAG 与 Agentic AI、高性能推理优化与量化部署（TensorRT-LLM / Triton / NIM），直达智算集群高速网络、多租户隔离与一线排障实战。

---

## 1. 项目简介与教材特色

本书专为准备参加 **2026 年英伟达官方认证体系（NVIDIA Certification）** 的零基础初学者与一线工程师编写，坚持“**生活化直觉引入 $\to$ 极简具体数字手算 $\to$ 架构对比总表 $\to$ 完整可运行代码/CLI 实验室 $\to$ 初学者误区诊所 $\to$ 双轨考点映射**”的六步教学闭环：

- **全栈五卷十八章体系**：覆盖 AI 数学、CUDA/SIMT 与 Roofline 模型、Transformer 与 5D 并行、数据策展与对齐（SFT / LoRA / QLoRA / PPO / DPO / GRPO）、企业级 RAG（BM25 / RRF / `cuVS CAGRA`）、多智能体编排（`LangGraph` / `MCP` / Guided Decoding / `NeMo Guardrails` / `RAGAS` / `OpenTelemetry`）、推理引擎内核与量化（Prefill & Decode / `PagedAttention` / `Chunked Prefill` / 投机解码 / PD 分离 / `SmoothQuant` / `AWQ` / `GPTQ` / `FP8` / `2:4` 稀疏 / `TensorRT-LLM` / `Triton` / `NIM`）以及智算基础设施运维（`NVSwitch` / `InfiniBand` / `Spectrum-X RoCEv2` / `GPUDirect` / `MIG` + `MPS` / `GPU Operator` / `Slurm` / `DCGM` / `Xid` / `nccl-tests` / `nsys` & `ncu`）。
- **双轨考纲精准对齐**：每章均附 `NCA`（助理级）与 `NCP`（专业级）双轨考点映射，助力零基础读者走“先 `NCA` 后 `NCP`”两步走路线，或帮助有工程基础的读者 4 周直冲 `NCP`。
- **高密度实战演练与通关题库**：
  - 第一至第四卷每卷末均配 **8 道综合实战习题**（共 32 题），第五卷第 16、17 章各配 **4 道硬核运维排障练习**（共 8 题），全部附带逐步手算与逐项排除解析。
  - **第 18 章终极通关宝典**：包含 **50 道双档全真通关检测试卷**（`题目 1～15` 对应 Associate、`题目 16～50` 对应 Professional）、**零基础 60 天（8 周）每日学习课表 + 4 周冲刺表 + NVIDIA DLI 实验映射**、以及 **12 大核心精算公式总表** 与 **160 条核心英文缩写“一句话人话词典”**。
- **本地实操约定**：集群与推理实验可在本地或主机 `MisakaMikoto` 模拟环境中执行。

---

## 2. 仓库目录结构

```text
NVIDIA_ZeroToHero_Textbook_2026/
├── README.md                              # 本项目说明、五卷十八章总览与考纲构建指南
├── .gitignore                             # 忽略缓存、中间分片、PDF 产物与视觉质检截图
├── build_textbook_pdf.py                  # 全书 Markdown + 离线 KaTeX + Pygments + SVG -> A4 PDF 编译引擎
└── chapters/
    ├── vol1_foundations.md                # 第一卷：第 1～4 章 + 卷末综合习题 1～8
    ├── vol2_llm_and_training.md           # 第二卷：第 5～8 章 + 卷末综合习题 1～8
    ├── vol3_rag_and_agents.md             # 第三卷：第 9～12 章 + 卷末综合习题 1～8
    ├── vol4_inference_and_serving.md      # 第四卷：第 13～15 章 + 卷末综合习题 1～8
    └── vol5_infra_and_exams.md            # 第五卷：第 16～18 章（硬件网络、集群排障、50 题模考、60 天课表与词典）
```

> **编译产物输出位置**：执行全书构建脚本后，会在工程同级与待审目录输出合并版 Markdown 与印刷级 A4 PDF（已被 `.gitignore` 排除，不直接提交二进制大文件至 Git 仓库）：
> - `../英伟达认证工程师从零精通标准课本_2026全栈详尽版.md`
> - `../英伟达认证工程师从零精通标准课本_2026全栈详尽版.pdf`
> - `../../03_待审/英伟达认证工程师从零精通标准课本_2026全栈详尽版.pdf`

---

## 3. 五卷十八章完整内容大纲

### 第一卷：零基础破冰篇（[`chapters/vol1_foundations.md`](chapters/vol1_foundations.md)）
*主题：英伟达认证全景路线图、AI 数学直觉、GPU/CUDA 底层原理与 PyTorch 从零手推*

| 章回 | 章节标题 | 核心内容与实验室 |
| :--- | :--- | :--- |
| **第 1 章** | 英伟达认证体系与 AI 全栈生态全景图 | 英伟达软硬件四层生态架构、70B 算力显存账、2026 全科目选档矩阵、`Certiverse` 线上监考十大避坑指南、GPU/CUDA/PyTorch 巡检 CLI 实验室 |
| **第 2 章** | 零基础 AI 数学与计算机浮点数底层原理 | `[B, S, H]` 张量维度、4 词词表手算 `Softmax`/`Cross-Entropy`/`Perplexity`、链式法则与 `SGD -> AdamW`、二进制浮点逐位拆解（`FP32` 到 `NVFP4`）、NumPy 反向传播实验室 |
| **第 3 章** | 从零看懂 GPU、CUDA 并行计算与 Roofline 性能模型 | CPU vs GPU 与 SIMT 并发、Warp 分支发散与合并访存、显存金字塔与带宽数量级、Roofline 屋脊线与算术强度拐点手推、PyTorch/CUDA Roofline Benchmark |
| **第 4 章** | 神经网络积木与 PyTorch 从零实战 | 多层线性坍缩证明与 `SwiGLU` 手算、残差连接与 `LayerNorm` vs `RMSNorm`、PyTorch 核心五件套、手写现代微型自回归语言模型实战项目 |
| **卷末演练** | 第一卷综合实战习题（共 8 题） | 认证权重精算、14B 显存账、PPL 手算、Loss Scaling 与 AdamW、Roofline 拐点反转、PyTorch 排障 |

### 第二卷：大模型底座与分布式训练篇（[`chapters/vol2_llm_and_training.md`](chapters/vol2_llm_and_training.md)）
*主题：分词器、Transformer 手写推导、万卡并行与 NeMo 数据对齐*

| 章回 | 章节标题 | 核心内容与实验室 |
| :--- | :--- | :--- |
| **第 5 章** | 从字符到 Token——分词器与词嵌入底层原理 | `BPE` / `BBPE` 合并推演、`Weight Tying` 权重共享精算、手写 BPE 训练器与 One-Hot 查表等价验证 |
| **第 6 章** | Transformer 与现代 LLM 架构从零手推与手写 | 缩放点积注意力与因果掩码手算、`KV Cache` 与 `MHA`/`MQA`/`GQA`/`MLA`、`RoPE` 与 `PI`/`NTK`/`YaRN` 外推、`FlashAttention` 与 `MoE` 路由、Llama 风格 Decoder 与 KV Cache 等价实验室 |
| **第 7 章** | 分布式并行训练从零图解与显存精算 | $16\Psi$ 静态显存账与激活重算、5D 并行（`DDP` / `ZeRO` / `FSDP2` / `TP` / `PP` 1F1B 气泡 / `CP` / `EP`）、`PyTorch FSDP2` 与 `NeMo 2.0` 3D 并行配置 |
| **第 8 章** | 数据工程（NeMo Curator）、微调（SFT/LoRA/QLoRA）与偏好对齐（PPO/DPO/GRPO） | MinHash + LSH 模糊去重、`SFT` `-100` 掩码、`LoRA`/`QLoRA` (`NF4`) 参数与显存账、`PPO` $\to$ `DPO` $\to$ `GRPO` 组相对优势对齐实验 |
| **卷末演练** | 第二卷综合课后习题（共 8 题） | BPE 合并与中文压缩比、注意力小矩阵手算、32B KV Cache 显存墙、$16\Psi$ 与 64 卡 3D/5D 并行气泡、Jaccard 去重与 GRPO |

### 第三卷：RAG 检索增强与 Agentic AI 智能体系统工程篇（[`chapters/vol3_rag_and_agents.md`](chapters/vol3_rag_and_agents.md)）
*主题：企业知识检索、多智能体编排、安全护栏与评估*

| 章回 | 章节标题 | 核心内容与实验室 |
| :--- | :--- | :--- |
| **第 9 章** | 从零吃透企业级 RAG 与 GPU 加速向量数据库 | 多模态 PDF 摄取与五大 Chunking 策略、三代检索漏斗（Dense + Okapi `BM25` + `RRF` + Cross-Encoder Rerank）、`cuVS / CAGRA` GPU 向量索引、`CRAG`/`Self-RAG`、端到端 Hybrid Search 引擎 |
| **第 10 章** | Agentic AI 认知架构与多智能体编排 | Agent 选型红线、`ReAct` / `Plan-and-Execute` / `Reflexion` / `ToT` Token 成本手算、Supervisor/Hierarchical/Swarm 拓扑、生产级 `LangGraph`（`TypedDict` + Reducer + Checkpointer + `HITL` 断点恢复） |
| **第 11 章** | 工具调用（Function Calling）、MCP 协议、结构化解码与四层记忆系统 | Tool Calling 幂等/自愈/熔断、`MCP` 三大原语（`Tools`/`Resources`/`Prompts`）控制权边界、Guided Decoding `FSM` 掩码手算、四层记忆与 `Pre-filtering RBAC`、FSM 掩码与 Tool 运行时实验室 |
| **第 12 章** | AI 安全护栏（NeMo Guardrails）、全链路评估与可观测性 | `OWASP Top 10 for LLM` 与间接注入防御、`NeMo Guardrails` 与 `Colang 2.0` 协程护栏、`RAGAS` 手算归因、`pass@k` vs `pass^k`、LLM-as-a-Judge 去偏与 `OpenTelemetry` `Trace -> Span` 综合评测台 |
| **卷末演练** | 第三卷综合实战演练（共 8 题） | BM25+RRF 手算、多模态 PDF+CAGRA、ReAct 成本、LangGraph 归约器排障、MCP+RBAC、FSM Softmax、护栏拦截、RAGAS+`pass^k` |

### 第四卷：高性能推理优化、模型量化与生产级云原生部署篇（[`chapters/vol4_inference_and_serving.md`](chapters/vol4_inference_and_serving.md)）
*主题：引擎内核、量化手推、TensorRT-LLM、Triton 与 NIM*

| 章回 | 章节标题 | 核心内容与实验室 |
| :--- | :--- | :--- |
| **第 13 章** | 大模型推理引擎内核解剖——Prefill/Decode、PagedAttention 与批处理调度 | Prefill vs Decode 算术强度与 Roofline、`TTFT`/`TPOT`/`ITL` 与 `Goodput`、`PagedAttention` 与 `CoW` Prefix Caching、三代调度（`Static -> IFB -> Chunked Prefill`）、投机解码、PD 分离与 `NVIDIA Dynamo & NIXL` 仿真实验室 |
| **第 14 章** | 模型量化、剪枝与知识蒸馏从零手算 | 对称/非对称 `INT8`/`INT4` 量化误差与分组（Group=128）开销、`SmoothQuant` 数学等价平移、`AWQ` vs `GPTQ`、`FP8` (`E4M3`/`E5M2`) KV Cache、`2:4` 结构化稀疏与 `Minitron` 剪枝蒸馏、`ModelOpt` 实战实验室 |
| **第 15 章** | NVIDIA 生产推理全栈实战——TensorRT-LLM、Triton 与 NIM 云原生部署 | `ModelOpt -> trtllm-build` 编译流水线、`Triton Inference Server` (`config.pbtxt` / `decoupled: True`) 与 `GenAI-Perf` 压测、`NVIDIA NIM`（Air-Gapped 离线部署、`Multi-LoRA`、K8s 探针与 KV Cache HPA） |
| **卷末演练** | 第四卷综合课后实战习题（共 8 题） | Decode Roofline、Goodput 选型、PagedAttention 块表、Chunked Prefill 与投机解码负优化、INT8/4 与 SmoothQuant 手算、Triton/NIM 排障与 HPA 架构 |

### 第五卷：NVIDIA 智算集群硬件网络、云原生运维排障与 50 题双档全真通关题库篇（[`chapters/vol5_infra_and_exams.md`](chapters/vol5_infra_and_exams.md)）
*主题：数据中心高速互联、多租户调度排障、50 题双档全真模考、60 天课表与术语公式宝典*

| 章回 | 章节标题 | 核心内容与模块 |
| :--- | :--- | :--- |
| **第 16 章** | NVIDIA 数据中心硬件与高速网络从零图解 | A100/H100/H200/B200 规格对比、`NVSwitch`、`InfiniBand (NDR/XDR/SHARPv3)` vs `Spectrum-X RoCEv2 (ECN/PFC)`、8 轨拓扑、`GPUDirect (P2P/RDMA/GDS)`、拓扑自检实验室 + 4 道运维练习 |
| **第 17 章** | 多租户隔离、集群调度与救火排障实战 | `MIG (GI/CI)` 7 切片约束 + `MPS` + `Time-Slicing` + `vGPU`、`NVIDIA GPU Operator` 8 组件、`Slurm + Pyxis/Enroot`、`DCGM` 压测与 Row Remapper、十大 `Xid`（含 `Xid 94` vs `95`）、`nccl-tests` (`algbw`/`busbw`)、`nsys`/`ncu` 与 `dcgmi profile --pause` + 4 道排障练习 |
| **第 18 章** | 双档 50 题全真通关检测试卷 + 零基础 60 天学习课表 + 全书术语公式宝典 | **Part A**：50 题双档模拟试卷（`题目 1～15` 为 Associate，`题目 16～50` 为 Professional，均含解析与排除）<br>**Part B**：零基础 60 天（8 周）每日学习课表 + 有基础 4 周直冲 `NCP` 表 + NVIDIA DLI 官方课程映射<br>**Part C**：12 大考前必背核心精算公式总表 + 160 条全栈核心英文缩写“一句话人话词典” |

---

## 4. 2026 NCA / NCP 认证考纲映射总表

英伟达认证体系分为入门助理级 **NCA**（`NVIDIA-Certified Associate`，证书有效期 2 年）与专业实战级 **NCP**（`NVIDIA-Certified Professional`，证书有效期 2 年），全部主流科目主要通过 **Certiverse** 线上远程监考交付：

| 认证代码 | 级别与中文全称 | 题型与时长（参考费用） | 官方 Exam Blueprint 核心考域与权重分布 | 对应本书核心卷章 |
| :--- | :--- | :--- | :--- | :--- |
| `NCA-GENL` | 初级：生成式 AI 与大语言模型 | 50～60 道单选 / 60 分钟（约 125 美元） | 核心机器学习与 AI 知识 30%；软件开发 24%；实验调优 22%；数据分析 14%；可信 AI 10% | 第一卷（第 1～4 章）+ 第二卷（第 5～6 章基础）+ 第三卷（第 9 章入门）+ 第 18 章 `题目 1～15` |
| `NCA-AIIO` | 初级：AI 基础设施与运维 | 50 道单选 / 60 分钟（约 125 美元） | AI 基础设施 40%；AI 核心基础知识 38%；AI 运维操作 22% | 第一卷（第 1、3 章）+ 第五卷（第 16～17 章基础）+ 第 18 章 `题目 1～15` |
| `NCP-GENL` | 专业：生成式 AI 与大语言模型 | 60～70 道单选 / 120 分钟（约 200 美元起） | 模型优化 17%；GPU 加速与并行 14%；提示词工程 13%；微调（PEFT/LoRA/QLoRA）13%；数据准备 9%；模型部署 9%；评测 7%；生产监控与可靠性 7%；LLM 架构 6%；安全伦理合规 5% | 第一卷（第 2～4 章）+ 第二卷（第 5～8 章全章）+ 第四卷（第 13～15 章）+ 第 18 章 `题目 16～50` |
| `NCP-AAI` | 专业：Agentic AI 智能体系统 | 60～70 道单选 / 120 分钟（约 200 美元起） | 智能体架构设计 15%；智能体开发 15%；评估与调优 13%；部署与扩缩容 13%；认知规划与记忆 10%；知识集成与 RAG 10%；NVIDIA 平台栈（NIM/NeMo）7%；运维监控 7%；安全护栏合规 5%；人机协同监督 5% | 第三卷（第 9～12 章全套核心）+ 第四卷（第 15 章 NIM）+ 第 18 章 `题目 16～50` |
| `NCP-AII` | 专业：AI 基础设施工程 | 70～75 道单选 / 120 分钟（约 400 美元） | 集群测试与验证（NCCL/HPL 压测）33%；系统与服务器上线调测（固件/驱动）31%；控制平面安装配置（BCM/Kubernetes）19%；故障排查与优化 12%；物理层布线与光模块管理 5% | 第一卷（第 1、3 章）+ 第二卷第 7 章集合通信 + 第五卷（第 16～17 章全套）+ 第 18 章 `题目 16～50` |
| `NCP-ADS` | 专业：GPU 加速数据科学 | 60～70 道单选 / 120 分钟（约 200 美元起） | 数据处理与软件工程素养 19%；MLOps 19%；数据准备 17%；GPU 与云计算 16%；机器学习算法 15%；数据分析 14% | 第一卷（第 2、3 章）+ 第二卷第 8 章数据策展（NeMo Curator） |
| `NCP-AIO`（社区简写 `NCP-OAI` / 实操版 `NCP-AIOL`） | 专业：AI 运维与集群操作 | 约 35 道单选 + 3 道在线实操实验 / 120 分钟（约 500 美元） | 安装与部署（GPU Operator/NGC）31%；集群管理（MIG/BCM）23%；工作负载调度（Slurm/Kubernetes）23%；排障与性能调优 23% | 第一卷（第 1、3 章）+ 第四卷第 15 章云原生 + 第五卷第 16～17 章 |
| `NCP-AIN` | 专业：AI 高性能网络 | 70～75 题（含场景诊断与实验）/ 120 分钟（约 400 美元） | NVIDIA Spectrum 以太网（RoCEv2）30%；InfiniBand 网络 30%；排障工具与遥测（WJH/UFM）20%；自动化配置（Ansible/NVUE）10%；AI 数据中心拓扑设计 5%；Kubernetes RDMA 集成 5% | 第一卷第 1 章网络底座 + 第二卷第 7 章集合通信 + 第五卷第 16～17 章高速网络与 `nccl-tests` |

> **选考号防坑提示**：AI 集群运维专业认证在官方报考入口中的正式代码为 **`NCP-AIO`**（带交互式终端实操实验的版本为 **`NCP-AIOL`**，部分社区简写为 `NCP-OAI`），请勿与图形仿真开发认证 **`NCP-OUSD`**（OpenUSD Development）混淆。

---

## 5. PDF 构建与本地编译指南

### 5.1 运行环境依赖

[`build_textbook_pdf.py`](build_textbook_pdf.py) 采用“**保护管线 + 离线 KaTeX 预渲染 + Pygments 语法高亮 + Headless Chrome A4 排版 + Ghostscript 压缩保底**”架构：

- **Python 3** 及核心库：`markdown-it-py`、`pygments`
- **Node.js / npm** 与 **KaTeX**（优先读取离线 `/tmp/katex_bundle/node_modules/katex` 或本地 `node_modules/katex`，用于零网络依赖批量静态渲染全部 LaTeX 行内/块级公式；缺失时自动回退至内置纯 Python LaTeX 兜底渲染器）
- **Google Chrome**（`/usr/local/bin/google-chrome`，无头模式打印 A4 PDF）
- **Ghostscript (`gs`)** 与 **Poppler (`pdfinfo` / `pdftotext` / `pdftoppm`)**（用于页数检测与体积控制，确保最终 PDF 严格低于 `24,000,000` bytes 且保留矢量 CJK 与 KaTeX 字体）

### 5.2 快速验证与全书构建命令

```bash
cd /home/magnus51235608/GoogleDrive/05_制作/NVIDIA_ZeroToHero_Textbook_2026

# 1. 运行 Phase 1 样张单元测试（验证中文、KaTeX 公式、货币符号保护与长代码折行）
python3 build_textbook_pdf.py --test-sample

# 2. 编译全书五卷十八章（生成合并 Markdown 与印刷级 A4 PDF，并同步至 03_待审）
python3 build_textbook_pdf.py

# 3. 可选：编译单个分卷并保留中间 HTML 排查排版
python3 build_textbook_pdf.py --input chapters/vol1_foundations.md --output-pdf /tmp/vol1.pdf --dump-html /tmp/vol1.html
```

### 5.3 编译质量门禁

- **公式与代码隔离**：Fenced Code $\to$ Inline Code $\to$ Raw SVG $\to$ Display Math $\to$ Inline Math 严格占位保护，防止 `$d_k$` 下划线被误解析为斜体 `<em>`，并防止 `$100` 等货币文本被误判为公式。
- **体积与分页门禁**：全书合订 PDF 体积严格控制 `< 24,000,000` bytes（若初始渲染超限则自动调用 Ghostscript `/prepress` $\to$ `/ebook` $\to$ `/screen` 分级压缩并启用字体子集化），总页数 $\ge 200$ 页。
- **隐私与安全红线**：全书与示例配置严禁出现任何非 RFC5737 / 非回环明文服务器 IPv4 地址（仅使用 RFC 测试网段或 `MisakaMikoto` 等逻辑主机名）。
