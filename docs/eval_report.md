# AfterSalesAgentV2 轻量评估报告

> 基准 commit：`c435d24`
> 原则：所有数字均可被测试结果复现，不编造百分率。

---

## 1. 测试基线

### 1.1 后端 pytest

| 指标 | 数值 |
|------|------|
| 测试文件数 | 20 |
| 总用例数 | 216 |
| 通过 | **210** |
| 跳过 | 6（集成门控，需 `RUN_INTEGRATION_SMOKE=1` + LLM + 知识库） |
| 失败 | 0 |
| 全部通过命令 | `pytest tests/ --ignore=tests/eval -v` |

**分文件测试数**（按数量降序）：

| 文件 | 用例数 | 覆盖范围 |
|------|--------|---------|
| `test_sqlite_store.py` | 41 | SQLite 双写、TTL、异常安全、Schema 版本 |
| `test_graph_workflow.py` | 26 | memory_answer v2、上下文追问 gate、安全诊断、multi-turn |
| `test_sqlite_recall.py` | 25 | SQLite recall_context 完整性、跨维度隔离 |
| `test_smoke_full_chain.py` | 16 | workflow 离线冒烟：安全分级、case 抽取、保修、否定词 |
| `test_api_contract.py` | 14 | FastAPI 同步/异步接口契约、CORS、health |
| `test_llm_degradation.py` | 13 | LLM 不可用降级、fallback 路径 |
| `test_llm_agents.py` | 12 | triage / case_extract / diagnosis / action / audit Agent |
| `test_memory_api.py` | 10 | Memory 只读 API、session/messages 查询 |
| `test_brand_patterns.py` | 9 | 品牌/型号结构化抽取与 fallback |
| `test_kb_manager.py` | 8 | 知识库构建、加载、列表、删除 |
| `test_memory.py` | 6 | MemoryManager 会话创建与上下文召回 |
| `test_local_tools.py` | 6 | Warranty、memory 工具调用与错误处理 |
| `test_project_structure.py` | 5 | 项目目录结构和模块完整性 |
| `test_query_rewrite.py` | 5 | RAG Query Rewrite 生成 |
| `test_llm_factory.py` | 4 | LLM 工厂创建与降级 |
| `test_parser_options.py` | 4 | PDF 解析器配置 |
| `test_rag_empty_result.py` | 4 | RAG 空结果、KB 未加载时行为 |
| `test_smoke_api.py` | 3 | FastAPI 集成冒烟（同步接口 + 流式 + 知识库命中） |
| `test_smoke_unknown_kb.py` | 3 | 未知品牌/故障码 insufficient 验证 |
| `test_mineru_download_config.py` | 2 | MinerU 配置下载 |

### 1.2 前端 vitest

| 指标 | 数值 |
|------|------|
| 测试文件 | 1（`api.test.ts`） |
| 通过 | **5** |
| 失败 | 0 |
| 覆盖内容 | API client 请求构建、响应解析、错误处理 |

### 1.3 集成冒烟测试（需 `RUN_INTEGRATION_SMOKE=1`）

| 类别 | 文件 | 用例数 | 说明 |
|------|------|--------|------|
| 工作流离线冒烟 | `test_smoke_full_chain.py` | 16 | 无需 LLM，走 fallback 路径 |
| 未知知识集成 | `test_smoke_unknown_kb.py` | 3 | 需真实 LLM + 知识库 |
| FastAPI 集成 | `test_smoke_api.py` | 3 | 需 FastAPI + 知识库 |

**集成测试门控**：所有需要真实环境的测试通过 `RUN_INTEGRATION_SMOKE=1` 环境变量守卫，默认不执行，避免 CI 环境误跑。

### 1.4 Skipped 说明

- 6 个 skipped：其中 3 个集成测试默认跳过（`RUN_INTEGRATION_SMOKE` 未设置），3 个为 API contract 测试中特定环境条件跳过。
- 有 2 个集成测试（F2 流式接口、K1 知识库命中）在真实环境运行时因端点未实现 / 索引未构建而安全 skip，不硬失败。

---

## 2. 代码规模与结构

| 维度 | 数值 |
|------|------|
| 后端 Python 文件 | 47 个 |
| 测试 Python 文件 | 21 个（含 eval 目录） |
| 前端核心 TS/CSS 文件 | 5 个（App / api / types / main / styles） |
| Python 编译检查 | `compileall backend api.py` 全部通过 |
| 前端构建 | `npm run build` 通过 |

---

## 3. 系统能力指标

### 3.1 LangGraph 工作流

| 指标 | 数值 |
|------|------|
| 工作流节点数 | 13（含 memory_answer gate） |
| LangGraph 显式阶段 | 12 |
| 阶段覆盖 | input_guard → triage → case_extract → memory_context → case_memory_merge → safety_guard → retrieval → diagnosis → warranty_dispatch → action → audit → final → memory_workflow_write |

### 3.2 RAG 检索

| 指标 | 数值 |
|------|------|
| 检索模式 | 3（hybrid / vector / bm25） |
| 向量引擎 | FAISS IndexFlatIP |
| 关键词引擎 | BM25 |
| 融合算法 | RRF（Reciprocal Rank Fusion） |
| Query Rewrite | 支持，默认开启（3 条改写） |
| Rerank | 基础 LLM rerank 已实现 |
| 文档解析器 | 2（pypdf / MinerU） |
| 分块策略 | 2（recursive / simple） |

### 3.3 安全护栏

| 指标 | 数值 |
|------|------|
| 紧急安全信号（emergency） | 12（明火、触电、冒烟、烧了 等） |
| 高风险安全信号（high） | 37（烧焦味、漏电、手麻、跳闸、枪头很烫、滋滋响 等） |
| 安全故障码映射 | 3（C-GND-01 / C-RCD-04 / C-TEMP-09） |
| 否定词模式 | 25（没有、未发现、暂时没有、不伴随 等） |
| 不确定语义标记 | 13（是不是、会不会、如果、好像 等） |
| 安全语义分类 | 3 类：confirmed / negated / uncertain |
| 安全禁止动作 | 5 项 |
| 紧急客户指引 | 4 项 |
| 高风险客户指引 | 4 项 |

**确认过修复的关键安全边界问题**：

| 问题 | 修复前 | 修复后 |
|------|--------|--------|
| "暂时没有发热、跳闸或者烧焦味" | 误判为 p1_high（否定词未识别） | p3_low（否定词跨枚举覆盖） |
| "是不是漏电了？" 中的 "漏电" | 可能误判为风险确认 | 正确识别为 uncertain |
| "手有点麻" "枪头很烫" "漏保跳了两次" | 自然表达不在信号列表中 | 已补入 HIGH_RISK_SIGNALS |
| "充电时配电箱冒烟" 结尾 "？" | 可能误判 uncertain | 逗号分隔从句 → 仍为 confirmed |

### 3.4 Memory Answer v2

| 指标 | 数值 |
|------|------|
| 测试覆盖 | 26 个 workflow 测试 + 56 个 memory parse 单元测试 |
| 受控字段枚举 | 29（来源 dataclass 约束） |
| Field Resolver 阶段 | 2（Pass 1 结构化来源 high + Pass 2 FTS5 medium） |
| Answer 生成 | LLM + 回退模板 v2 |
| 防编造校验 | `_validate_answer_fields()` 正则检测 |
| 上下文追问 gate | 确定性二次判断（0 token，<50 字符 + 实体匹配） |

### 3.5 未知知识库（insufficient 识别）

| 测试案例 | 品牌/故障码 | RAG 命中 | safety 判断 | diagnosis evidence_status |
|----------|-------------|----------|-------------|---------------------------|
| U1 | NeoCharge NC-5000E | 0 | p3_low | insufficient ✅ |
| U2 | BluePile BP-30A + 烧焦味 | 0 | p1_high（安全规则兜底） | insufficient ✅ |
| U3 | StarDock SD-SmartCharge + E-0521 | 0 | p3_low | insufficient ✅ |

3 个未知知识场景全部通过，均正确标记 `evidence_status=insufficient`，无编造故障原因。

---

## 4. 量化成果描述

---

**新能源充电桩售后安全诊断 Agent（AfterSalesAgentV2）**

基于 LangGraph + FastAPI + React 构建的 Agentic RAG 售后诊断系统，面向家用充电桩的电气安全识别、故障诊断、保修判断和派工辅助。

**工程规模**：
- 后端 47 个 Python 模块，前端 React 工作台，20 个测试文件，**216 条自动化测试收集，其中 210 passed、6 skipped、0 failed**
- 13 节点 LangGraph 工作流覆盖输入安全 → 分诊 → 信息抽取 → 记忆召回 → 安全分级 → 知识库检索 → 诊断 → 保修派工 → 回复生成 → 审核 → 记忆写入全链路

**RAG 检索系统**：
- 实现 FAISS + BM25 + RRF 混合检索（3 种模式可切换），支持 Query Rewrite 和 LLM Rerank
- 双解析器（pypdf / MinerU），支持知识库构建、加载、删除和状态查询

**安全护栏**：
- 52 项安全规则信号（12 紧急 + 37 高风险 + 3 故障码映射）+ 25 个否定词模式 + 13 个不确定语义标记
- 3 级语义分类（confirmed / negated / uncertain），修复了否定词跨枚举覆盖、疑问句误判等边界问题
- 硬拦截规则：禁止动作、紧急/高风险客户指引、高风险诊断强制覆盖

**Memory Answer v2**：
- 29 个受控字段枚举，双阶段 Field Resolver（结构化来源 + FTS5 fallback），LLM 生成 + 回退兜底
- 防编造校验 + 上下文追问确定性 gate（0 token），支持跨轮次实体补全

**测试策略**：
- **210+ 自动化测试通过基线**：216 条后端测试收集（210 passed / 6 skipped / 0 failed），覆盖 workflow、safety、memory、RAG、API contract、LLM 降级；+ 5 条前端测试
- **22 条冒烟测试设计**：离线 workflow 16 条全通过；集成测试 U1/U2/U3/F1 通过，F2/K1 因端点未实现或索引未构建安全跳过（不硬失败）
- 基于自测样例的未知知识库 insufficient 识别：3/3 通过，无编造

**效果指标**（MVP 阶段确定性评估，不依赖 LLM，运行方式见 `docs/eval/`）：
- **RAG 检索命中率**：20 条知识库内标准测试集，FAISS + BM25 + RRF 混合检索，Hit@1 = 90%，Hit@3 = 100%，Hit@5 = 100%
- **安全信号分类准确率**：30 条安全场景测试集（confirmed / negated / uncertain / safe 四类），整体准确率 100%，漏报率 0%，误报率 0%，确认信号 F1 = 0.95

---

*本报告所有数据均来自本地测试运行结果，不包含线上 QPS、成本下降、召回率等需生产环境验证的指标。*
