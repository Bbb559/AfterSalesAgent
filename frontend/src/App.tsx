import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  buildKnowledgeBase,
  createSession,
  deleteKnowledgeBase,
  getHealth,
  getKnowledgeBaseStatus,
  getRunSummary,
  kbLabel,
  listKnowledgeBases,
  loadKnowledgeBase,
  readableError,
  searchRag,
  sourceSummaries,
  startAgent
} from "./api";
import type { KbItem, KbStatus, NodeEvent, RagResult, RunSummary } from "./types";

type TabId = "agent" | "kb" | "rag" | "status";

const tabs: Array<{ id: TabId; label: string }> = [
  { id: "agent", label: "充电桩安全诊断 Agent" },
  { id: "kb", label: "知识库管理" },
  { id: "rag", label: "知识检索功能" },
  { id: "status", label: "系统状态" }
];

const exampleQuestion = "VoltGate VG-11KW-Pro 无法启动充电，屏幕显示 C-RCD-04，漏保频繁跳闸，东莞。";
const backendSessionPattern = /^session_\d{8}_\d{6}_[a-f0-9]{8}$/i;

const nodeGroups: Array<{ id: string; title: string; nodes: string[] }> = [
  { id: "intent", title: "意图识别", nodes: ["input_guard", "triage", "case_extract"] },
  { id: "retrieval", title: "知识检索", nodes: ["memory_context", "case_memory_merge", "retrieval"] },
  { id: "diagnosis", title: "安全诊断", nodes: ["memory_answer", "safety_guard", "diagnosis"] },
  { id: "generation", title: "方案生成", nodes: ["memory_answer", "warranty_dispatch", "action"] },
  { id: "audit", title: "方案审核", nodes: ["memory_answer", "audit"] },
  { id: "final", title: "结束", nodes: ["final"] }
];

const statusText: Record<string, string> = {
  pending: "等待",
  running: "运行中",
  completed: "完成",
  failed: "失败",
  timeout: "超时",
  warning: "注意"
};

function App() {
  const [activeTab, setActiveTab] = useState<TabId>("agent");
  const [kbItems, setKbItems] = useState<KbItem[]>([]);
  const [kbStatus, setKbStatus] = useState<KbStatus | null>(null);
  const [selectedKb, setSelectedKb] = useState("");
  const [globalMessage, setGlobalMessage] = useState("React 工作台已就绪。FastAPI 未连接时页面仍可操作。");

  async function refreshKbData() {
    try {
      const [list, status] = await Promise.all([listKnowledgeBases(), getKnowledgeBaseStatus()]);
      setKbItems(list.items || []);
      setKbStatus(status);
      const loaded = status.current_database_id || status.database_id || "";
      setSelectedKb((current) => current || loaded || list.items?.[0]?.database_id || "");
      setGlobalMessage("知识库列表和系统状态已从 FastAPI 加载。");
    } catch (error) {
      setGlobalMessage(`FastAPI 未连接或响应失败：${readableError(error)}`);
    }
  }

  useEffect(() => {
    void refreshKbData();
  }, []);

  return (
    <main className="app-shell">
      <header className="app-header">
        <h1>新能源家用充电桩安全诊断工作台</h1>
        <p>React 薄壳只承载客服操作台；完整调试信息请查看后端 run log。</p>
      </header>

      <nav className="tabs" aria-label="工作台页面">
        {tabs.map((tab) => (
          <button key={tab.id} className={activeTab === tab.id ? "tab active" : "tab"} onClick={() => setActiveTab(tab.id)}>
            {tab.label}
          </button>
        ))}
      </nav>

      <div className="notice">{globalMessage}</div>

      {activeTab === "agent" && (
        <AgentPanel kbItems={kbItems} selectedKb={selectedKb} setSelectedKb={setSelectedKb} refreshKbData={refreshKbData} />
      )}
      {activeTab === "kb" && (
        <KnowledgeBasePanel
          kbItems={kbItems}
          kbStatus={kbStatus}
          selectedKb={selectedKb}
          setSelectedKb={setSelectedKb}
          refreshKbData={refreshKbData}
        />
      )}
      {activeTab === "rag" && <RagPanel kbItems={kbItems} selectedKb={selectedKb} setSelectedKb={setSelectedKb} />}
      {activeTab === "status" && <SystemStatusPanel kbStatus={kbStatus} refreshKbData={refreshKbData} />}
    </main>
  );
}

function AgentPanel({
  kbItems,
  selectedKb,
  setSelectedKb,
  refreshKbData
}: {
  kbItems: KbItem[];
  selectedKb: string;
  setSelectedKb: (value: string) => void;
  refreshKbData: () => Promise<void>;
}) {
  const [question, setQuestion] = useState(exampleQuestion);
  const [retrievalMode, setRetrievalMode] = useState("hybrid");
  const [topK, setTopK] = useState(5);
  const [sessionId, setSessionId] = useState(() => localStorage.getItem("charger_session_id") || "");
  const [reply, setReply] = useState("");
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [running, setRunning] = useState(false);
  const [creatingSession, setCreatingSession] = useState(false);
  const [error, setError] = useState("");
  const [sessionNotice, setSessionNotice] = useState("");
  const pollCancelled = useRef(false);
  const sessionRef = useRef(sessionId);

  useEffect(() => {
    if (sessionId) {
      localStorage.setItem("charger_session_id", sessionId);
    } else {
      localStorage.removeItem("charger_session_id");
    }
  }, [sessionId]);

  // 保持 ref 与 state 同步，避免异步回调中的过期闭包
  useEffect(() => {
    sessionRef.current = sessionId;
  }, [sessionId]);

  async function createNewSession(successMessage = "已新建会话。") {
    if (running) return;
    pollCancelled.current = true;
    setCreatingSession(true);
    setError("");
    setSessionNotice("");
    try {
      const session = await createSession();
      setSessionId(session.session_id);
      sessionRef.current = session.session_id;  // 立即同步 ref，避免异步竞态
      setReply("");
      setSummary(null);
      setSessionNotice(successMessage);
    } catch (err) {
      setSessionNotice("");
      setError(`新建会话失败：${readableError(err)}`);
    } finally {
      setCreatingSession(false);
    }
  }

  async function runAgent() {
    if (!question.trim()) {
      setError("请先输入客户问题。");
      return;
    }

    // 确保在调用后端前已持有有效后端 session，避免竞态导致后端每次生成新 session
    if (!backendSessionPattern.test(sessionRef.current)) {
      setCreatingSession(true);
      setError("");
      try {
        const session = await createSession();
        setSessionId(session.session_id);
        sessionRef.current = session.session_id;  // 立即同步 ref，避免异步竞态
      } catch (err) {
        setCreatingSession(false);
        setError(`创建会话失败：${readableError(err)}`);
        return;
      } finally {
        setCreatingSession(false);
      }
    }

    pollCancelled.current = false;
    setRunning(true);
    setError("");
    setReply("正在运行安全诊断 Agent...");
    try {
      const started = await startAgent(question, selectedKb, retrievalMode, topK, sessionRef.current);
      setSummary(started);
      await pollRun(started.run_id);
    } catch (err) {
      setError(`FastAPI 未连接或运行失败：${readableError(err)}`);
      setReply("FastAPI 未连接，当前无法运行 Agent。");
    } finally {
      setRunning(false);
    }
  }

  async function pollRun(runId: string) {
    for (let index = 0; index < 120; index += 1) {
      if (pollCancelled.current) return;
      const data = await getRunSummary(runId);
      setSummary(data);
      setSessionId(data.session_id || sessionRef.current);
      if (data.customer_reply) setReply(data.customer_reply);
      if (["completed", "failed", "timeout"].includes(data.status)) {
        if (data.error) setError(data.error);
        if (!data.customer_reply && data.status !== "completed") setReply(`运行${statusText[data.status] || data.status}：${data.error || "请查看日志。"}`);
        return;
      }
      await sleep(2000);
    }
    setError("前端轮询超时，请查看 run meta 和后端日志。");
  }

  return (
    <section className="agent-grid">
      <WorkflowStatus summary={summary} />
      <div className="panel">
        <div className="form-row align-end">
          <label className="field wide">
            <span>客户提问</span>
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={5} />
          </label>
        </div>
        <div className="form-row">
          <KnowledgeBaseSelect items={kbItems} value={selectedKb} onChange={setSelectedKb} />
          <label className="field compact">
            <span>检索方式</span>
            <select value={retrievalMode} onChange={(event) => setRetrievalMode(event.target.value)}>
              <option value="hybrid">hybrid</option>
              <option value="vector">vector</option>
              <option value="bm25">bm25</option>
            </select>
          </label>
          <label className="field compact">
            <span>TopK</span>
            <input type="number" min={1} max={10} value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
          </label>
        </div>
        <div className="form-row">
          <label className="field">
            <span>Session ID</span>
            <input value={sessionId || "正在创建后端会话..."} readOnly />
          </label>
          <button className="secondary" disabled={running || creatingSession} onClick={() => void createNewSession()} type="button">
            {creatingSession ? "创建中..." : "新建会话"}
          </button>
          <button className="secondary" onClick={() => void refreshKbData()} type="button">
            刷新知识库
          </button>
        </div>
        {sessionNotice && <div className="notice small">{sessionNotice}</div>}
        <button className="primary full" disabled={running || creatingSession} onClick={() => void runAgent()} type="button">
          {running ? "运行中..." : "运行安全诊断 Agent"}
        </button>
        {error && <div className="error">{error}</div>}
        <label className="field wide">
          <span>回复客户（可直接复制）</span>
          <textarea value={reply} readOnly rows={12} />
        </label>
        <RunMeta summary={summary} />
      </div>
    </section>
  );
}

function KnowledgeBasePanel({
  kbItems,
  kbStatus,
  selectedKb,
  setSelectedKb,
  refreshKbData
}: {
  kbItems: KbItem[];
  kbStatus: KbStatus | null;
  selectedKb: string;
  setSelectedKb: (value: string) => void;
  refreshKbData: () => Promise<void>;
}) {
  const [files, setFiles] = useState<FileList | null>(null);
  const [message, setMessage] = useState("");
  const [building, setBuilding] = useState(false);
  const [form, setForm] = useState({
    display_name: "充电桩售后安全知识库",
    doc_type: "售后运维与安全指南",
    product_line: "新能源家用充电设备",
    item_identifier: "VoltGate",
    version: "2025",
    parser_name: "pypdf",
    chunk_size: "700",
    chunk_overlap: "80"
  });

  async function onBuild(event: FormEvent) {
    event.preventDefault();
    if (!files?.length) {
      setMessage("请选择 PDF 文件。");
      return;
    }
    setBuilding(true);
    setMessage("正在构建知识库...");
    try {
      const data = new FormData();
      Array.from(files).forEach((file) => data.append("files", file));
      Object.entries(form).forEach(([key, value]) => data.append(key, value));
      data.append("splitter_name", "recursive");
      const result = await buildKnowledgeBase(data);
      setMessage(`知识库构建完成：${String(result.database_id || form.display_name)}`);
      await refreshKbData();
    } catch (error) {
      setMessage(`构建失败：${readableError(error)}`);
    } finally {
      setBuilding(false);
    }
  }

  async function onLoad() {
    if (!selectedKb) {
      setMessage("请先选择知识库。");
      return;
    }
    try {
      await loadKnowledgeBase(selectedKb);
      setMessage(`知识库已加载：${selectedKb}`);
      await refreshKbData();
    } catch (error) {
      setMessage(`加载失败：${readableError(error)}`);
    }
  }

  async function onDelete() {
    if (!selectedKb) {
      setMessage("请先选择知识库。");
      return;
    }
    try {
      await deleteKnowledgeBase(selectedKb);
      setSelectedKb("");
      setMessage(`知识库已删除：${selectedKb}`);
      await refreshKbData();
    } catch (error) {
      setMessage(`删除失败：${readableError(error)}`);
    }
  }

  return (
    <section className="two-column">
      <form className="panel" onSubmit={onBuild}>
        <label className="drop-zone">
          <span>上传 PDF</span>
          <input type="file" accept=".pdf" multiple onChange={(event) => setFiles(event.target.files)} />
        </label>
        <div className="form-row">
          <TextInput label="知识库名称" value={form.display_name} onChange={(value) => setForm({ ...form, display_name: value })} />
          <TextInput label="文档类型" value={form.doc_type} onChange={(value) => setForm({ ...form, doc_type: value })} />
        </div>
        <div className="form-row">
          <TextInput label="产品线" value={form.product_line} onChange={(value) => setForm({ ...form, product_line: value })} />
          <TextInput label="产品/服务标识" value={form.item_identifier} onChange={(value) => setForm({ ...form, item_identifier: value })} />
          <TextInput label="版本号" value={form.version} onChange={(value) => setForm({ ...form, version: value })} />
        </div>
        <div className="form-row">
          <label className="field compact">
            <span>PDF 解析器</span>
            <select value={form.parser_name} onChange={(event) => setForm({ ...form, parser_name: event.target.value })}>
              <option value="pypdf">pypdf</option>
              <option value="mineru">mineru</option>
            </select>
          </label>
          <TextInput label="chunk_size" value={form.chunk_size} onChange={(value) => setForm({ ...form, chunk_size: value })} />
          <TextInput label="chunk_overlap" value={form.chunk_overlap} onChange={(value) => setForm({ ...form, chunk_overlap: value })} />
        </div>
        <button className="primary full" disabled={building} type="submit">
          {building ? "构建中..." : "构建知识库"}
        </button>
        {message && <div className="notice small">{message}</div>}
      </form>
      <div className="panel">
        <KnowledgeBaseSelect items={kbItems} value={selectedKb} onChange={setSelectedKb} label="已有知识库" />
        <div className="button-row">
          <button className="secondary" onClick={() => void refreshKbData()} type="button">
            刷新知识库列表
          </button>
          <button className="secondary" onClick={() => void onLoad()} type="button">
            加载知识库
          </button>
          <button className="danger" onClick={() => void onDelete()} type="button">
            删除知识库
          </button>
        </div>
        <KbStatusCard status={kbStatus} />
      </div>
    </section>
  );
}

function RagPanel({
  kbItems,
  selectedKb,
  setSelectedKb
}: {
  kbItems: KbItem[];
  selectedKb: string;
  setSelectedKb: (value: string) => void;
}) {
  const [question, setQuestion] = useState("VG-11KW-Pro C-RCD-04 漏保频繁跳闸 售后处理");
  const [retrievalMode, setRetrievalMode] = useState("hybrid");
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState<RagResult | null>(null);
  const [message, setMessage] = useState("");

  async function onSearch() {
    if (!question.trim()) {
      setMessage("请输入检索关键词。");
      return;
    }
    setMessage("检索中...");
    try {
      const data = await searchRag(question, selectedKb, retrievalMode, topK);
      setResult(data);
      setMessage(data.message || "检索完成。");
    } catch (error) {
      setMessage(`检索失败：${readableError(error)}`);
    }
  }

  const sources = useMemo(() => (result ? sourceSummaries(result) : []), [result]);

  return (
    <section className="panel">
      <div className="form-row">
        <label className="field wide">
          <span>检索关键词</span>
          <input value={question} onChange={(event) => setQuestion(event.target.value)} />
        </label>
        <KnowledgeBaseSelect items={kbItems} value={selectedKb} onChange={setSelectedKb} />
      </div>
      <div className="form-row">
        <label className="field compact">
          <span>检索方式</span>
          <select value={retrievalMode} onChange={(event) => setRetrievalMode(event.target.value)}>
            <option value="hybrid">hybrid</option>
            <option value="vector">vector</option>
            <option value="bm25">bm25</option>
          </select>
        </label>
        <label className="field compact">
          <span>TopK</span>
          <input type="number" min={1} max={10} value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
        </label>
      </div>
      <button className="primary" onClick={() => void onSearch()} type="button">
        检索知识库
      </button>
      {message && <div className="notice small">{message}</div>}
      {result && (
        <div className="source-list">
          <div className="metric">命中数：{String(result.result_count ?? result.results?.length ?? sources.length)}</div>
          {sources.map((source) => (
            <div className="source-card" key={`${source.title}-${source.page}`}>
              <strong>{source.title}</strong>
              <span>页码：{source.page}</span>
              <span>分数：{source.score}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function SystemStatusPanel({ kbStatus, refreshKbData }: { kbStatus: KbStatus | null; refreshKbData: () => Promise<void> }) {
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [message, setMessage] = useState("");

  async function refresh() {
    try {
      const healthData = await getHealth();
      await refreshKbData();
      setHealth(healthData);
      setMessage("系统状态已刷新。");
    } catch (error) {
      setMessage(`刷新失败：${readableError(error)}`);
    }
  }

  return (
    <section className="panel">
      <button className="secondary" onClick={() => void refresh()} type="button">
        刷新系统状态
      </button>
      {message && <div className="notice small">{message}</div>}
      <div className="status-grid">
        <InfoCard title="FastAPI" value={health ? String(health.status || "ok") : "待刷新"} detail={String(health?.service || "")} />
        <InfoCard title="知识库" value={kbStatus?.loaded === false ? "未加载" : kbStatus ? "已连接" : "待刷新"} detail={currentKbText(kbStatus)} />
        <InfoCard title="调试策略" value="日志文件" detail="trace / tool_history / raw JSON 不进入前端高频渲染" />
      </div>
    </section>
  );
}

function WorkflowStatus({ summary }: { summary: RunSummary | null }) {
  const nodeStatuses = summary?.node_statuses_compact || {};
  return (
    <aside className="workflow">
      <div className="workflow-head">
        <h2>工作流执行状态</h2>
        <span>{statusText[summary?.status || "pending"] || summary?.status || "等待"}</span>
      </div>
      {nodeGroups.map((group) => {
        const event = aggregateGroup(group.nodes, nodeStatuses);
        return (
          <div className={`step ${event.status || "pending"}`} key={group.id}>
            <div className="step-head">
              <strong>{group.title}</strong>
              <span>{statusText[event.status || "pending"] || event.status || "等待"}</span>
            </div>
            <div className="step-line">输出：{event.output}</div>
            <div className="step-line">耗时：{event.duration}</div>
          </div>
        );
      })}
    </aside>
  );
}

function RunMeta({ summary }: { summary: RunSummary | null }) {
  if (!summary) {
    return <div className="run-meta">运行摘要：等待运行。</div>;
  }
  return (
    <div className="run-meta">
      <div>status：{summary.status}</div>
      <div>run_id：{summary.run_id || "未生成"}</div>
      <div>session_id：{summary.session_id || "未生成"}</div>
      <div>debug_log_path：{summary.debug_log_path || "等待日志"}</div>
      {summary.error && <div>error：{summary.error}</div>}
    </div>
  );
}

function KnowledgeBaseSelect({
  items,
  value,
  onChange,
  label = "知识库选择"
}: {
  items: KbItem[];
  value: string;
  onChange: (value: string) => void;
  label?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">未选择知识库</option>
        {items.map((item) => (
          <option key={item.database_id || kbLabel(item)} value={item.database_id || ""}>
            {kbLabel(item)}
          </option>
        ))}
      </select>
    </label>
  );
}

function KbStatusCard({ status }: { status: KbStatus | null }) {
  return (
    <div className="card">
      <h3>知识库状态</h3>
      <p>当前加载知识库：{currentKbText(status)}</p>
      <p>文本块：{String(status?.chunk_count || "未记录")}</p>
      <p>文件：{status?.metadata?.file_names?.slice(0, 3).join("、") || "未记录"}</p>
    </div>
  );
}

function TextInput({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function InfoCard({ title, value, detail }: { title: string; value: string; detail: string }) {
  return (
    <div className="card">
      <h3>{title}</h3>
      <strong>{value}</strong>
      <p>{detail}</p>
    </div>
  );
}

function currentKbText(status: KbStatus | null): string {
  if (!status) return "未知";
  return status.current_database_id || status.database_id || "未加载";
}

function aggregateGroup(nodes: string[], nodeStatuses: Record<string, NodeEvent>) {
  const events = nodes.map((node) => nodeStatuses[node]).filter(Boolean);
  const status = events.some((event) => event.status === "failed")
    ? "failed"
    : events.some((event) => event.status === "running")
      ? "running"
      : events.some((event) => event.status === "warning")
        ? "warning"
        : events.some((event) => event.status === "completed")
          ? "completed"
          : "pending";
  const lastOutput = [...events].reverse().find((event) => event.output)?.output;
  const duration = events.reduce((sum, event) => sum + (typeof event.duration === "number" ? event.duration : 0), 0);
  return {
    status,
    output: summarizeOutput(lastOutput),
    duration: duration ? `${duration.toFixed(2)}秒` : "未记录"
  };
}

function summarizeOutput(output: Record<string, unknown> | undefined): string {
  if (!output) return "等待输出";
  const preferred = ["summary", "message", "error", "risk_level", "priority", "result_count", "keys"];
  const parts = preferred
    .filter((key) => output[key] !== undefined && output[key] !== "")
    .map((key) => `${key}=${Array.isArray(output[key]) ? (output[key] as unknown[]).length : String(output[key])}`);
  return parts.length ? parts.join(", ") : "已更新";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export default App;
