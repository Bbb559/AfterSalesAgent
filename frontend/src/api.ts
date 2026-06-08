import type { ApiResponse, KbItem, KbStatus, MemorySession, RagResult, RunSummary, SourceSummary } from "./types";

const REQUEST_TIMEOUT_MS = 10_000;

async function requestJson<T>(url: string, init: RequestInit = {}, timeoutMs = REQUEST_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const timer = globalThis.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    const payload = (await response.json()) as ApiResponse<T>;
    if (!response.ok || !payload.success) {
      throw new Error(payload.error || `请求失败：${response.status}`);
    }
    return payload.data;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("请求超时，请检查 FastAPI 是否正常响应。");
    }
    throw error;
  } finally {
    globalThis.clearTimeout(timer);
  }
}

export function readableError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error || "未知错误");
}

export async function getHealth(): Promise<Record<string, unknown>> {
  const controller = new AbortController();
  const timer = globalThis.setTimeout(() => controller.abort(), 3000);
  try {
    const response = await fetch("/health", { signal: controller.signal });
    if (!response.ok) throw new Error(`FastAPI 健康检查失败：${response.status}`);
    return (await response.json()) as Record<string, unknown>;
  } finally {
    globalThis.clearTimeout(timer);
  }
}

export function listKnowledgeBases(): Promise<{ items: KbItem[] }> {
  return requestJson<{ items: KbItem[] }>("/api/kb/list", {}, 3000);
}

export function getKnowledgeBaseStatus(): Promise<KbStatus> {
  return requestJson<KbStatus>("/api/kb/status", {}, 3000);
}

export function createSession(): Promise<MemorySession> {
  return requestJson<MemorySession>("/api/memory/sessions", { method: "POST" }, 5000);
}

export function loadKnowledgeBase(databaseId: string): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>(
    "/api/kb/load",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ database_id: databaseId })
    },
    60_000
  );
}

export function deleteKnowledgeBase(databaseId: string): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>(`/api/kb/${encodeURIComponent(databaseId)}`, { method: "DELETE" }, 60_000);
}

export function buildKnowledgeBase(formData: FormData): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>("/api/kb/build", { method: "POST", body: formData }, 120_000);
}

export function searchRag(question: string, databaseId: string, retrievalMode: string, topK: number): Promise<RagResult> {
  return requestJson<RagResult>(
    "/api/rag/search",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        database_id: databaseId || null,
        retrieval_options: {
          retrieval_mode: retrievalMode,
          final_top_k: topK
        }
      })
    },
    30_000
  );
}

export function startAgent(
  userInput: string,
  databaseId: string,
  retrievalMode: string,
  topK: number,
  sessionId: string
): Promise<RunSummary> {
  return requestJson<RunSummary>(
    "/api/charger-diagnosis/start",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_input: userInput,
        database_id: databaseId || null,
        session_id: sessionId || null,
        retrieval_options: {
          retrieval_mode: retrievalMode,
          final_top_k: topK
        }
      })
    },
    10_000
  );
}

export function getRunSummary(runId: string): Promise<RunSummary> {
  return requestJson<RunSummary>(`/api/charger-diagnosis/runs/${encodeURIComponent(runId)}?view=summary`, {}, 10_000);
}

export function kbLabel(item: KbItem): string {
  return (
    item.label ||
    [
      item.display_name || item.metadata?.display_name || item.database_id || "未命名知识库",
      item.chunk_count ? `${item.chunk_count} chunks` : "",
      item.metadata?.parser_name || "",
      item.metadata?.item_identifier || ""
    ]
      .filter(Boolean)
      .join(" | ")
  );
}

export function sourceSummaries(data: RagResult): SourceSummary[] {
  const rawItems = Array.isArray(data.sources) && data.sources.length ? data.sources : data.results || [];
  return rawItems.slice(0, 3).map((item, index) => {
    const fileName = String(item.file_name || item.source || item.title || `来源 ${index + 1}`);
    const page = item.page === undefined || item.page === null || item.page === "" ? "未标注" : String(item.page);
    const score = typeof item.score === "number" ? item.score.toFixed(3) : String(item.score || "未记录");
    return { title: fileName, page, score };
  });
}
