import { describe, expect, it, vi } from "vitest";
import { createSession, getRunSummary, readableError, sourceSummaries, startAgent } from "./api";

describe("api client", () => {
  it("polls the lightweight run summary endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        success: true,
        data: {
          run_id: "run_1",
          session_id: "session_1",
          status: "completed",
          customer_reply: "已完成"
        }
      })
    });
    vi.stubGlobal("fetch", fetchMock);

    const data = await getRunSummary("run_1");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/charger-diagnosis/runs/run_1?view=summary",
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    );
    expect(data.status).toBe("completed");
    expect(JSON.stringify(data)).not.toContain("tool_history");
    expect(JSON.stringify(data)).not.toContain("trace");
  });

  it("creates sessions through the backend instead of local ids", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          data: { session_id: "session_20260607_010101_aabbccdd", message_count: 0 }
        })
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          data: { session_id: "session_20260607_010102_eeff0011", message_count: 0 }
        })
      });
    vi.stubGlobal("fetch", fetchMock);

    const first = await createSession();
    const second = await createSession();

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/memory/sessions",
      expect.objectContaining({ method: "POST", signal: expect.any(AbortSignal) })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/memory/sessions",
      expect.objectContaining({ method: "POST", signal: expect.any(AbortSignal) })
    );
    expect(first.session_id).not.toBe(second.session_id);
  });

  it("sends the current backend session id when starting an agent run", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        success: true,
        data: {
          run_id: "run_1",
          session_id: "session_backend",
          status: "running"
        }
      })
    });
    vi.stubGlobal("fetch", fetchMock);

    await startAgent("充电桩异常", "kb_1", "hybrid", 5, "session_backend");

    const body = JSON.parse(String(fetchMock.mock.calls[0][1]?.body));
    expect(fetchMock.mock.calls[0][0]).toBe("/api/charger-diagnosis/start");
    expect(body.session_id).toBe("session_backend");
  });

  it("limits RAG display sources and excludes raw chunk text", () => {
    const summaries = sourceSummaries({
      results: [
        { file_name: "a.pdf", page: 1, score: 0.91, text: "raw chunk 1" },
        { file_name: "b.pdf", page: 2, score: 0.82, text: "raw chunk 2" },
        { file_name: "c.pdf", page: 3, score: 0.73, text: "raw chunk 3" },
        { file_name: "d.pdf", page: 4, score: 0.64, text: "raw chunk 4" }
      ]
    });

    expect(summaries).toHaveLength(3);
    expect(JSON.stringify(summaries)).not.toContain("raw chunk");
  });

  it("normalizes unknown errors for UI messages", () => {
    expect(readableError(new Error("FastAPI 未连接"))).toBe("FastAPI 未连接");
    expect(readableError("boom")).toBe("boom");
  });
});
