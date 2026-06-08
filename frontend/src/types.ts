export type ApiResponse<T> = {
  success: boolean;
  data: T;
  error?: string;
};

export type KbItem = {
  database_id?: string;
  display_name?: string;
  label?: string;
  chunk_count?: number;
  metadata?: {
    display_name?: string;
    file_names?: string[];
    parser_name?: string;
    item_identifier?: string;
    updated_at?: string;
  };
};

export type KbStatus = {
  loaded?: boolean;
  current_database_id?: string;
  database_id?: string;
  chunk_count?: number;
  metadata?: KbItem["metadata"];
  sqlite?: Record<string, unknown>;
};

export type RunSummary = {
  run_id: string;
  session_id: string;
  status: "pending" | "running" | "completed" | "failed" | "timeout" | string;
  error?: string;
  node_statuses_compact?: Record<string, NodeEvent>;
  customer_reply?: string;
  debug_log_path?: string;
  updated_at?: number | string;
};

export type MemorySession = {
  session_id: string;
  message_count: number;
  created_at?: string;
  current_session_id?: string;
};

export type NodeEvent = {
  node?: string;
  title?: string;
  status?: string;
  duration?: number;
  timestamp?: number | string;
  input?: unknown;
  output?: Record<string, unknown>;
};

export type RagResult = {
  results?: Array<Record<string, unknown>>;
  sources?: Array<Record<string, unknown>>;
  result_count?: number;
  message?: string;
  trace?: Record<string, unknown>;
  error?: string;
};

export type SourceSummary = {
  title: string;
  page: string;
  score: string;
};
