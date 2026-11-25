export type User = {
  id: number;
  username: string;
  email: string;
  full_name: string;
  role: "user" | "admin";
  status: string;
  created_at: string;
};

export type Model = {
  key: string;
  display_name: string;
  family: string;
  target_column: string;
  description: string;
  is_proposed: boolean;
  is_available: boolean;
  unavailable_reason?: string | null;
};

export type Prediction = {
  record_index: number;
  predicted_label: string | number | boolean;
  confidence: number;
  class_probabilities: Record<string, number>;
  target_column: string;
  prediction_stage: string;
};

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1").replace(/\/$/, "");

function csrfToken() {
  const item = document.cookie.split("; ").find((part) => part.startsWith("csrf_token="));
  return item ? decodeURIComponent(item.slice("csrf_token=".length)) : "";
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const method = (init.method || "GET").toUpperCase();
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
    const csrf = csrfToken();
    if (csrf) headers.set("X-CSRF-Token", csrf);
  }
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    method,
    headers,
    credentials: "include",
  });
  const payload = response.status === 204 ? null : await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail;
    const message = typeof detail === "string" ? detail : detail?.message || JSON.stringify(detail || payload || "Request failed");
    throw new Error(message || `Request failed (${response.status})`);
  }
  return payload as T;
}

export const api = {
  me: () => request<User>("/auth/me"),
  login: (username_or_email: string, password: string) =>
    request<User>("/auth/login", { method: "POST", body: JSON.stringify({ username_or_email, password }) }),
  register: (data: { username: string; email: string; full_name: string; password: string }) =>
    request<{ message: string }>("/auth/register", { method: "POST", body: JSON.stringify(data) }),
  logout: () => request<{ message: string }>("/auth/logout", { method: "POST" }),
  models: () => request<Model[]>("/models"),
  predict: (modelKey: string, records: Record<string, unknown>[]) =>
    request<{ model_key: string; predictions: Prediction[] }>(`/models/${encodeURIComponent(modelKey)}/predict`, {
      method: "POST",
      body: JSON.stringify({ records }),
    }),
  pendingUsers: () => request<User[]>("/admin/users/pending"),
  approveUser: (id: number) => request<{ message: string }>(`/admin/users/${id}/approve`, { method: "POST" }),
  rejectUser: (id: number) => request<{ message: string }>(`/admin/users/${id}/reject`, { method: "POST" }),
};