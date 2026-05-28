"use client";

import { useState } from "react";
import { authFetch } from "@/lib/auth-fetch";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { RotateTokenDialog } from "./rotate-token-dialog";

// Mirrors the shape returned by `GET /api/tokens` (`app/tokens/routes_api.py::_enrich`).
// Optional fields are nullable in the backend payload, so we surface that
// here rather than papering over with `?: string`. `created_at` is added by
// the backend patch landing alongside this component — without it the
// "Created" column would have nothing to show.
//
// S5 (#104) additions:
//   - rate_limit_tps: NULL = unlimited; positive integer = sliding-10s budget
//   - priority: 0..9 STRICT; 9 always served first, 0 may starve indefinitely
//   - usage_24h: rollup totals from token_usage_minute over the trailing 24h
export interface TokenUsage24h {
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface TokenItem {
  id: string;
  name: string;
  prefix: string;
  preview: string;
  created_at: string;
  last_used_at: string | null;
  expires_at: string | null;
  rotated_at: string | null;
  rotated_from: string | null;
  successor_id: string | null;
  successor_deleted: boolean;
  is_expired: boolean;
  is_near_expiry: boolean;
  revoked_at: string | null;
  rate_limit_tps: number | null;
  priority: number;
  usage_24h: TokenUsage24h;
}

interface StatusInfo {
  label: string;
  variant: "default" | "success" | "warning" | "error" | "info";
}

// Status precedence is defined in the §11.6 spec — encoded here as a
// short-circuit ladder so callers can't accidentally surface two states at
// once. The "Revoked" branch is a safety net: the backend filter strips
// revoked-and-not-rotated rows from the list response, but if the contract
// ever changes the row component still renders something sane.
function deriveStatus(item: TokenItem): StatusInfo {
  if (item.revoked_at != null && item.rotated_at == null) {
    return { label: "Revoked", variant: "error" };
  }
  if (item.is_expired) {
    return { label: "Expired", variant: "error" };
  }
  if (item.rotated_at != null && item.successor_deleted) {
    return { label: "Rotated (orphan)", variant: "error" };
  }
  if (item.rotated_at != null) {
    return { label: "Rotated (grace)", variant: "warning" };
  }
  if (item.is_near_expiry) {
    return { label: "Expiring soon", variant: "warning" };
  }
  return { label: "Active", variant: "success" };
}

// SQLite emits naive UTC strings like "2026-01-01 00:00:00" (see
// `sqlite_utc_now` in app/db/repos/tokens.py). `new Date(s)` parses these
// as local time on some browsers — append "Z" so it's unambiguously UTC.
function formatTs(value: string | null): string {
  if (!value) return "—";
  const isoish = value.includes("T") ? value : value.replace(" ", "T") + "Z";
  const d = new Date(isoish);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString();
}

// Compact human number — operators scan the 24h column for "is this token
// alive?" not for accountancy precision. 12345 → "12.3k", 1_234_567 → "1.2M".
function formatCompact(n: number): string {
  if (n < 1_000) return n.toString();
  if (n < 1_000_000) return `${(n / 1_000).toFixed(n < 10_000 ? 1 : 0)}k`;
  if (n < 1_000_000_000) return `${(n / 1_000_000).toFixed(n < 10_000_000 ? 1 : 0)}M`;
  return `${(n / 1_000_000_000).toFixed(1)}B`;
}

// Priority tier → Badge variant. 0..3 muted (default slate), 4..6 normal
// (info), 7..8 elevated (warning amber), 9 critical (error red). Visual
// loudness scales with starvation risk to the priorities BELOW this row,
// not the urgency of the row itself — a deliberate "this token outranks
// others" cue rather than a status indicator.
function priorityVariant(p: number): "default" | "info" | "warning" | "error" {
  if (p >= 9) return "error";
  if (p >= 7) return "warning";
  if (p >= 4) return "info";
  return "default";
}

// Tooltip text MUST warn about starvation under STRICT scheduling — this is
// a contract surface for the operator UI (see dispatch + docs/operating.md).
// Kept short enough to fit in a native title attribute; long-form lives in
// the docs page linked from the table header.
const PRIORITY_TOOLTIP =
  "STRICT scheduler: higher priority is ALWAYS served first. " +
  "A priority-0 token can wait indefinitely behind a steady stream of " +
  "priority-9 traffic (starvation by design).";

interface TestResult {
  ok: boolean;
  status: number;
  ms: number;
  detail?: string;
}

interface TokenRowProps {
  item: TokenItem;
  onChange: () => void;
}

export function TokenRow({ item, onChange }: TokenRowProps) {
  const [busy, setBusy] = useState(false);
  const [rotateOpen, setRotateOpen] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [testing, setTesting] = useState(false);
  const status = deriveStatus(item);
  const totalTokens24h =
    item.usage_24h.prompt_tokens + item.usage_24h.completion_tokens;

  async function onDelete() {
    if (busy) return;
    // confirm() is fine for a destructive-but-recoverable op like this —
    // worst case the operator deletes a token and has to rotate the
    // calling client. Reach for a real Modal-based confirm if we ever
    // need a richer surface (e.g. "type the token name to confirm").
    if (typeof window !== "undefined" && !window.confirm(
      `Delete token "${item.name}"? Any client using it will lose access immediately.`,
    )) {
      return;
    }
    setBusy(true);
    setDeleteError(null);
    try {
      const r = await authFetch(`/api/tokens/${encodeURIComponent(item.id)}`, {
        method: "DELETE",
      });
      if (!r.ok) {
        // Surface the error inline. Pattern matches the dialogs' inline
        // red text — a row-local <p role="alert"> below the Actions
        // cell. Without this branch the failed DELETE was silently
        // swallowed and the row popped back in on the next 10s poll,
        // leaving the operator with no idea what went wrong.
        const detail = await r.json().catch(() => null);
        const msg = (detail && typeof detail === "object" && "detail" in detail
          ? String((detail as { detail: unknown }).detail)
          : null) ?? `Failed to delete token (HTTP ${r.status})`;
        setDeleteError(msg);
      }
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : "Network error");
    } finally {
      setBusy(false);
      // Revalidating on error is harmless — the row will simply reappear
      // from the server's perspective, which is the correct state.
      onChange();
    }
  }

  function onRotateClose() {
    setRotateOpen(false);
    onChange();
  }

  async function onTest() {
    if (testing) return;
    setTesting(true);
    setTestResult(null);
    const t0 = performance.now();
    try {
      const r = await authFetch(`/api/tokens/${encodeURIComponent(item.id)}/test`, {
        method: "POST",
      });
      const ms = Math.round(performance.now() - t0);
      if (!r.ok) {
        // HTTP-level failure (404, 5xx, etc.) — surface the backend's
        // error detail; fall back to status text.
        const body = await r.json().catch(() => null);
        const detail = (body && typeof body === "object" && "detail" in body
          ? String((body as { detail: unknown }).detail)
          : undefined) ?? r.statusText;
        setTestResult({ ok: false, status: r.status, ms, detail });
        return;
      }
      // 200 OK — the backend always returns `ok: true` for a valid lookup;
      // the operator-relevant failure modes are encoded in the payload
      // (revoked/expired/allowed_models=[]/proxy unreachable).
      const body = await r.json().catch(() => null) as
        | {
            ok: boolean;
            revoked: boolean;
            expired: boolean;
            proxy_reachable: boolean;
            allowed_models: string[];
          }
        | null;
      if (!body) {
        setTestResult({ ok: false, status: r.status, ms, detail: "empty response body" });
        return;
      }
      const semanticIssues: string[] = [];
      if (body.revoked) semanticIssues.push("token is revoked");
      if (body.expired) semanticIssues.push("token is expired");
      if (!body.proxy_reachable) semanticIssues.push("proxy /healthz unreachable");
      if (body.allowed_models.length === 0) {
        semanticIssues.push("no loaded models match this token's scope");
      }
      if (semanticIssues.length > 0) {
        setTestResult({
          ok: false,
          status: r.status,
          ms,
          detail: semanticIssues.join("; "),
        });
        return;
      }
      setTestResult({
        ok: true,
        status: r.status,
        ms,
        detail: `${body.allowed_models.length} model(s) reachable`,
      });
    } catch (err) {
      const ms = Math.round(performance.now() - t0);
      setTestResult({
        ok: false,
        status: 0,
        ms,
        detail: err instanceof Error ? err.message : "Network error",
      });
    } finally {
      setTesting(false);
    }
  }

  return (
    <>
      <tr className={deleteError || testResult ? "" : "border-b border-slate-800"}>
        <td className="px-2 py-2 font-medium">{item.name}</td>
        <td className="px-2 py-2 font-mono text-xs">{item.prefix}</td>
        <td className="px-2 py-2 text-xs text-slate-400" data-testid="token-created">
          {formatTs(item.created_at)}
        </td>
        <td className="px-2 py-2 text-xs text-slate-400">{formatTs(item.expires_at)}</td>
        <td className="px-2 py-2 text-xs text-slate-400">{formatTs(item.last_used_at)}</td>
        <td
          className="px-2 py-2 text-right font-mono text-xs tabular-nums text-slate-300"
          data-testid="token-rate"
        >
          {item.rate_limit_tps == null
            ? <span className="text-slate-500">unlimited</span>
            : <>{item.rate_limit_tps.toLocaleString()}<span className="text-slate-500"> tps</span></>}
        </td>
        <td className="px-2 py-2" data-testid="token-priority">
          <Badge
            variant={priorityVariant(item.priority)}
            title={PRIORITY_TOOLTIP}
            // tabIndex so keyboard users can also surface the native tooltip
            // via focus on browsers that honour `title` on focus.
            tabIndex={0}
            aria-label={`Priority ${item.priority}. ${PRIORITY_TOOLTIP}`}
          >
            P{item.priority}
          </Badge>
        </td>
        <td
          className="px-2 py-2 text-right font-mono text-xs tabular-nums text-slate-300"
          // Tooltip exposes the exact request count + prompt/completion split
          // for operators who need numbers (e.g. cost reconciliation). The
          // visible cell stays compact so the row doesn't blow up.
          title={
            `${item.usage_24h.requests.toLocaleString()} requests · ` +
            `${item.usage_24h.prompt_tokens.toLocaleString()} prompt + ` +
            `${item.usage_24h.completion_tokens.toLocaleString()} completion tokens`
          }
          data-testid="token-usage-24h"
        >
          {item.usage_24h.requests === 0 ? (
            <span className="text-slate-500">—</span>
          ) : (
            <>
              {formatCompact(item.usage_24h.requests)}
              <span className="text-slate-500"> req · </span>
              {formatCompact(totalTokens24h)}
              <span className="text-slate-500"> tok</span>
            </>
          )}
        </td>
        <td className="px-2 py-2">
          <Badge variant={status.variant}>{status.label}</Badge>
        </td>
        <td className="px-2 py-2 text-right">
          <div className="flex justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              aria-label="Test token authentication"
              onClick={onTest}
              disabled={testing}
              data-testid="token-test"
            >
              {testing ? "Testing…" : "Test"}
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              aria-label="Rotate"
              onClick={() => setRotateOpen(true)}
              disabled={busy || item.rotated_at != null}
              // Disable rotate on already-rotated rows — rotating the
              // predecessor would chain a second grace period. If we
              // want that, we'd add it as an explicit "extend grace"
              // action rather than overloading rotate.
            >
              Rotate
            </Button>
            <Button
              type="button"
              variant="destructive"
              size="sm"
              onClick={onDelete}
              disabled={busy}
            >
              Delete
            </Button>
          </div>
        </td>
      </tr>
      {testResult && (
        // Sub-row hosting the test result. Same pattern as deleteError —
        // separate <tr> so a long error message doesn't reflow the actions.
        // colSpan covers all 10 columns of the table from page.tsx.
        <tr className={deleteError ? "" : "border-b border-slate-800"}>
          <td colSpan={10} className="px-2 pb-2 text-right">
            <p
              role="status"
              className={`text-sm ${testResult.ok ? "text-emerald-400" : "text-red-500"}`}
            >
              {testResult.ok
                ? `OK — ${testResult.detail ?? "verified"} (${testResult.ms} ms)`
                : `FAIL — HTTP ${testResult.status || "n/a"} (${testResult.ms} ms)` +
                  (testResult.detail ? `: ${testResult.detail}` : "")}
            </p>
          </td>
        </tr>
      )}
      {deleteError && (
        // Sub-row hosting the delete error. colSpan covers the 10-column
        // table from page.tsx (was 7 before S5 added Rate/Priority/Last24h).
        // We render it as a separate <tr> rather than overflowing the
        // Actions cell because the error message can be long enough to push
        // the action buttons around.
        <tr className="border-b border-slate-800">
          <td colSpan={10} className="px-2 pb-2 text-right">
            <p role="alert" className="text-sm text-red-500">
              {deleteError}
            </p>
          </td>
        </tr>
      )}
      <RotateTokenDialog
        open={rotateOpen}
        tokenId={item.id}
        onClose={onRotateClose}
      />
    </>
  );
}
