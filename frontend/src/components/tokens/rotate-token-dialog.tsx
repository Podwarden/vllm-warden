"use client";

import { useId, useState } from "react";
import { Modal } from "@/components/ui/modal";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { authFetch } from "@/lib/auth-fetch";
import { copyToClipboard } from "@/lib/utils";

interface RotateTokenDialogProps {
  open: boolean;
  tokenId: string;
  onClose: () => void;
}

const GRACE_MIN = 0;
const GRACE_MAX = 720;
const DAYS_MIN = 0;
const DAYS_MAX = 3650;

interface RotateResponse {
  id: string;
  // #150 — the freshly-minted row keeps the ORIGINAL name; the old row
  // is renamed to `"{name} (old N)"`. Both names are surfaced so the
  // success modal can tell the operator exactly where each token went.
  name: string;
  plaintext: string;
  prefix?: string;
  rotated_from: string;
  renamed_to: string;
}

export function RotateTokenDialog({ open, tokenId, onClose }: RotateTokenDialogProps) {
  const [graceHours, setGraceHours] = useState("24");
  const [expiresInDays, setExpiresInDays] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // Plaintext lives in component-local state and is wiped on close. The
  // §11.6 spec is explicit: "After the user dismisses the dialog, the
  // plaintext must NOT remain accessible — clear it from state on close."
  const [rotated, setRotated] = useState<RotateResponse | null>(null);
  const [copied, setCopied] = useState(false);
  // Tracks a clipboard write failure (HTTP context, permissions denied,
  // missing API). Same reasoning as the create dialog: plaintext is
  // surfaced once and we cannot silently swallow a failed copy.
  const [copyFailed, setCopyFailed] = useState(false);

  const graceId = useId();
  const daysId = useId();

  function reset() {
    setGraceHours("24");
    setExpiresInDays("");
    setError(null);
    setRotated(null);
    setCopied(false);
    setCopyFailed(false);
  }

  function handleClose() {
    reset();
    onClose();
  }

  async function submit() {
    setError(null);

    const grace = Number(graceHours);
    if (!Number.isInteger(grace) || grace < GRACE_MIN || grace > GRACE_MAX) {
      setError(`Grace hours must be an integer between ${GRACE_MIN} and ${GRACE_MAX}`);
      return;
    }

    const body: Record<string, unknown> = { grace_hours: grace };
    if (expiresInDays.trim()) {
      const days = Number(expiresInDays);
      if (!Number.isInteger(days) || days < DAYS_MIN || days > DAYS_MAX) {
        setError(`Expires in days must be an integer between ${DAYS_MIN} and ${DAYS_MAX} (0 = never)`);
        return;
      }
      body.expires_in_days = days;
    }

    setBusy(true);
    try {
      const r = await authFetch(`/api/tokens/${encodeURIComponent(tokenId)}/rotate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const detail = await r.json().catch(() => null);
        const msg = (detail && typeof detail === "object" && "detail" in detail
          ? String((detail as { detail: unknown }).detail)
          : null) ?? `Failed to rotate token (HTTP ${r.status})`;
        setError(msg);
        return;
      }
      const payload = (await r.json()) as RotateResponse;
      setRotated(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Network error");
    } finally {
      setBusy(false);
    }
  }

  async function handleCopy() {
    if (!rotated) return;
    setCopyFailed(false);
    // See create-token-dialog: shared copyToClipboard handles the
    // non-secure-context (#149) fallback and only throws when both
    // navigator.clipboard and document.execCommand("copy") fail.
    try {
      await copyToClipboard(rotated.plaintext);
      setCopied(true);
    } catch {
      setCopyFailed(true);
    }
  }

  return (
    <Modal
      open={open}
      onClose={busy ? () => {} : handleClose}
      title={rotated ? "Token rotated" : "Rotate token"}
    >
      {rotated ? (
        <div className="space-y-3">
          <p className="text-sm text-slate-300">
            This is the only time you&apos;ll see this token — copy it now.
          </p>
          <pre className="select-all whitespace-pre-wrap break-all rounded-md border border-slate-700 bg-slate-950 p-3 font-mono text-sm">
            {rotated.plaintext}
          </pre>
          {copyFailed && (
            <p className="text-sm text-red-500">
              Copy failed — select and copy the token manually.
            </p>
          )}
          <p className="text-xs text-slate-500">
            New active token: <span className="font-mono">{rotated.name}</span>.
            The previous token was renamed to{" "}
            <span className="font-mono">{rotated.renamed_to}</span> and will keep
            working through the grace period, then expire.
          </p>
          <div className="flex justify-end gap-2 pt-2">
            <Button type="button" variant="outline" size="sm" onClick={handleCopy}>
              {copied ? "Copied" : "Copy"}
            </Button>
            <Button type="button" onClick={handleClose}>Done</Button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <p className="text-sm text-slate-300">
            Issue a new token to replace this one. The existing token keeps
            working for the grace period below, then expires.
          </p>

          <label htmlFor={graceId} className="block space-y-1">
            <span className="text-sm">Grace period (hours)</span>
            <Input
              id={graceId}
              type="number"
              value={graceHours}
              onChange={(e) => setGraceHours(e.target.value)}
              min={GRACE_MIN}
              max={GRACE_MAX}
              inputMode="numeric"
            />
            <span className="text-xs text-slate-500">
              How long the old token keeps working. Default 24h, max 720h (30 days).
            </span>
          </label>

          <label htmlFor={daysId} className="block space-y-1">
            <span className="text-sm">New token expires in days (optional)</span>
            <Input
              id={daysId}
              type="number"
              value={expiresInDays}
              onChange={(e) => setExpiresInDays(e.target.value)}
              placeholder="inherit from old token"
              min={DAYS_MIN}
              max={DAYS_MAX}
              inputMode="numeric"
            />
            <span className="text-xs text-slate-500">
              Leave blank to inherit the predecessor&apos;s expiry. 0 = never.
            </span>
          </label>

          {error && <p className="text-sm text-red-500">{error}</p>}

          <div className="flex justify-end gap-2 pt-2">
            <Button type="button" variant="outline" onClick={handleClose} disabled={busy}>
              Cancel
            </Button>
            <Button type="button" onClick={submit} disabled={busy}>
              {busy ? "Rotating…" : "Rotate"}
            </Button>
          </div>
        </div>
      )}
    </Modal>
  );
}
