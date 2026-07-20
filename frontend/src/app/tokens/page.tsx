"use client";

import { useState } from "react";
import useSWR from "swr";
import { authFetchJSON } from "@/lib/auth-fetch";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ExpiryBanner } from "@/components/tokens/expiry-banner";
import { TokenRow, type TokenItem } from "@/components/tokens/token-row";
import { CreateTokenDialog } from "@/components/tokens/create-token-dialog";

interface TokensResponse {
  items: TokenItem[];
}

export default function TokensPage() {
  const [createOpen, setCreateOpen] = useState(false);
  const { data, error, isLoading, mutate } = useSWR<TokensResponse>(
    "/api/tokens",
    authFetchJSON,
    // Match the cadence of the rest of the operator UI. The detail of
    // last_used_at moves slowly so 10s is fine, and pausing on hidden
    // tab keeps us off the API when nobody is watching.
    {
      refreshInterval: () =>
        typeof document !== "undefined" && document.hidden ? 0 : 10000,
    },
  );

  const items = data?.items ?? [];

  function onCreateClose() {
    setCreateOpen(false);
    // Best-effort revalidate — the dialog has already POSTed (if it did)
    // and the 10s poll will catch any miss.
    mutate().catch(() => {});
  }

  function onRowChange() {
    mutate().catch(() => {});
  }

  return (
    <div className="space-y-4">
      <ExpiryBanner items={items} />

      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">API tokens</h1>
        <Button onClick={() => setCreateOpen(true)}>Create token</Button>
      </div>

      {isLoading && (
        <div className="space-y-2">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      )}

      {error && !isLoading && (
        <p className="text-sm text-red-500">
          Failed to load tokens{error instanceof Error ? `: ${error.message}` : "."}
        </p>
      )}

      {!isLoading && !error && items.length === 0 && (
        <div className="rounded-md border border-dashed border-slate-700 bg-slate-900/30 p-8 text-center text-slate-400">
          <p className="text-sm">No tokens yet — create one to authenticate API clients.</p>
        </div>
      )}

      {!isLoading && !error && items.length > 0 && (
        <div className="overflow-x-auto rounded-md border border-slate-800">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-800 bg-slate-900/50 text-left text-xs uppercase text-slate-400">
              <tr>
                <th className="px-2 py-2">Name</th>
                <th className="px-2 py-2">Prefix</th>
                <th className="px-2 py-2">Created</th>
                <th className="px-2 py-2">Expires</th>
                <th className="px-2 py-2">Last used</th>
                <th className="px-2 py-2 text-right">Rate</th>
                <th
                  className="px-2 py-2"
                  title={
                    "STRICT priority scheduler — higher priority is always " +
                    "served first. See docs/operating.md for starvation semantics."
                  }
                >
                  Priority
                </th>
                <th className="px-2 py-2 text-right">Last 24h</th>
                <th className="px-2 py-2">Status</th>
                <th className="px-2 py-2 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {items.map((it) => (
                <TokenRow key={it.id} item={it} onChange={onRowChange} />
              ))}
            </tbody>
          </table>
        </div>
      )}

      <CreateTokenDialog open={createOpen} onClose={onCreateClose} />
    </div>
  );
}
