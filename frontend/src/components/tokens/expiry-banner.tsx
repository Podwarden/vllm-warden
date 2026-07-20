"use client";

import type { TokenItem } from "./token-row";

interface ExpiryBannerProps {
  items: TokenItem[];
}

export function ExpiryBanner({ items }: ExpiryBannerProps) {
  // Filter strictly by `is_near_expiry` — the backend already excludes
  // already-expired tokens from that flag (see _enrich in
  // app/tokens/routes_api.py), so the "expiring soon" copy stays honest.
  const expiring = items.filter((it) => it.is_near_expiry);
  if (expiring.length === 0) return null;

  return (
    <div
      role="alert"
      className="rounded-md border border-amber-500/40 bg-amber-100/10 p-3 text-amber-200"
    >
      <p className="text-sm font-semibold">
        Tokens expiring soon ({expiring.length})
      </p>
      <ul className="mt-1 list-disc pl-5 text-xs text-amber-100/90">
        {expiring.map((it) => (
          <li key={it.id}>
            <span className="font-medium">{it.name}</span>
            <span className="ml-2 font-mono text-amber-200/70">{it.prefix}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
