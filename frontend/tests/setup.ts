// Vitest setup — extends Vitest's `expect` with @testing-library/jest-dom
// matchers (`toBeInTheDocument`, `toHaveAttribute`, etc.). Importing the
// dedicated `/vitest` entry point auto-extends the matcher registry, so a
// bare `import` (no symbols) is sufficient.
import '@testing-library/jest-dom/vitest';
import React from 'react';
import { beforeEach, vi } from 'vitest';
import { __resetLoginRedirectInFlightForTests } from '@/lib/auth-fetch';

// v17.11 Bundle 2 — `react-virtuoso` uses ResizeObserver + getBoundingClientRect
// for window measurement, neither of which work in jsdom. The library
// silently renders zero items in this environment, breaking every test
// that asserts on a row rendered through Virtuoso.
//
// We replace Virtuoso with a plain "render all items in order" shim that
// passes through the props tests actually care about (style, role, data,
// itemContent, components.List). The shim also drops `followOutput` and
// `atBottomStateChange` since jsdom has no scroll geometry to drive them
// — tests that need to assert on sticky-mode behaviour should drive the
// `useStickyBottom` hook directly.
vi.mock('react-virtuoso', () => {
  type Row = { kind?: string; row?: unknown; event?: unknown } | unknown;
  function Virtuoso(props: {
    data?: ReadonlyArray<Row>;
    itemContent?: (index: number, item: Row) => React.ReactNode;
    computeItemKey?: (index: number, item: Row) => React.Key;
    style?: React.CSSProperties;
    components?: { List?: React.ComponentType<React.HTMLAttributes<HTMLDivElement>> };
  }) {
    const { data = [], itemContent, computeItemKey, style, components } = props;
    const children = data.map((item, idx) => {
      const key = computeItemKey ? computeItemKey(idx, item) : idx;
      return React.createElement(
        'div',
        { key, 'data-virtuoso-row-index': idx },
        itemContent ? itemContent(idx, item) : null,
      );
    });
    const ListComp = components?.List;
    if (ListComp) {
      return React.createElement('div', { style }, React.createElement(ListComp, {}, children));
    }
    return React.createElement('div', { style }, children);
  }
  return { Virtuoso };
});

// v2026.05.15.3 — `auth-fetch.ts` carries a module-level
// `loginRedirectInFlight` flag that de-dupes parallel 401 redirects in a
// real page. Module state persists across test files in vitest's worker,
// so a test that exhausts a 401 refresh leaves the flag stuck `true`
// and subsequent tests (e.g. sse-terminal-error checking 404/502/429)
// see the SSE preflight short-circuit to 401. Reset between every test
// so test isolation is preserved. Production code must never call this
// — the flag's job is to NOT reset until a real navigation reloads the
// module.
beforeEach(() => {
  __resetLoginRedirectInFlightForTests();
});
