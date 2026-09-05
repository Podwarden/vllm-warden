import { defineConfig } from 'vitest/config';
import path from 'node:path';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: false,
    setupFiles: ['./tests/setup.ts'],
    // `tests/conformance/**` needs a live backend and node's fetch; it runs
    // only via `npm run test:conformance` (vitest.conformance.config.ts),
    // which `make conformance` drives after booting the server.
    exclude: ['node_modules/**', 'tests/e2e/**', 'tests/conformance/**'],
    server: {
      deps: {
        // `@podwarden/chat-ui` imports `katex/dist/katex.min.css` for math
        // rendering. Left externalized, that import is handed to Node's ESM
        // loader, which throws ERR_UNKNOWN_FILE_EXTENSION on `.css` before a
        // single test collects. Inlining routes the package through Vite,
        // which understands CSS imports (and, with `css` off by default,
        // resolves them to nothing). A `vi.mock` cannot fix this: mocks only
        // intercept Vite-transformed imports, and the offending one lives
        // inside the externalized package.
        inline: ['@podwarden/chat-ui'],
      },
    },
  },
  esbuild: { jsx: 'automatic' },
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
});
