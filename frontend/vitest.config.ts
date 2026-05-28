import { defineConfig } from 'vitest/config';
import path from 'node:path';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: false,
    setupFiles: ['./tests/setup.ts'],
    exclude: ['node_modules/**', 'tests/e2e/**'],
  },
  esbuild: { jsx: 'automatic' },
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
});
