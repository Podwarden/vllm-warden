import type { Config } from "tailwindcss";

// eslint-disable-next-line @typescript-eslint/no-require-imports
const chatUiPreset = require('@podwarden/chat-ui/tailwind-preset') as Partial<Config>;

const config: Config = {
  presets: [chatUiPreset],
  darkMode: ["class"],
  content: [
    // EVERY directory that contains className strings must be listed: a class
    // used only in an unscanned file silently generates no CSS (the chat2
    // "-m-6"/"-translate-x-1/2" purge bug, 2026-08-24).
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/lib/**/*.{js,ts,jsx,tsx,mdx}",
    // the shared chat UI ships source-level Tailwind classes; scan its dist
    "./node_modules/@podwarden/chat-ui/dist/**/*.js",
  ],
  theme: {
    extend: {
      colors: {
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
      },
    },
  },
  plugins: [],
};

export default config;
