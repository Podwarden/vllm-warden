import './globals.css';
import { ThemeProvider } from '@/lib/theme';
import { NavBar } from '@/components/nav-bar';

export const metadata = { title: 'vllm-warden', description: 'vLLM operator UI' };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider>
          <NavBar />
          <main className="container mx-auto p-6">{children}</main>
        </ThemeProvider>
      </body>
    </html>
  );
}
