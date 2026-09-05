import './globals.css';
import { ThemeProvider } from '@/lib/theme';
import { NavBar } from '@/components/nav-bar';

export const metadata = { title: 'LLM Warden', description: 'LLM operator UI' };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider>
          <NavBar />
          {/* id="app-root" is the target of @podwarden/chat-ui's Modal
              rootInertId — chat2/page.tsx passes rootInertId="app-root" so
              an open modal can set `inert` on everything outside itself. */}
          <main id="app-root" className="container mx-auto p-6">
            {children}
          </main>
        </ThemeProvider>
      </body>
    </html>
  );
}
