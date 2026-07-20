'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { setAccessToken } from '@/lib/auth-fetch';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  // Entry gate: until we know setup is complete, render nothing so the
  // sign-in form does NOT flash for a fresh install (which must be funneled
  // to the setup wizard instead). On fetch failure we fall back to showing
  // the form so a transient backend hiccup can't trap the user.
  const [setupChecked, setSetupChecked] = useState(false);
  const router = useRouter();

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // Public, cookie-less probe — omit credentials.
        const r = await fetch('/api/setup/state', { credentials: 'omit' });
        if (!r.ok) throw new Error(`setup state ${r.status}`);
        const { done } = await r.json();
        if (cancelled) return;
        // Fail toward the wizard: only a successful probe that explicitly
        // reports done === true shows the sign-in form. A malformed/non-boolean
        // body funnels to /setup/welcome rather than exposing a sign-in form
        // for an account that may not exist. (Probe FAILURE is handled by the
        // catch below, which falls back to the form so a hiccup can't trap.)
        if (done !== true) {
          router.replace('/setup/welcome');
          return;
        }
      } catch {
        // Network/5xx — fall through to the login form rather than trap.
      }
      if (!cancelled) setSetupChecked(true);
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const r = await fetch('/api/auth/login', {
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/json', 'Origin': window.location.origin },
      body: JSON.stringify({ username, password }),
    });
    if (!r.ok) { setError('Invalid credentials'); return; }
    const { access_token, expires_in } = await r.json();
    // Pass expires_in (#97) so auth-fetch schedules the proactive refresh
    // timer at 80% of the access TTL. Without this the timer would only
    // start firing after the first user-driven refresh, leaving the
    // initial 15-minute window vulnerable to backend bounces.
    setAccessToken(
      access_token,
      typeof expires_in === 'number' ? expires_in : undefined,
    );
    router.replace('/models');
  }

  if (!setupChecked) return null;

  return (
    <form onSubmit={submit} className="max-w-sm mx-auto mt-20 space-y-4">
      <h1 className="text-xl font-semibold">vllm-warden</h1>
      <label className="block">Username<Input name="username" value={username} onChange={(e) => setUsername(e.target.value)} /></label>
      <label className="block">Password<Input name="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} /></label>
      {error && <p className="text-red-500 text-sm">{error}</p>}
      <Button type="submit">Log in</Button>
    </form>
  );
}
