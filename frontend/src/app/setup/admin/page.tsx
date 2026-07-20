'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';

export default function AdminPage() {
  const router = useRouter();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!username.trim()) { setError('Username required'); return; }
    if (password.length < 6) { setError('Password must be at least 6 characters'); return; }
    // bcrypt silently truncates passwords longer than 72 bytes — reject up front.
    if (new TextEncoder().encode(password).length > 72) {
      setError('Password must be at most 72 bytes');
      return;
    }
    if (password !== confirm) { setError('Passwords do not match'); return; }

    setBusy(true);
    try {
      const r = await fetch('/api/setup/admin', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim(), password }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        setError(body.detail ?? 'Failed to create admin');
        return;
      }
      router.push('/setup/done');
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <h1 className="text-2xl font-semibold">Create admin account</h1>
      <p className="text-sm text-slate-400">
        This is the initial administrator. You can add more users later.
      </p>
      <label className="block space-y-1">
        <span className="text-sm">Username</span>
        <Input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" />
      </label>
      <label className="block space-y-1">
        <span className="text-sm">Password</span>
        <Input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="new-password"
        />
      </label>
      <label className="block space-y-1">
        <span className="text-sm">Confirm password</span>
        <Input
          type="password"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          autoComplete="new-password"
        />
      </label>
      {error && <p className="text-red-500 text-sm">{error}</p>}
      <Button type="submit" disabled={busy}>Create admin</Button>
    </form>
  );
}
