'use client';
import { usePathname } from 'next/navigation';

// keep in sync with STEPS in app/setup/state_machine.py
const steps = ['welcome', 'gpus', 'hf-token', 'admin', 'done'];

export default function SetupLayout({ children }: { children: React.ReactNode }) {
  const path = usePathname().split('/').pop() ?? '';
  return (
    <div className="max-w-xl mx-auto space-y-6">
      <ol className="flex gap-4 text-sm">
        {steps.map((s, i) => (
          <li key={s} className={path === s ? 'font-bold' : 'text-slate-500'}>{i + 1}. {s}</li>
        ))}
      </ol>
      {children}
    </div>
  );
}
