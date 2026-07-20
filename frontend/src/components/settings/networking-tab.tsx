"use client";

import {
  RUNTIME_NETWORKING_KEYS,
  RUNTIME_HINTS,
  type FieldHint,
} from "@/lib/settings-hints";
import { useRuntimeSettings } from "@/components/settings/hooks/use-runtime-settings";
import { SettingsTabShell } from "@/components/settings/settings-tab-shell";
import { SettingSection } from "@/components/settings/setting-section";
import { RuntimeField } from "@/components/settings/runtime-field";

// ---------------------------------------------------------------------------
// Networking tab — Public access section (public_url + landing_page_enabled).
//
// This tab is the home of the new #154 `public_url` field. The field
// powers `getPublicBaseUrl()` so user-facing snippets (curl examples,
// OpenAI client configs) render the right hostname when the warden is
// behind a reverse proxy.
//
// Today only one section. The spec leaves the tab as the natural home
// for future networking knobs (TLS cert reload, CORS allowlist, trusted
// proxy ranges) — those are filed as follow-ups, not in scope here.
// ---------------------------------------------------------------------------

export function NetworkingTab() {
  const controls = useRuntimeSettings(RUNTIME_NETWORKING_KEYS);
  const { draft, editing, saving, setField } = controls;
  const disabled = !editing || saving;

  return (
    <SettingsTabShell title="Networking" controls={controls}>
      {draft && (
        <SettingSection
          title="Public access"
          subtitle="How clients outside the host see this warden."
        >
          {RUNTIME_NETWORKING_KEYS.map((k) => {
            const hint: FieldHint | undefined = RUNTIME_HINTS[k];
            if (!hint) return null;
            return (
              <RuntimeField
                key={k}
                fieldKey={k}
                hint={hint}
                draft={draft}
                setField={setField}
                disabled={disabled}
              />
            );
          })}
        </SettingSection>
      )}
    </SettingsTabShell>
  );
}
