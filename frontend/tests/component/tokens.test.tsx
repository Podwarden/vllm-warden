import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { RotateTokenDialog } from '@/components/tokens/rotate-token-dialog';
import { CreateTokenDialog } from '@/components/tokens/create-token-dialog';
import { TokenRow, type TokenItem } from '@/components/tokens/token-row';
import { ExpiryBanner } from '@/components/tokens/expiry-banner';
import { setAccessToken, setCsrfToken } from '@/lib/auth-fetch';

// The Modal uses portals (document.body) and the `inert` attribute on
// #app-root; testing-library's default cleanup pulls the portal back out
// between renders only when globals: true, but vitest.config sets
// globals: false. Each test must therefore `cleanup()` itself.

function fakeItem(overrides: Partial<TokenItem> = {}): TokenItem {
  return {
    id: 't1',
    name: 'ci-bot',
    prefix: 'vw_abcde',
    preview: 'vw_abcde',
    created_at: '2026-01-01 00:00:00',
    last_used_at: null,
    expires_at: null,
    rotated_at: null,
    rotated_from: null,
    successor_id: null,
    successor_deleted: false,
    is_expired: false,
    is_near_expiry: false,
    revoked_at: null,
    // S5 (#104) defaults — matches the backend's "unlimited / mid-priority /
    // no usage" defaults for a freshly minted token.
    rate_limit_tps: null,
    priority: 5,
    usage_24h: {
      requests: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
    },
    ...overrides,
  };
}

describe('RotateTokenDialog', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it('shows new plaintext once and a copy button', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"plaintext":"vw_newnewnew","rotated_from":"old","id":"new","name":"ci-bot","renamed_to":"ci-bot (old 1)"}', { status: 201 })));
    render(<RotateTokenDialog open tokenId="old" onClose={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /rotate/i }));
    await waitFor(() => expect(screen.getByText('vw_newnewnew')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: /copy/i })).toBeInTheDocument();
  });

  it('surfaces the new active name and the predecessor\'s renamed_to (#150)', async () => {
    // Pin both names so a future change that drops the "renamed_to" field
    // from the response (or the dialog stops reading it) surfaces in CI
    // instead of as a quiet UI regression — operators rely on the modal
    // to confirm WHICH row is now the active one.
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"plaintext":"vw_renametest","rotated_from":"old","id":"new","name":"prod-bot","renamed_to":"prod-bot (old 2)"}',
      { status: 201 },
    )));
    render(<RotateTokenDialog open tokenId="old" onClose={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /rotate/i }));
    await waitFor(() => expect(screen.getByText('vw_renametest')).toBeInTheDocument());
    // Active token name in the success footer.
    expect(screen.getByText('prod-bot')).toBeInTheDocument();
    // Predecessor's new name after the rename.
    expect(screen.getByText('prod-bot (old 2)')).toBeInTheDocument();
  });

  it('clears plaintext from DOM after close', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"plaintext":"vw_secret_value","rotated_from":"old","id":"new"}',
      { status: 201 },
    )));
    const onClose = vi.fn();
    render(<RotateTokenDialog open tokenId="old" onClose={onClose} />);
    fireEvent.click(screen.getByRole('button', { name: /rotate/i }));
    await waitFor(() => expect(screen.getByText('vw_secret_value')).toBeInTheDocument());

    // Click Done — this should call reset() (wiping the plaintext from
    // local state) and then onClose. The parent in this test deliberately
    // does NOT flip `open`, so the dialog stays mounted; that lets us
    // assert the wipe happened by checking the plaintext is no longer
    // rendered. If a future change removes reset() from handleClose,
    // this test will fail because the post-success view would still be
    // visible.
    fireEvent.click(screen.getByRole('button', { name: /done/i }));
    expect(onClose).toHaveBeenCalled();
    expect(screen.queryByText('vw_secret_value')).not.toBeInTheDocument();
  });

  it('uses execCommand fallback on non-secure context and reports Copied (#149)', async () => {
    // Mirrors the d5 production setup: navigator.clipboard is undefined
    // because the page is served over plain HTTP. Without the textarea
    // fallback added in #149 this test would land on the "select
    // manually" failure path — pinning the success path here prevents
    // a regression that drops the fallback.
    Object.defineProperty(window, 'isSecureContext', {
      value: false, configurable: true, writable: true,
    });
    vi.stubGlobal('navigator', { ...navigator, clipboard: undefined } as unknown as Navigator);
    // jsdom drops document.execCommand (deprecated WHATWG API); attach
    // an own property so vi.spyOn has something to wrap.
    (document as unknown as { execCommand: (cmd: string) => boolean }).execCommand = () => true;
    const execSpy = vi.spyOn(document, 'execCommand').mockReturnValue(true);

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"plaintext":"vw_fallback","rotated_from":"old","id":"new"}', { status: 201 })));
    render(<RotateTokenDialog open tokenId="old" onClose={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /rotate/i }));
    await waitFor(() => expect(screen.getByText('vw_fallback')).toBeInTheDocument());

    fireEvent.click(screen.getByRole('button', { name: /^copy$/i }));
    await waitFor(() => expect(screen.getByRole('button', { name: /copied/i })).toBeInTheDocument());
    expect(execSpy).toHaveBeenCalledWith('copy');
    expect(screen.queryByText(/select and copy the token manually/i)).not.toBeInTheDocument();
  });

  it('shows "select manually" only when BOTH copy paths fail (#149)', async () => {
    Object.defineProperty(window, 'isSecureContext', {
      value: false, configurable: true, writable: true,
    });
    vi.stubGlobal('navigator', { ...navigator, clipboard: undefined } as unknown as Navigator);
    (document as unknown as { execCommand: (cmd: string) => boolean }).execCommand = () => false;
    vi.spyOn(document, 'execCommand').mockReturnValue(false);

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"plaintext":"vw_doomed","rotated_from":"old","id":"new"}', { status: 201 })));
    render(<RotateTokenDialog open tokenId="old" onClose={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: /rotate/i }));
    await waitFor(() => expect(screen.getByText('vw_doomed')).toBeInTheDocument());

    fireEvent.click(screen.getByRole('button', { name: /^copy$/i }));
    await waitFor(() =>
      expect(screen.getByText(/select and copy the token manually/i)).toBeInTheDocument(),
    );
  });
});

describe('CreateTokenDialog', () => {
  beforeEach(() => {
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it('shows plaintext once after successful create', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"id":"new","name":"ci-bot","plaintext":"vw_freshtoken","prefix":"vw_fresh","preview":"vw_fresh","expires_at":null}',
      { status: 201 },
    )));
    render(<CreateTokenDialog open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/name/i), { target: { value: 'ci-bot' } });
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));
    await waitFor(() => expect(screen.getByText('vw_freshtoken')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: /copy/i })).toBeInTheDocument();
  });

  it('clears plaintext from DOM after close', async () => {
    // Mirror of the RotateTokenDialog wipe test. Pins the §11.6
    // "plaintext must NOT remain accessible — clear it from state on
    // close" invariant for the create dialog too. Without this, removing
    // `reset()` from handleClose() would slip past CI.
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"id":"new","name":"ci-bot","plaintext":"vw_createdsecret","prefix":"vw_creat","preview":"vw_creat","expires_at":null}',
      { status: 201 },
    )));
    const onClose = vi.fn();
    render(<CreateTokenDialog open onClose={onClose} />);
    fireEvent.change(screen.getByLabelText(/name/i), { target: { value: 'ci-bot' } });
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));
    await waitFor(() => expect(screen.getByText('vw_createdsecret')).toBeInTheDocument());

    fireEvent.click(screen.getByRole('button', { name: /done/i }));
    expect(onClose).toHaveBeenCalled();
    expect(screen.queryByText('vw_createdsecret')).not.toBeInTheDocument();
  });

  it('uses execCommand fallback on non-secure context and reports Copied (#149)', async () => {
    Object.defineProperty(window, 'isSecureContext', {
      value: false, configurable: true, writable: true,
    });
    vi.stubGlobal('navigator', { ...navigator, clipboard: undefined } as unknown as Navigator);
    (document as unknown as { execCommand: (cmd: string) => boolean }).execCommand = () => true;
    const execSpy = vi.spyOn(document, 'execCommand').mockReturnValue(true);

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"id":"new","name":"ci-bot","plaintext":"vw_create_fallback","prefix":"vw_create_fb","preview":"vw_create_fb","expires_at":null}',
      { status: 201 },
    )));
    render(<CreateTokenDialog open onClose={() => {}} />);
    fireEvent.change(screen.getByLabelText(/name/i), { target: { value: 'ci-bot' } });
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));
    await waitFor(() => expect(screen.getByText('vw_create_fallback')).toBeInTheDocument());

    fireEvent.click(screen.getByRole('button', { name: /^copy$/i }));
    await waitFor(() => expect(screen.getByRole('button', { name: /copied/i })).toBeInTheDocument());
    expect(execSpy).toHaveBeenCalledWith('copy');
    expect(screen.queryByText(/select and copy the token manually/i)).not.toBeInTheDocument();
  });
});

describe('TokenRow', () => {
  afterEach(() => { cleanup(); });

  it('renders all spec columns and surfaces created_at', () => {
    render(
      <table><tbody>
        <TokenRow item={fakeItem({ created_at: '2026-02-03 10:11:12' })} onChange={() => {}} />
      </tbody></table>,
    );
    expect(screen.getByText('ci-bot')).toBeInTheDocument();
    expect(screen.getByText(/vw_abcde/)).toBeInTheDocument();
    // The created_at column renders something derived from the timestamp.
    // We don't pin on the exact locale string — just that it's there.
    const created = screen.getByTestId('token-created');
    expect(created.textContent).toBeTruthy();
  });

  it('shows "Expiring soon" status when is_near_expiry', () => {
    render(
      <table><tbody>
        <TokenRow item={fakeItem({ is_near_expiry: true })} onChange={() => {}} />
      </tbody></table>,
    );
    expect(screen.getByText(/expiring soon/i)).toBeInTheDocument();
  });

  it('shows "Expired" status when is_expired', () => {
    render(
      <table><tbody>
        <TokenRow item={fakeItem({ is_expired: true, is_near_expiry: false })} onChange={() => {}} />
      </tbody></table>,
    );
    expect(screen.getByText(/^expired$/i)).toBeInTheDocument();
  });

  it('shows "Rotated (orphan)" when rotated and successor deleted', () => {
    render(
      <table><tbody>
        <TokenRow
          item={fakeItem({ rotated_at: '2026-02-01 00:00:00', successor_deleted: true })}
          onChange={() => {}}
        />
      </tbody></table>,
    );
    expect(screen.getByText(/orphan/i)).toBeInTheDocument();
  });

  it('surfaces delete error to the operator', async () => {
    // Without the r.ok branch in token-row.tsx::onDelete, a 500 from
    // DELETE /api/tokens/:id was silently swallowed: the row would
    // disappear from the optimistic refresh, then reappear on the next
    // 10s poll with no explanation. This test pins that the failure
    // mode surfaces inline.
    setAccessToken('test-jwt');
    setCsrfToken('test-csrf');
    vi.stubGlobal('confirm', () => true);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      '{"detail":"token in use"}', { status: 500 },
    )));
    render(
      <table><tbody>
        <TokenRow item={fakeItem()} onChange={() => {}} />
      </tbody></table>,
    );
    fireEvent.click(screen.getByRole('button', { name: /delete/i }));
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
    expect(screen.getByRole('alert').textContent).toMatch(/token in use/i);
    vi.unstubAllGlobals();
  });
});

describe('ExpiryBanner', () => {
  afterEach(() => { cleanup(); });

  it('hides when no items are near expiry', () => {
    const { container } = render(
      <ExpiryBanner items={[fakeItem({ is_near_expiry: false })]} />,
    );
    expect(container.textContent).toBe('');
  });

  it('renders amber alert with names when near-expiry items exist', () => {
    render(
      <ExpiryBanner
        items={[
          fakeItem({ id: 'a', name: 'about-to-expire', is_near_expiry: true }),
          fakeItem({ id: 'b', name: 'fine', is_near_expiry: false }),
        ]}
      />,
    );
    const alert = screen.getByRole('alert');
    expect(alert.textContent).toMatch(/expiring soon/i);
    expect(alert.textContent).toMatch(/about-to-expire/);
    expect(alert.textContent).not.toMatch(/^.*\bfine\b.*$/);
  });
});
