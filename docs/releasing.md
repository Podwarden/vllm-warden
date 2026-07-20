# Releasing vllm-warden

## CalVer tag scheme

Tags follow `v{YYYY.MM.DD.N}` where `N` is a daily counter starting at `1`
(e.g. `v2026.05.15.1`, `v2026.05.15.2`).

Push an **annotated** tag to the `develop` branch tip after the MR merges:

```bash
git tag -a v2026.05.15.N -m "v2026.05.15.N"
git push origin v2026.05.15.N
```

Tagging triggers the CI tag pipeline (lint → test → build smoke).

## Manual image publish required

> **Important:** vllm-warden's `.gitlab-ci.yml` does **not** auto-push to
> `registry.podwarden.com`. The `build-image` CI job is a smoke test only —
> it builds locally and stops there, with no `docker push`. There are also no
> CI rules for tag pipelines, so a CalVer tag push has no build stage at all.
>
> Source: workspace memory note `project_vllm_warden_ci_publish_gap.md`
> (`/home/ip/.claude/projects/-home-ip-projects-pw/memory/project_vllm_warden_ci_publish_gap.md`).
> The root cause was confirmed during the 2026-05-11 v2026.05.11.2 incident —
> the cure commit shipped through git tag + Hub publish but the registry tag
> continued to point at pre-cure content because CI never pushed the image.

After the tag pipeline succeeds, the **release engineer** must publish
**both images** manually from a workstation with registry-push
credentials. There are two images per release — the backend
(`vllm-warden`) and the UI (`vllm-warden-ui`). Forgetting the UI is a
known foot-gun (closes vllm-warden#40); always run both commands.

### Backend image

```bash
docker buildx build \
  --push \
  --build-arg VW_BUILD_VERSION=vYYYY.MM.DD.N \
  --build-arg VW_BUILD_SHA=$(git rev-parse HEAD) \
  -t registry.podwarden.com/podwarden/apps/vllm-warden:vYYYY.MM.DD.N \
  -t registry.podwarden.com/podwarden/apps/vllm-warden:staging \
  .
```

### UI image

```bash
docker buildx build --push --platform linux/amd64 \
  --build-arg VW_BUILD_VERSION=vYYYY.MM.DD.N \
  --build-arg VW_BUILD_SHA=$(git rev-parse HEAD) \
  -t registry.podwarden.com/podwarden/apps/vllm-warden-ui:vYYYY.MM.DD.N \
  -t registry.podwarden.com/podwarden/apps/vllm-warden-ui:latest \
  -t registry.podwarden.com/podwarden/apps/vllm-warden-ui:production \
  frontend/
```

Replace `vYYYY.MM.DD.N` with the actual CalVer tag (e.g. `v2026.05.15.4`).
The `VW_BUILD_VERSION` and `VW_BUILD_SHA` build-args bake the real release
identity into each image so the `/api/version` endpoint (and the UI
version banner that reads it) shows the tag instead of the Dockerfile
defaults `dev` / `unknown`. Pass both args to **both** buildx commands —
the backend Dockerfile bakes them into the API response and the UI
Dockerfile bakes them into the static banner. Omitting them produced
the `vdev · unknown` banner on every v2026.05.17.* bonus build —
closes #45.
Use `:production` and `:latest` instead of `:staging` when releasing to
production (i.e. when merging to `main`). The UI command above writes
`:latest` + `:production` because the deploy host (mtl-pwh /
vllm.protrener.com) pulls `:latest` (the default in deploy/hub/compose.yaml);
`:production` is pushed as a human-readable bookmark matching the backend
convention. If you are publishing a develop-only build, swap both for `:staging`.

### Verify the digest before announcing the release

Run the digest check on **each** image — backend and UI both.

```bash
# Backend
docker manifest inspect \
  registry.podwarden.com/podwarden/apps/vllm-warden:vYYYY.MM.DD.N \
  | jq -r '.manifests[0].digest // .config.digest'

# UI
docker buildx imagetools inspect \
  registry.podwarden.com/podwarden/apps/vllm-warden-ui:vYYYY.MM.DD.N
```

Confirm each digest matches the digest printed by `docker buildx build --push`
at the end of the push. If they diverge the push silently failed — re-push
before announcing. (See workspace memory note
`project_vllm_warden_ci_publish_gap.md` — the registry tag and local
config digest must be verified after every push.)

### Kubelet image cache gotcha

`imagePullPolicy: IfNotPresent` caches by tag. If a node already pulled the
old tag you must evict it before `rollout restart` will pick up the new image:

```bash
sudo crictl rmi registry.podwarden.com/podwarden/apps/vllm-warden:vYYYY.MM.DD.N
```

Run this on every affected node, then restart the relevant deployment.

## Follow-up

Promote `build-image` to a proper publish job on tag pipelines so this stops
being a manual step. Until that work lands, do not assume the registry tag is
fresh just because the git tag exists.
