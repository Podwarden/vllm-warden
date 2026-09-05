# syntax=docker/dockerfile:1
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* .npmrc ./
# The GitLab registry token arrives as a BuildKit secret (env= needs the
# dockerfile 1.10+ frontend — hence the unpinned `1` syntax line). It is read
# by npm for this RUN only and never lands in a layer.
#
# `required=true`: without it a build that forgot `--secret id=npm,...` mounts
# an empty value and fails fast with "secret npm not found" instead of a late,
# confusing npm E401 partway through the install.
#
# The npm cache is a BuildKit cache mount, not a layer: `npm ci` deletes and
# recreates node_modules every time, so when this step IS invalidated (a
# package-lock.json change) the tarballs are already local and only the
# unpack cost remains. sharing=locked because npm's cache is not safe against
# concurrent writers, and build:ui / publish:images can overlap on a runner.
RUN --mount=type=secret,id=npm,env=NPM_TOKEN,required=true \
    --mount=type=cache,target=/root/.npm,sharing=locked \
    npm ci

FROM node:20-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# next.config.ts:rewrites() reads BACKEND_URL — but Next 15 standalone bakes
# rewrites into routes-manifest.json at BUILD time, so passing BACKEND_URL at
# runtime is a no-op. Hard-coding the Docker service name here works for both
# the local compose stack and deploy/hub/compose.yaml (both expose the API as
# `api:8080`). If a future deployment uses a different service name, this
# becomes a build arg.
ENV BACKEND_URL=http://api:8080

# #67 fix — bake the release identity into the UI bundle. Next.js
# inlines ``NEXT_PUBLIC_*`` env vars into the client bundle at build
# time, so consumers of these values (today: none; tomorrow: the
# version banner footer + e2e tooling) get the same identity as the
# backend's GET /api/version response. The default fallbacks ("dev" /
# "unknown") mirror the backend Dockerfile (see Dockerfile §74-77 in
# the repo root) so a no-build-arg local build produces "v? · ?" in
# the footer fallback path. CI passes both via the build:ui job in
# .gitlab-ci.yml. Manual release builds pass them via the buildx
# command in docs/releasing.md.
ARG VW_BUILD_VERSION=dev
ARG VW_BUILD_SHA=unknown
ENV NEXT_PUBLIC_VW_BUILD_VERSION=${VW_BUILD_VERSION} \
    NEXT_PUBLIC_VW_BUILD_SHA=${VW_BUILD_SHA}

# PERFORMANCE NOTE, deliberately not "fixed". NEXT_PUBLIC_VW_BUILD_SHA changes
# on every commit and is set immediately above this line, so this step is
# invalidated on every commit even when `COPY . .` reports CACHED -- measured at
# 53-70 s per build (job 228612). It cannot be reordered away: Next inlines
# NEXT_PUBLIC_* into the client bundle AT BUILD TIME, so the value has to be in
# the environment before `npm run build` runs, and moving the ENV after the
# build would silently return the footer to the `dev · unknown` banner that #45
# and #67 exist to prevent. Making the UI read its identity at runtime instead
# would recover the 53-70 s, but that is a product change to how the version
# banner works, not a build optimisation, and it belongs in its own MR.
RUN mkdir -p public && npm run build

FROM node:20-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production PORT=3000
RUN addgroup --system --gid 1001 nodejs \
 && adduser --system --uid 1001 nextjs
COPY --from=build --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=build --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=build --chown=nextjs:nodejs /app/public ./public
USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
