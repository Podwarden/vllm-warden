FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

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
