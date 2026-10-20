# Next.js PWA - Crop Advisor

## Local development

```bash
# Install (in apps/web/)
npm install

# Run
npm run dev
```

The backend FastAPI must be running at `NEXT_PUBLIC_API_URL` (default
`http://localhost:8000`).

## Build

The app is deployed as a **static export** (no Node server at runtime).
`next.config.js` switches to `output: "export"` when `NEXT_OUTPUT=export`
is set, so the deploy build is:

```bash
NEXT_OUTPUT=export npm run build
```

This writes the site to `apps/web/out`.

`NEXT_PUBLIC_API_URL` is inlined into the JavaScript bundle **at build
time**. Changing it needs a new build, not just a restart.

## Deploy to Netlify

The repository root has a `netlify.toml` with the build settings:

```toml
[build]
  base = "apps/web"
  command = "npm ci && NEXT_OUTPUT=export npm run build"
  publish = "out"
```

Steps:

1. In Netlify: **Add new site -> Import an existing project** and pick
   this repository.
2. The build settings come from `netlify.toml`; leave them as they are.
3. Set the environment variable (Netlify: **Site settings -> Environment
   variables -> Add**):

   | Key | Value |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | `https://your-api-host` |

4. **Deploy site.**

Set the variable before the first deploy - it is baked into the bundle,
so a later change needs a rebuild (Netlify: **Triggers -> Clear build
cache and deploy site**).

## The free API sleeps

A free API service goes to sleep after about 15 minutes of inactivity.
The first request after that has to wake it up, which can take up to
about a minute. The page shows a short "Waking up the server, this can
take up to a minute." message after 5 seconds of waiting, in the
language that is selected (the text lives in `strings/*.json`, shared
with the API).

The "Listen to advice" button is disabled with a clear label after the
API reports that no speech provider is configured, so it does not look
broken.

## Note for local development

`next dev` and `next build` share the `.next` directory. Stop the dev
server before running a build, or the build can fail with a
`PageNotFoundError` while the dev server rewrites the same files.
