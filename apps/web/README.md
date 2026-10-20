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
   | `NEXT_PUBLIC_API_URL` | `https://<your-hf-space>.hf.space` |

4. **Deploy site.**

Set the variable before the first deploy - it is baked into the bundle,
so a later change needs a rebuild (Netlify: **Triggers -> Clear build
cache and deploy site**).
