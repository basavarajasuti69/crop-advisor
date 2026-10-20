/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: [],
    remotePatterns: [],
  },
  // The UI strings live in the shared strings/ directory at the repo root,
  // so the bundler has to be allowed to reach outside apps/web.
  experimental: {
    externalDir: true,
  },
  // Static export for static hosting (Netlify).
  // Set NEXT_OUTPUT=export for the deploy build; the default build is
  // unchanged.
  ...(process.env.NEXT_OUTPUT === "export" ? { output: "export" } : {}),
}

module.exports = nextConfig
