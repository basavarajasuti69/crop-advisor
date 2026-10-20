/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: [],
    remotePatterns: [],
  },
  // Static export for Azure Static Web Apps. Set NEXT_OUTPUT=export for the
  // deploy build; the default build is unchanged.
  ...(process.env.NEXT_OUTPUT === "export" ? { output: "export" } : {}),
}

module.exports = nextConfig
