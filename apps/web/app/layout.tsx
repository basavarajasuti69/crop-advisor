import type { ReactNode } from "react"
import "./globals.css"

export const metadata = {
  title: "Fieldnote — Crop Advisor",
  description: "A thoughtful first step toward understanding what your crop is telling you.",
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#181818" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body>{children}</body>
    </html>
  )
}
