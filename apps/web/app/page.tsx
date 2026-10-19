"use client"

import { useState, useRef, useEffect } from "react"

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
const LANGUAGES = [
  { code: "en", name: "English" },
  { code: "hi", name: "हिन्दी (Hindi)" },
]

export default function Home() {
  const [image, setImage] = useState<string | null>(null)
  const [language, setLanguage] = useState("en")
  const [crop, setCrop] = useState("rice")
  const [loading, setLoading] = useState(false)
const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const crops = ["rice", "wheat", "cotton", "maize", "tomato"]

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setError(null)
    setResult(null)
    const reader = new FileReader()
    reader.onloadend = () => setImage(reader.result as string)
    reader.readAsDataURL(file)
  }

  const submit = async () => {
    if (!image) return
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      // Convert data URL to blob
      const res = await fetch(image)
      const blob = await res.blob()
      form.append("file", blob, "leaf.jpg")
      const resp = await fetch(`${API}/advice/image?language=${language}`, {
        method: "POST",
        body: form,
      })
      if (!resp.ok) throw new Error("Advice request failed")
      const data = await resp.json()
      setResult(data)
    } catch (err: any) {
      setError(err.message || "Unknown error")
    } finally {
      setLoading(false)
    }
  }

  const speak = async () => {
    if (!result?.advice) return
    try {
      const resp = await fetch(`${API}/speak`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: result.advice, language: result.language }),
      })
      if (!resp.ok) throw new Error("TTS failed")
      const blob = await resp.blob()
      const url = URL.createObjectURL(blob)
      if (audioRef.current) audioRef.current.src = url
      else audioRef.current = new Audio(url)
      audioRef.current.play()
    } catch (err: any) {
      setError(err.message || "TTS unavailable")
    }
  }

  return (
    <div className="min-h-screen p-4">
      <div className="max-w-md mx-auto">
        <header className="mb-6">
          <h1 className="text-2xl font-bold text-center">🌿 Crop Disease Advisor</h1>
          <p className="text-sm text-center text-gray-600">
            Photo • Voice • Language
          </p>
        </header>

        {/* Language + Crop selectors */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium mb-2">Language</label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-lg"
            >
              {LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Crop</label>
            <select
              value={crop}
              onChange={(e) => setCrop(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-lg"
            >
              {crops.map((c) => (
                <option key={c} value={c}>
                  {c.charAt(0).toUpperCase() + c.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Image preview + upload */}
        <div className="mb-6">
          {image ? (
            <div className="relative">
              <img src={image} alt="Leaf" className="w-full h-64 object-cover rounded-lg border" />
              <input
                type="file"
                accept="image/*"
                onChange={handleFile}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
            </div>
          ) : (
            <div className="aspect-video w-full bg-gray-200 rounded-lg border-2 border-dashed flex flex-col items-center justify-center">
              <input
                type="file"
                accept="image/*"
                onChange={handleFile}
                className="hidden"
                id="file-input"
              />
              <button
                onClick={() => document.getElementById("file-input")?.click()}
                className="text-green-600"
              >
                📷 Take / upload leaf photo
              </button>
              <p className="text-xs text-gray-500 mt-2">JPG or PNG</p>
            </div>
          )}
        </div>

        {/* Action buttons */}
        <div className="flex gap-3 mb-6">
          <button
            onClick={submit}
            disabled={loading || !image}
            className="flex-1 bg-green-600 text-white py-4 rounded-lg font-semibold disabled:opacity-50"
          >
            {loading ? "Analyzing…" : "Diagnose"}
          </button>
          <button
            onClick={speak}
            disabled={!result}
            className="bg-blue-600 text-white py-4 rounded-lg font-semibold disabled:opacity-50"
          >
            🔊 Speak
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6 p-3 bg-red-100 border border-red-400 text-red-800 rounded-lg text-sm">
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-4">
            <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
              <h2 className="text-lg font-bold mb-2">Diagnosis</h2>
              <p className="text-gray-700">
                {result.diagnosis.predictions
                  .map((p: any) => `${p.label}: ${(p.score * 100).toFixed(0)}%`)
                  .join(", ")}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Confidence: {(result.confidence * 100).toFixed(0)}% |{" "}
                {result.safe ? "✅ Safe" : "⚠️ Review needed"}
              </p>
            </div>

            <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
              <h2 className="text-lg font-bold mb-2">Advice</h2>
              <p className="text-gray-800">{result.advice}</p>
              {result.citations?.length && (
                <p className="text-xs text-gray-500 mt-2">
                  Sources: {result.citations.join(", ")}
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
