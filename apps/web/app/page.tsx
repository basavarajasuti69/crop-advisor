"use client"

import { useRef, useState } from "react"

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
const LANGUAGES = [
  { code: "en", name: "English" },
  { code: "hi", name: "हिन्दी" },
  { code: "kn", name: "ಕನ್ನಡ" },
]
const CROPS = ["rice", "wheat", "cotton", "maize", "tomato"]

type Prediction = { label: string; score: number }
type Diagnosis = {
  case_id: string
  crop: string
  language: string
  diagnosis: { predictions: Prediction[]; quality_ok: boolean; quality_issue?: string | null }
  advice: string
  citations: string[]
  confidence: number
  safe: boolean
  escalate: boolean
}

export default function Home() {
  const [image, setImage] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [language, setLanguage] = useState("en")
  const [crop, setCrop] = useState("rice")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<Diagnosis | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const handleFile = (selected?: File) => {
    if (!selected) return
    setError(null)
    setResult(null)
    setFile(selected)
    setImage(URL.createObjectURL(selected))
  }

  const submit = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append("file", file)
      form.append("crop", crop)
      form.append("language", language)
      const response = await fetch(`${API}/advice/image`, { method: "POST", body: form })
      if (!response.ok) {
        const detail = await response.json().catch(() => null)
        throw new Error(detail?.detail || "We couldn’t analyze that photo. Please try again.")
      }
      setResult(await response.json())
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Connection failed. Check that the advisor service is running.")
    } finally {
      setLoading(false)
    }
  }

  const speak = async () => {
    if (!result?.advice) return
    try {
      const response = await fetch(`${API}/speak`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: result.advice, language: result.language }),
      })
      if (!response.ok) throw new Error("Audio is not available right now.")
      const audioUrl = URL.createObjectURL(await response.blob())
      if (audioRef.current) audioRef.current.src = audioUrl
      else audioRef.current = new Audio(audioUrl)
      await audioRef.current.play()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Audio is not available right now.")
    }
  }

  return (
    <main>
      <section className="hero" id="top">
        <header className="topbar">
          <a className="wordmark" href="#top" aria-label="Crop Advisor home">
            <span className="wordmark-icon" aria-hidden="true">✳</span>
            <span>FIELDNOTE <small>BY CROP ADVISOR</small></span>
          </a>
          <nav className="top-nav" aria-label="Main navigation">
            <a href="#how-it-works">HOW IT WORKS</a>
            <a href="#diagnose">DIAGNOSE A LEAF</a>
          </nav>
          <label className="language-control">
            <span className="sr-only">Language</span>
            <select value={language} onChange={(event) => setLanguage(event.target.value)}>
              {LANGUAGES.map((item) => <option value={item.code} key={item.code}>{item.name}</option>)}
            </select>
          </label>
        </header>

        <div className="hero-copy">
          <p className="eyebrow"><span /> FIELD INTELLIGENCE FOR GROWING SEASONS</p>
          <h1>A clearer read<br />on every leaf.</h1>
          <p className="hero-description">A thoughtful first step toward understanding what your crop is telling you.</p>
          <a className="button button-primary" href="#diagnose">START A DIAGNOSIS <span aria-hidden="true">↗</span></a>
        </div>
        <div className="hero-caption"><span>01 / FIELD NOTES</span><span>BUILT FOR THE PEOPLE WHO GROW</span></div>
        <a className="scroll-cue" href="#how-it-works" aria-label="Scroll to learn more">↓</a>
      </section>

      <section className="intro section-wrap" id="how-it-works">
        <p className="eyebrow"><span /> OBSERVE · UNDERSTAND · ACT</p>
        <div className="intro-grid">
          <h2>Good advice<br />starts with a closer look.</h2>
          <p>Share a clear photo of a leaf. The advisor checks it against crop guidance and gives you a practical place to begin. When the result is uncertain, it will say so.</p>
        </div>
        <div className="steps">
          <article><span>01</span><h3>Bring a photo</h3><p>Choose a clear image of the affected leaf.</p></article>
          <article><span>02</span><h3>Add context</h3><p>Select the crop and language for your advice.</p></article>
          <article><span>03</span><h3>Review guidance</h3><p>Read the findings and listen to the response.</p></article>
        </div>
      </section>

      <section className="diagnosis-band" id="diagnose">
        <div className="section-wrap diagnosis-layout">
          <div className="diagnosis-copy">
            <p className="eyebrow"><span /> YOUR FIELD NOTE</p>
            <h2>Let’s take<br />a closer look.</h2>
            <p>Start with one leaf. A clear, well-lit photo helps us give you a more useful first read.</p>
            <div className="photo-tips"><span>PHOTO GUIDE</span><p>Use natural light · Keep the leaf in focus · Include the affected area</p></div>
          </div>

          <div className="diagnosis-panel">
            <div className="panel-heading"><span>NEW DIAGNOSIS</span><span>FIELDNOTE / 001</span></div>
            <button className={`upload-area ${image ? "has-image" : ""}`} onClick={() => fileInput.current?.click()} type="button">
              {image ? <img src={image} alt="Selected crop leaf" /> : <span className="upload-prompt"><span className="upload-mark">＋</span><strong>Choose a leaf photo</strong><span>JPG or PNG · Tap to browse</span></span>}
              <input
                ref={fileInput}
                className="sr-only"
                type="file"
                accept="image/*"
                capture="environment"
                onChange={(event) => handleFile(event.target.files?.[0])}
                aria-label="Choose a leaf photo"
              />
              {image && <span className="change-photo">CHANGE PHOTO</span>}
            </button>

            <div className="form-row">
              <label>CROP<select value={crop} onChange={(event) => setCrop(event.target.value)}>{CROPS.map((item) => <option key={item} value={item}>{item[0].toUpperCase() + item.slice(1)}</option>)}</select></label>
              <label>ADVICE LANGUAGE<select value={language} onChange={(event) => setLanguage(event.target.value)}>{LANGUAGES.map((item) => <option key={item.code} value={item.code}>{item.name}</option>)}</select></label>
            </div>
            <button className="button button-primary diagnose-button" onClick={submit} disabled={!file || loading}>
              {loading ? "READING YOUR PHOTO…" : "ANALYZE PHOTO"}<span aria-hidden="true">↗</span>
            </button>
            <p className="privacy-note">Your photo is used to prepare this diagnosis.</p>

            {error && <div className="notice notice-error" role="alert">{error}</div>}

            {result && (
              <div className="result-card" aria-live="polite">
                <div className="result-topline"><span>FIELD NOTE / {result.case_id.slice(0, 6).toUpperCase()}</span><span className={result.safe ? "status-safe" : "status-review"}>{result.safe ? "PRELIMINARY" : "REVIEW ADVISED"}</span></div>
                <h3>What we found</h3>
                {result.diagnosis.predictions.length > 0 ? (
                  <ul className="prediction-list">{result.diagnosis.predictions.map((prediction) => <li key={prediction.label}><span>{prediction.label.replaceAll("_", " ")}</span><strong>{Math.round(prediction.score * 100)}%</strong></li>)}</ul>
                ) : <p className="result-muted">{result.diagnosis.quality_issue || "No clear result yet. Try a closer, brighter photo."}</p>}
                <div className="result-advice"><p className="eyebrow"><span /> NEXT STEP</p><p>{result.advice}</p></div>
                {result.citations.length > 0 && <p className="citation-line">GUIDANCE: {result.citations.join(" · ")}</p>}
                <button className="button button-outline" onClick={speak}>LISTEN TO ADVICE <span aria-hidden="true">↗</span></button>
              </div>
            )}
          </div>
        </div>
      </section>

      <footer className="footer section-wrap">
        <a className="wordmark" href="#top"><span className="wordmark-icon" aria-hidden="true">✳</span><span>FIELDNOTE <small>BY CROP ADVISOR</small></span></a>
        <p>Clearer observations. Better growing decisions.</p>
        <a className="footer-top" href="#top">BACK TO TOP ↑</a>
      </footer>
    </main>
  )
}
