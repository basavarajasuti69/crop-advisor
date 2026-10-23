"use client"

import { useRef, useState, useCallback, useEffect } from "react"
import enStrings from "../../../strings/en.json"
import hiStrings from "../../../strings/hi.json"
import knStrings from "../../../strings/kn.json"

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
const LANGUAGES = [
  { code: "en", name: "English", bcp47: "en-IN" },
  { code: "hi", name: "हिन्दी", bcp47: "hi-IN" },
  { code: "kn", name: "ಕನ್ನಡ", bcp47: "kn-IN" },
]
const CROPS = ["rice", "wheat", "cotton", "maize", "tomato"]

const STRINGS: Record<string, Record<string, string | null>> = {
  en: enStrings,
  hi: hiStrings,
  kn: knStrings,
}

const WAKE_UP_DELAY_MS = 5000
const CONFIDENCE_LOW = 0.70

type Prediction = { label: string; score: number }
type Diagnosis = {
  case_id: string
  crop: string
  language: string
  diagnosis: {
    predictions: Prediction[]
    top_predictions?: Prediction[]
    confidence: number
    confidence_is_calibrated: boolean
    guidance_available: boolean
    quality_ok: boolean
    quality_issue?: string | null
  }
  advice: string
  citations: string[]
  confidence: number
  confidence_is_calibrated: boolean
  safe: boolean
  escalate: boolean
  disclaimer: string
}

export default function Home() {
  const [image, setImage] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [language, setLanguage] = useState("en")
  const [crop, setCrop] = useState("rice")
  const [loading, setLoading] = useState(false)
  const [slowLoading, setSlowLoading] = useState(false)
  const [audioUnavailable, setAudioUnavailable] = useState(false)
  const [result, setResult] = useState<Diagnosis | null>(null)
  const [error, setError] = useState<string | null>(null)

  // TTS state
  const [speaking, setSpeaking] = useState(false)
  const [paused, setPaused] = useState(false)
  const [voiceAvailable, setVoiceAvailable] = useState<boolean | null>(null)
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const fileInput = useRef<HTMLInputElement>(null)

  const t = (key: string): string => {
    const table = STRINGS[language] ?? STRINGS.en
    return (table[key] as string) || (STRINGS.en[key] as string) || key
  }

  // Check voice availability when language changes
  useEffect(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) {
      setVoiceAvailable(false)
      return
    }
    const checkVoices = () => {
      const lang = LANGUAGES.find((l) => l.code === language)
      const bcp47 = lang?.bcp47 || "en-IN"
      const voices = window.speechSynthesis.getVoices()
      const match = voices.some((v) => v.lang.startsWith(bcp47.split("-")[0]))
      setVoiceAvailable(match)
    }
    checkVoices()
    window.speechSynthesis.onvoiceschanged = checkVoices
    return () => {
      window.speechSynthesis.onvoiceschanged = null
    }
  }, [language])

  // Cancel speech when result changes or component unmounts
  useEffect(() => {
    return () => {
      if (typeof window !== "undefined" && window.speechSynthesis) {
        window.speechSynthesis.cancel()
      }
    }
  }, [])

  const handleFile = (selected?: File) => {
    if (!selected) return
    setError(null)
    setResult(null)
    setAudioUnavailable(false)
    setSpeaking(false)
    setPaused(false)
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel()
    }
    setFile(selected)
    setImage(URL.createObjectURL(selected))
  }

  const submit = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setSlowLoading(false)
    setSpeaking(false)
    setPaused(false)
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel()
    }
    const wakeUpTimer = setTimeout(() => setSlowLoading(true), WAKE_UP_DELAY_MS)
    try {
      const form = new FormData()
      form.append("file", file)
      form.append("crop", crop)
      form.append("language", language)
      const response = await fetch(`${API}/advice/image`, { method: "POST", body: form })
      if (!response.ok) {
        const detail = await response.json().catch(() => null)
        throw new Error(detail?.detail || "We couldn't analyze that photo. Please try again.")
      }
      setResult(await response.json())
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Connection failed. Check that the advisor service is running.")
    } finally {
      clearTimeout(wakeUpTimer)
      setSlowLoading(false)
      setLoading(false)
    }
  }

  const speakWithBrowser = useCallback((text: string) => {
    if (typeof window === "undefined" || !window.speechSynthesis) return false
    const lang = LANGUAGES.find((l) => l.code === language)
    const bcp47 = lang?.bcp47 || "en-IN"
    const voices = window.speechSynthesis.getVoices()
    const voice = voices.find((v) => v.lang === bcp47) || voices.find((v) => v.lang.startsWith(bcp47.split("-")[0]))
    if (!voice) return false

    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.voice = voice
    utterance.lang = bcp47
    utterance.rate = 0.9
    utterance.onstart = () => { setSpeaking(true); setPaused(false) }
    utterance.onend = () => { setSpeaking(false); setPaused(false) }
    utterance.onerror = () => { setSpeaking(false); setPaused(false) }
    utteranceRef.current = utterance
    window.speechSynthesis.speak(utterance)
    return true
  }, [language])

  const speakWithBackend = useCallback(async (text: string) => {
    try {
      const response = await fetch(`${API}/speak`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, language: result?.language || language }),
      })
      if (response.status === 503) {
        setAudioUnavailable(true)
        return
      }
      if (!response.ok) throw new Error(t("audio_unavailable"))
      const audioUrl = URL.createObjectURL(await response.blob())
      if (audioRef.current) {
        audioRef.current.src = audioUrl
      } else {
        audioRef.current = new Audio(audioUrl)
      }
      audioRef.current.onplay = () => { setSpeaking(true); setPaused(false) }
      audioRef.current.onended = () => { setSpeaking(false); setPaused(false) }
      audioRef.current.onpause = () => setPaused(true)
      await audioRef.current.play()
    } catch {
      setError(t("audio_unavailable"))
      setSpeaking(false)
      setPaused(false)
    }
  }, [language, result, t])

  const speak = useCallback(async () => {
    if (!result?.advice) return
    setAudioUnavailable(false)
    // Try browser speech first, fall back to backend
    if (!speakWithBrowser(result.advice)) {
      await speakWithBackend(result.advice)
    }
  }, [result, speakWithBrowser, speakWithBackend])

  const pauseResume = useCallback(() => {
    if (typeof window !== "undefined" && window.speechSynthesis && utteranceRef.current) {
      if (paused) {
        window.speechSynthesis.resume()
        setPaused(false)
      } else {
        window.speechSynthesis.pause()
        setPaused(true)
      }
    } else if (audioRef.current) {
      if (paused) {
        audioRef.current.play()
        setPaused(false)
      } else {
        audioRef.current.pause()
        setPaused(true)
      }
    }
  }, [paused])

  const stopSpeaking = useCallback(() => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel()
    }
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
    }
    setSpeaking(false)
    setPaused(false)
  }, [])

  const predictions = result?.diagnosis?.predictions || []
  const topPredictions = result?.diagnosis?.top_predictions || predictions.slice(0, 3)
  const confidence = result?.confidence || 0
  const isLowConfidence = confidence < CONFIDENCE_LOW
  const qualityOk = result?.diagnosis?.quality_ok ?? true
  const qualityIssue = result?.diagnosis?.quality_issue

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
            <select value={language} onChange={(event) => { stopSpeaking(); setLanguage(event.target.value) }}>
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
            <h2>Let's take<br />a closer look.</h2>
            <p>Start with one leaf. A clear, well-lit photo helps us give you a more useful first read.</p>
            <div className="photo-tips"><span>PHOTO GUIDE</span><p>Use natural light · Keep the leaf in focus · Include the affected area</p></div>
          </div>

          <div className="diagnosis-panel">
            <div className="panel-heading"><span>NEW DIAGNOSIS</span><span>FIELDNOTE / 001</span></div>
            <button className={`upload-area ${image ? "has-image" : ""}`} onClick={() => fileInput.current?.click()} type="button">
              {image ? <img src={image} alt="Selected crop leaf" /> : <span className="upload-prompt"><span className="upload-mark">＋</span><strong>Choose a leaf photo</strong><span>JPG or PNG · Tap to browse or use camera</span></span>}
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
              <label>ADVICE LANGUAGE<select value={language} onChange={(event) => { stopSpeaking(); setLanguage(event.target.value) }}>{LANGUAGES.map((item) => <option key={item.code} value={item.code}>{item.name}</option>)}</select></label>
            </div>
            <button className="button button-primary diagnose-button" onClick={submit} disabled={!file || loading}>
              {loading ? "READING YOUR PHOTO…" : "ANALYZE PHOTO"}<span aria-hidden="true">↗</span>
            </button>
            {loading && slowLoading && <p className="notice notice-slow" role="status">{t("waking_up")}</p>}
            <p className="privacy-note">Your photo is used to prepare this diagnosis. Predictions are informational, not guaranteed diagnoses.</p>

            {error && <div className="notice notice-error" role="alert">{error}</div>}

            {result && (
              <div className="result-card" aria-live="polite">
                <div className="result-topline">
                  <span>FIELD NOTE / {result.case_id.slice(0, 6).toUpperCase()}</span>
                  <span className={result.safe ? "status-safe" : "status-review"}>{result.safe ? "PRELIMINARY" : "REVIEW ADVISED"}</span>
                </div>
                <h3>What we found</h3>

                {/* Quality check */}
                {!qualityOk && qualityIssue && (
                  <div className="notice notice-warning" role="alert">
                    <strong>Image quality issue:</strong> {qualityIssue}
                  </div>
                )}

                {/* Low confidence warning */}
                {isLowConfidence && (
                  <div className="notice notice-warning" role="alert">
                    <strong>Low confidence:</strong> The model is not confident in this prediction.
                    Consider uploading a clearer photo or consulting an agricultural expert.
                  </div>
                )}

                {/* Uncalibrated confidence disclaimer */}
                {!result.confidence_is_calibrated && (
                  <p className="result-muted" style={{ fontSize: "0.8rem", marginTop: "4px" }}>
                    Confidence scores are uncalibrated model outputs, not disease probabilities.
                  </p>
                )}

                {predictions.length > 0 ? (
                  <>
                    <ul className="prediction-list">
                      {topPredictions.map((prediction, idx) => (
                        <li key={prediction.label + idx}>
                          <span>{idx === 0 ? "★ " : ""}{prediction.label.replaceAll("_", " ")}</span>
                          <strong>{Math.round(prediction.score * 100)}%</strong>
                        </li>
                      ))}
                    </ul>
                    {predictions.length > 3 && (
                      <details className="result-muted" style={{ marginTop: "8px" }}>
                        <summary style={{ cursor: "pointer", fontSize: "0.85rem" }}>Show all {predictions.length} predictions</summary>
                        <ul className="prediction-list" style={{ marginTop: "4px" }}>
                          {predictions.map((prediction, idx) => (
                            <li key={prediction.label + "_all_" + idx}>
                              <span>{prediction.label.replaceAll("_", " ")}</span>
                              <strong>{Math.round(prediction.score * 100)}%</strong>
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                  </>
                ) : (
                  <p className="result-muted">{result.diagnosis.quality_issue || "No clear result yet. Try a closer, brighter photo."}</p>
                )}

                <div className="result-advice"><p className="eyebrow"><span /> NEXT STEP</p><p>{result.advice}</p></div>

                {result.citations.length > 0 && <p className="citation-line">GUIDANCE: {result.citations.join(" · ")}</p>}

                {/* Guidance availability */}
                {!result.diagnosis.guidance_available && (
                  <p className="result-muted" style={{ fontSize: "0.85rem", marginTop: "4px" }}>
                    No specific treatment guidance is available for this class. Consult a local agricultural officer.
                  </p>
                )}

                {/* TTS Controls */}
                <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", marginTop: "12px" }}>
                  {audioUnavailable ? (
                    <button className="button button-outline" disabled>{t("audio_unavailable")}</button>
                  ) : (
                    <>
                      <button className="button button-outline" onClick={speak} disabled={speaking && !paused}>
                        {speaking ? (paused ? "RESUME" : "SPEAKING…") : "LISTEN TO ADVICE"} <span aria-hidden="true">↗</span>
                      </button>
                      {speaking && (
                        <button className="button button-outline" onClick={pauseResume}>
                          {paused ? "RESUME" : "PAUSE"}
                        </button>
                      )}
                      {speaking && (
                        <button className="button button-outline" onClick={stopSpeaking}>
                          STOP
                        </button>
                      )}
                      {!speaking && result.advice && (
                        <button className="button button-outline" onClick={speak}>
                          REPLAY <span aria-hidden="true">↻</span>
                        </button>
                      )}
                    </>
                  )}
                </div>

                {/* Voice availability hint */}
                {voiceAvailable === false && !audioUnavailable && (
                  <p className="result-muted" style={{ fontSize: "0.8rem", marginTop: "4px" }}>
                    No {LANGUAGES.find(l => l.code === language)?.name} voice found on this device.
                    Speech will use the server if available.
                  </p>
                )}
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
