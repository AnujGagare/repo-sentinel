import { useState, useEffect, useRef, useCallback } from 'react'
import { fetchStatus, askQuestion, startIndexRepo, getIndexJobStatus } from './api.js'

const JOB_POLL_INTERVAL_MS = 2000

function RepoSwitcher({ onRepoReady }) {
  const [open, setOpen] = useState(false)
  const [url, setUrl] = useState('')
  const [job, setJob] = useState(null)
  const [error, setError] = useState(null)
  const pollRef = useRef(null)

  useEffect(() => () => clearInterval(pollRef.current), [])

  function pollJob(jobId) {
    pollRef.current = setInterval(async () => {
      try {
        const j = await getIndexJobStatus(jobId)
        setJob(j)
        if (j.status === 'ready') {
          clearInterval(pollRef.current)
          onRepoReady()
        } else if (j.status === 'failed') {
          clearInterval(pollRef.current)
          setError(j.error || 'indexing failed')
        }
      } catch (err) {
        clearInterval(pollRef.current)
        setError(err.message)
      }
    }, JOB_POLL_INTERVAL_MS)
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!url.trim()) return
    setError(null)
    setJob(null)
    try {
      const j = await startIndexRepo(url.trim())
      setJob(j)
      pollJob(j.job_id)
    } catch (err) {
      setError(err.message)
    }
  }

  function reset() {
    clearInterval(pollRef.current)
    setOpen(false)
    setUrl('')
    setJob(null)
    setError(null)
  }

  const busy = job && job.status !== 'ready' && job.status !== 'failed'

  if (!open) {
    return (
      <button className="repo-switch-trigger" onClick={() => setOpen(true)}>
        Index a different repo
      </button>
    )
  }

  return (
    <div className="repo-switch">
      <form className="repo-switch__form" onSubmit={handleSubmit}>
        <input
          className="repo-switch__input"
          type="text"
          placeholder="https://github.com/owner/repo"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={busy}
          autoComplete="off"
        />
        <button className="repo-switch__submit" type="submit" disabled={busy || !url.trim()}>
          Index
        </button>
        <button className="repo-switch__cancel" type="button" onClick={reset}>
          Cancel
        </button>
      </form>

      {job && (
        <div className="repo-switch__status">
          <span className={`repo-switch__status-dot ${busy ? 'repo-switch__status-dot--busy' : ''}`} />
          {job.status}
          {job.detail && <span className="repo-switch__status-detail"> &mdash; {job.detail}</span>}
        </div>
      )}
      {error && <div className="repo-switch__error">{error}</div>}
    </div>
  )
}

function StatusPill({ status, error }) {
  if (error) {
    return (
      <div className="status-pill status-pill--error">
        <span className="status-dot status-dot--error" />
        backend unreachable
      </div>
    )
  }
  if (!status) {
    return (
      <div className="status-pill">
        <span className="status-dot status-dot--pending" />
        checking index&hellip;
      </div>
    )
  }
  return (
    <div className={`status-pill ${status.is_stale ? 'status-pill--stale' : 'status-pill--live'}`}>
      <span className={`status-dot ${status.is_stale ? 'status-dot--stale' : 'status-dot--live'}`} />
      {status.is_stale ? 'index stale' : 'synced'}
      <span className="status-sep">&middot;</span>
      <span className="mono">{status.chunks_indexed}</span> chunks
      <span className="status-sep">&middot;</span>
      <span className="mono">{status.indexed_commit?.slice(0, 7)}</span>
    </div>
  )
}

function ElapsedTimer({ startedAt }) {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setElapsed(((Date.now() - startedAt) / 1000).toFixed(1)), 100)
    return () => clearInterval(id)
  }, [startedAt])
  return <span className="mono">{elapsed}s</span>
}

function SourceCard({ citation }) {
  const [expanded, setExpanded] = useState(false)
  const hasSnippet = Boolean(citation.source_snippet)

  return (
    <div className={`source-card ${expanded ? 'source-card--expanded' : ''}`}>
      <button
        className="source-card__toggle"
        onClick={() => setExpanded((e) => !e)}
        disabled={!hasSnippet}
        aria-expanded={expanded}
      >
        <div className="source-card__toggle-text">
          <div className="source-card__symbol">{citation.symbol_name}</div>
          <div className="source-card__location mono">
            {citation.file_path}:{citation.start_line}&ndash;{citation.end_line}
          </div>
        </div>
        {hasSnippet && <span className="source-card__chevron">{expanded ? '\u2212' : '+'}</span>}
      </button>
      {expanded && hasSnippet && (
        <pre className="source-card__snippet mono"><code>{citation.source_snippet}</code></pre>
      )}
    </div>
  )
}

function SourcesRail({ citations, latencyMs }) {
  if (!citations) {
    return (
      <aside className="sources-rail">
        <div className="sources-rail__header">Sources</div>
        <div className="sources-rail__empty">
          Citations for the assistant's most recent answer will appear here. Click one to see the actual retrieved code.
        </div>
      </aside>
    )
  }
  return (
    <aside className="sources-rail">
      <div className="sources-rail__header">
        Sources
        {latencyMs != null && <span className="sources-rail__latency mono">{latencyMs}ms</span>}
      </div>
      <div className="sources-rail__list">
        {citations.map((c, i) => (
          <SourceCard citation={c} key={i} />
        ))}
      </div>
    </aside>
  )
}

function ChatMessage({ message }) {
  if (message.role === 'user') {
    return (
      <div className="msg msg--user">
        <div className="msg__bubble">{message.text}</div>
      </div>
    )
  }
  if (message.role === 'error') {
    return (
      <div className="msg msg--assistant">
        <div className="msg__bubble msg__bubble--error">{message.text}</div>
      </div>
    )
  }
  if (message.role === 'pending') {
    return (
      <div className="msg msg--assistant">
        <div className="msg__bubble msg__bubble--pending">
          <span className="scan-dot" />
          retrieving &amp; generating&hellip;
          <ElapsedTimer startedAt={message.startedAt} />
        </div>
      </div>
    )
  }
  return (
    <div className="msg msg--assistant">
      <div className="msg__bubble">{message.text}</div>
      {message.indexedCommit && (
        <div className="msg__meta mono">indexed at commit {message.indexedCommit.slice(0, 7)}</div>
      )}
    </div>
  )
}

const SUGGESTED_QUESTIONS = [
  'How does FastAPI resolve dependencies declared with Depends()?',
  'How is an HTTPException converted into a JSON error response?',
  'What class represents a background task run after the response?',
  'How does FastAPI generate the OpenAPI schema for an app?',
]

export default function App() {
  const [status, setStatus] = useState(null)
  const [statusError, setStatusError] = useState(false)
  const [messages, setMessages] = useState([
    { role: 'assistant', text: "This assistant answers questions about the indexed FastAPI codebase by retrieving the actual source code, not guessing from memory. Every answer cites the exact file and line it came from — click a source on the right to see the code itself." },
  ])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [lastCitations, setLastCitations] = useState(null)
  const [lastLatency, setLastLatency] = useState(null)
  const scrollRef = useRef(null)

  const refreshStatus = useCallback(async () => {
    try {
      const s = await fetchStatus()
      setStatus(s)
      setStatusError(false)
    } catch {
      setStatusError(true)
    }
  }, [])

  useEffect(() => {
    refreshStatus()
    const id = setInterval(refreshStatus, 15000)
    return () => clearInterval(id)
  }, [refreshStatus])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  async function submitQuestion(question) {
    if (!question || busy) return

    setInput('')
    setBusy(true)
    setMessages((prev) => [
      ...prev,
      { role: 'user', text: question },
      { role: 'pending', startedAt: Date.now() },
    ])

    try {
      const data = await askQuestion(question)
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { role: 'assistant', text: data.answer, indexedCommit: data.indexed_commit },
      ])
      setLastCitations(data.citations)
      setLastLatency(data.latency_ms)
    } catch (err) {
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { role: 'error', text: `Error: ${err.message}` },
      ])
    } finally {
      setBusy(false)
      refreshStatus()
    }
  }

  function handleSubmit(e) {
    e.preventDefault()
    submitQuestion(input.trim())
  }

  const showSuggestions = messages.length === 1 && !busy

  function handleRepoReady() {
    refreshStatus()
    setLastCitations(null)
    setLastLatency(null)
    setMessages([
      { role: 'assistant', text: 'New repo indexed. Ask a question about it — every answer is grounded in the retrieved source, with citations shown on the right.' },
    ])
  }

  return (
    <div className="app">
      <header className="app__header">
        <div className="app__title">
          <span className="app__title-main">Repo Sentinel</span>
          <span className="app__title-sub">live-synced RAG over a codebase</span>
        </div>
        <div className="app__header-right">
          <RepoSwitcher onRepoReady={handleRepoReady} />
          <StatusPill status={status} error={statusError} />
        </div>
      </header>

      <div className="app__body">
        <main className="chat">
          <div className="chat__scroll" ref={scrollRef}>
            {messages.map((m, i) => (
              <ChatMessage message={m} key={i} />
            ))}

            {showSuggestions && (
              <div className="suggestions">
                <div className="suggestions__label">Try asking</div>
                <div className="suggestions__chips">
                  {SUGGESTED_QUESTIONS.map((q) => (
                    <button
                      key={q}
                      type="button"
                      className="suggestion-chip"
                      onClick={() => submitQuestion(q)}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          <form className="chat__form" onSubmit={handleSubmit}>
            <input
              className="chat__input"
              type="text"
              placeholder="Ask about the codebase&hellip;"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={busy}
              autoComplete="off"
            />
            <button className="chat__submit" type="submit" disabled={busy || !input.trim()}>
              Ask
            </button>
          </form>
        </main>

        <SourcesRail citations={lastCitations} latencyMs={lastLatency} />
      </div>
    </div>
  )
}
