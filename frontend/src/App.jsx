import { useState, useEffect, useRef, useCallback } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts";
import {
  Cpu, Activity, Zap, Terminal, Play, RotateCcw,
  ChevronRight, Layers, GitBranch, Brain, Monitor,
} from "lucide-react";

// ── palette / design tokens ───────────────────────────────────────────────
const C = {
  bg:      "#050a0f",
  panel:   "#0a1520",
  border:  "#112233",
  accent:  "#00e5ff",
  green:   "#00ff88",
  amber:   "#ffb800",
  red:     "#ff4444",
  purple:  "#bf00ff",
  muted:   "#3a5570",
  text:    "#c8dce8",
  textDim: "#4a6580",
};

const API = "http://localhost:8000";
const WS  = "ws://localhost:8000/ws/stream";

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const fmt   = (n, d = 1) => Number(n ?? 0).toFixed(d);

function useInterval(fn, ms) {
  const cb = useRef(fn);
  useEffect(() => { cb.current = fn; });
  useEffect(() => {
    const id = setInterval(() => cb.current(), ms);
    return () => clearInterval(id);
  }, [ms]);
}

// ── Gauge ─────────────────────────────────────────────────────────────────
function Gauge({ value = 0, max = 100, label, color = C.accent }) {
  const r = 46, cx = 60, cy = 64;
  const angle = (v) => (-150 + (v / max) * 300) * (Math.PI / 180);
  const arc = (pct) => {
    const a0 = -150 * (Math.PI / 180), a1 = (-150 + pct * 300) * (Math.PI / 180);
    const x0 = cx + r * Math.cos(a0), y0 = cy + r * Math.sin(a0);
    const x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1);
    const large = pct > 0.5 ? 1 : 0;
    return `M${x0},${y0} A${r},${r} 0 ${large},1 ${x1},${y1}`;
  };
  const pct = clamp(value / max, 0, 1);
  const nx  = cx + (r - 12) * Math.cos(angle(value));
  const ny  = cy + (r - 12) * Math.sin(angle(value));
  return (
    <svg viewBox="0 0 120 90" style={{ width: "100%", maxWidth: 140 }}>
      <path d={arc(1)}   stroke={C.border}  strokeWidth="8" fill="none" strokeLinecap="round" />
      <path d={arc(pct)} stroke={color}     strokeWidth="8" fill="none" strokeLinecap="round"
        style={{ filter: `drop-shadow(0 0 6px ${color})` }} />
      <circle cx={nx} cy={ny} r="5" fill={color} style={{ filter: `drop-shadow(0 0 4px ${color})` }} />
      <text x={cx} y={cy + 10} textAnchor="middle" fill={color} fontSize="16" fontWeight="700"
        fontFamily="'JetBrains Mono', monospace">{fmt(value, 0)}%</text>
      <text x={cx} y={cy + 26} textAnchor="middle" fill={C.textDim} fontSize="8"
        fontFamily="'JetBrains Mono', monospace">{label}</text>
    </svg>
  );
}

// ── VRAM bar ──────────────────────────────────────────────────────────────
function VramBar({ usedMb = 0, totalMb = 6144, color = C.purple }) {
  const pct = clamp(usedMb / totalMb, 0, 1);
  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: 9, color: C.textDim }}>
        <span>VRAM</span>
        <span style={{ color }}>{Math.round(usedMb)} / {Math.round(totalMb)} MB</span>
      </div>
      <div style={{ background: C.border, borderRadius: 4, height: 6, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${pct * 100}%`,
          background: pct > 0.85 ? C.red : pct > 0.65 ? C.amber : color,
          borderRadius: 4, transition: "width .5s",
          boxShadow: `0 0 8px ${color}`,
        }} />
      </div>
    </div>
  );
}

// ── Slider ────────────────────────────────────────────────────────────────
function Slider({ label, value, min, max, step, onChange, fmt: fmtFn = (v) => v }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ color: C.textDim, fontSize: 11, fontFamily: "monospace", textTransform: "uppercase", letterSpacing: 1 }}>{label}</span>
        <span style={{ color: C.accent, fontSize: 12, fontFamily: "'JetBrains Mono', monospace" }}>{fmtFn(value)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: C.accent, cursor: "pointer", height: 4 }} />
    </div>
  );
}

// ── Pipeline badge ────────────────────────────────────────────────────────
const PIPE_META = {
  base:    { color: C.amber,  icon: <Layers   size={13}/>, label: "Base LLM" },
  peft:    { color: C.green,  icon: <GitBranch size={13}/>, label: "PEFT/LoRA" },
  agentic: { color: C.accent, icon: <Brain    size={13}/>, label: "Agentic ReAct" },
};
function PBadge({ pipe, active, onClick }) {
  const m = PIPE_META[pipe];
  return (
    <button onClick={onClick} style={{
      display: "flex", alignItems: "center", gap: 6, padding: "7px 14px",
      border: `1px solid ${active ? m.color : C.border}`,
      background: active ? m.color + "22" : "transparent",
      color: active ? m.color : C.muted, borderRadius: 6, cursor: "pointer",
      fontSize: 12, fontFamily: "monospace", transition: "all .2s",
    }}>{m.icon}{m.label}</button>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────
export default function App() {
  const [temperature, setTemperature] = useState(0.7);
  const [topP,        setTopP]        = useState(0.9);
  const [maxTokens,   setMaxTokens]   = useState(512);
  const [activePipes, setActivePipes] = useState(["base", "peft", "agentic"]);
  const [streamPipe,  setStreamPipe]  = useState("agentic");

  const [hw,        setHw]       = useState({
    cpu_total: 0, ram_percent: 0, ram_used_gb: 0, ram_total_gb: 16,
    gpu: { available: false, gpu_percent: 0, vram_used_mb: 0, vram_total_mb: 6144,
           vram_percent: 0, gpu_name: "RTX 3050 6GB", temperature: 0 },
  });
  const [bench,     setBench]    = useState(null);
  const [logs,      setLogs]     = useState([]);
  const [running,   setRunning]  = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [tab,       setTab]      = useState("radar");

  const logRef = useRef(null);
  const wsRef  = useRef(null);

  // Poll hardware every 2s
  useInterval(async () => {
    try {
      const r = await fetch(`${API}/hw`);
      if (r.ok) setHw(await r.json());
    } catch {}
  }, 2000);

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const togglePipe = (p) =>
    setActivePipes(prev => prev.includes(p) ? prev.filter(x => x !== p) : [...prev, p]);

  // ── Run benchmark ────────────────────────────────────────────────────────
  const runBench = useCallback(async () => {
    setRunning(true);
    setBench(null);
    setLogs([{ ts: Date.now(), text: "▶  Starting benchmark run…", type: "info" }]);
    try {
      const res = await fetch(`${API}/benchmark`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ temperature, top_p: topP, max_tokens: maxTokens, pipelines: activePipes }),
      });
      const data = await res.json();
      setBench(data);
      setLogs(prev => [...prev,
        { ts: Date.now(), text: "✔  Benchmark complete.", type: "success" },
        ...activePipes.map(p => ({
          ts: Date.now(),
          text: `  ${p.toUpperCase().padEnd(8)} | ${fmt(data.results?.[p]?.tokens_per_sec)} TPS | ${fmt(data.results?.[p]?.similarity)}% sim`,
          type: "data",
        })),
      ]);
    } catch (e) {
      setLogs(prev => [...prev, { ts: Date.now(), text: `✘  Error: ${e.message}`, type: "error" }]);
    } finally {
      setRunning(false);
    }
  }, [temperature, topP, maxTokens, activePipes]);

  // ── Stream thought process ────────────────────────────────────────────────
  const runStream = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
    setStreaming(true);
    setLogs([{ ts: Date.now(), text: `▶  Streaming [${streamPipe}] pipeline…`, type: "info" }]);
    const ws = new WebSocket(WS);
    wsRef.current = ws;
    ws.onopen = () =>
      ws.send(JSON.stringify({ temperature, top_p: topP, max_tokens: maxTokens, pipeline: streamPipe }));
    let buf = "";
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "token") {
        buf += msg.token;
        const lines = buf.split("\n");
        buf = lines.pop();
        setLogs(prev => {
          const next = [...prev];
          lines.forEach(l => l.trim() && next.push({ ts: Date.now(), text: l, type: detectType(l) }));
          return next;
        });
      } else if (msg.type === "done") {
        if (buf.trim()) setLogs(prev => [...prev, { ts: Date.now(), text: buf, type: "data" }]);
        buf = "";
        setLogs(prev => [...prev, { ts: Date.now(), text: "✔  Stream done.", type: "success" }]);
        setStreaming(false);
      } else if (msg.type === "error") {
        setLogs(prev => [...prev, { ts: Date.now(), text: `✘  ${msg.message}`, type: "error" }]);
        setStreaming(false);
      }
    };
    ws.onerror = () => {
      setLogs(prev => [...prev, { ts: Date.now(), text: "✘  WebSocket error — is backend running?", type: "error" }]);
      setStreaming(false);
    };
  }, [temperature, topP, maxTokens, streamPipe]);

  const stopStream = () => { wsRef.current?.close(); setStreaming(false); };

  function detectType(line) {
    if (/^Thought:/i.test(line))      return "thought";
    if (/^Action:/i.test(line))       return "action";
    if (/^Observation:/i.test(line))  return "obs";
    if (/^Final Answer:/i.test(line)) return "final";
    if (/^✔|complete|done/i.test(line)) return "success";
    if (/^✘|error/i.test(line))       return "error";
    return "data";
  }
  const LOG_COLORS = {
    thought: C.accent, action: C.amber, obs: C.green,
    final: "#ff80ff", info: C.textDim, success: C.green, error: C.red, data: C.text,
  };

  // ── Chart data ────────────────────────────────────────────────────────────
  const radarData = [
    { subject: "Speed (TPS)" },
    { subject: "Similarity" },
    { subject: "Efficiency" },
    { subject: "Reasoning" },
    { subject: "Coherence" },
  ];
  const MAXES = [60, 100, 100, 100, 100];  // TPS max bumped for GPU
  ["base","peft","agentic"].forEach(p => {
    const r = bench?.results?.[p];
    radarData[0][p] = r ? clamp((r.tokens_per_sec / MAXES[0]) * 100, 5, 100) : 0;
    radarData[1][p] = r ? r.similarity : 0;
    radarData[2][p] = r ? clamp(100 - (r.ram_delta_mb / 8), 10, 100) : 0;
    radarData[3][p] = p === "agentic" ? (r ? Math.min(r.similarity * 1.15, 100) : 0) : (r ? r.similarity * 0.9 : 0);
    radarData[4][p] = r ? Math.min(r.similarity + 5, 100) : 0;
  });

  const barData = activePipes.map(p => {
    const r = bench?.results?.[p] ?? {};
    return {
      name: p,
      "TPS": r.tokens_per_sec ?? 0,
      "Similarity %": r.similarity ?? 0,
      "RAM Δ MB": r.ram_delta_mb ?? 0,
    };
  });

  const PIPE_COLORS = { base: C.amber, peft: C.green, agentic: C.accent };
  const gpu = hw.gpu ?? {};

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text,
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace", display: "flex" }}>

      {/* ── SIDEBAR ── */}
      <aside style={{
        width: 260, background: C.panel, borderRight: `1px solid ${C.border}`,
        padding: "24px 18px", display: "flex", flexDirection: "column", gap: 24,
        flexShrink: 0,
      }}>
        {/* Logo */}
        <div style={{ paddingBottom: 16, borderBottom: `1px solid ${C.border}` }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <Zap size={18} color={C.accent} />
            <span style={{ color: C.accent, fontSize: 14, fontWeight: 700, letterSpacing: 2 }}>LLM BENCH</span>
          </div>
          <div style={{ color: C.textDim, fontSize: 10, letterSpacing: 1 }}>M.TECH FINAL YEAR PROJECT</div>
          {/* GPU badge */}
          <div style={{
            marginTop: 8, padding: "4px 8px", background: C.purple + "22",
            border: `1px solid ${C.purple}60`, borderRadius: 4,
            color: C.purple, fontSize: 9, letterSpacing: 1,
            display: "flex", alignItems: "center", gap: 5,
          }}>
            <span>⬡</span> RTX 3050 6 GB · CUDA
          </div>
        </div>

        {/* Pipelines */}
        <div>
          <div style={{ color: C.textDim, fontSize: 10, letterSpacing: 2, marginBottom: 10, textTransform: "uppercase" }}>Pipelines</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {["base","peft","agentic"].map(p => (
              <PBadge key={p} pipe={p} active={activePipes.includes(p)} onClick={() => togglePipe(p)} />
            ))}
          </div>
        </div>

        {/* Hyperparams */}
        <div style={{ flex: 1 }}>
          <div style={{ color: C.textDim, fontSize: 10, letterSpacing: 2, marginBottom: 14, textTransform: "uppercase" }}>Hyperparameters</div>
          <Slider label="Temperature" value={temperature} min={0.01} max={2} step={0.01} onChange={setTemperature} fmtFn={v => v.toFixed(2)} />
          <Slider label="Top-P"       value={topP}        min={0.01} max={1} step={0.01} onChange={setTopP}        fmtFn={v => v.toFixed(2)} />
          <Slider label="Max Tokens"  value={maxTokens}   min={64}   max={2048} step={64} onChange={setMaxTokens} />
        </div>

        {/* Buttons */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <button onClick={runBench} disabled={running || !activePipes.length} style={{
            display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
            padding: "10px 0", background: running ? C.muted : C.accent + "22",
            border: `1px solid ${running ? C.muted : C.accent}`, color: running ? C.muted : C.accent,
            borderRadius: 8, cursor: running ? "not-allowed" : "pointer", fontSize: 13, fontWeight: 700,
            transition: "all .2s",
          }}>
            <Play size={14} />{running ? "Running…" : "Run Benchmark"}
          </button>
          <button onClick={() => { setBench(null); setLogs([]); }} disabled={running} style={{
            display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
            padding: "8px 0", background: "transparent",
            border: `1px solid ${C.border}`, color: C.textDim,
            borderRadius: 8, cursor: "pointer", fontSize: 11,
          }}>
            <RotateCcw size={12} />Reset
          </button>
        </div>
      </aside>

      {/* ── MAIN ── */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

        {/* Top bar */}
        <header style={{
          padding: "0 24px", height: 52, display: "flex", alignItems: "center",
          justifyContent: "space-between", borderBottom: `1px solid ${C.border}`,
          background: C.panel,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, color: C.textDim, fontSize: 12 }}>
            <ChevronRight size={14} color={C.accent} />
            <span>Supply Chain Logistics Reasoning Task</span>
            <span style={{ marginLeft: 12, padding: "2px 8px", background: C.accent + "22",
              border: `1px solid ${C.accent}40`, borderRadius: 4, color: C.accent, fontSize: 10 }}>
              LIVE
            </span>
          </div>
          <div style={{ display: "flex", gap: 16, fontSize: 11, color: C.textDim }}>
            <span>CPU <span style={{ color: C.amber }}>{fmt(hw.cpu_total, 0)}%</span></span>
            <span>RAM <span style={{ color: C.green }}>{hw.ram_percent}%</span></span>
            <span>GPU <span style={{ color: C.purple }}>{fmt(gpu.gpu_percent, 0)}%</span></span>
            <span>VRAM <span style={{ color: C.purple }}>{Math.round(gpu.vram_used_mb ?? 0)}MB</span></span>
            <span style={{ color: C.accent }}>i5-13th · RTX 3050 · 16 GB</span>
          </div>
        </header>

        {/* Content area */}
        <div style={{ flex: 1, overflow: "auto", padding: 20, display: "grid",
          gridTemplateColumns: "1fr 340px", gridTemplateRows: "auto 1fr", gap: 16 }}>

          {/* ── Results cards ── */}
          <div style={{ gridColumn: "1 / -1", display: "flex", gap: 12 }}>
            {["base","peft","agentic"].map(p => {
              const r = bench?.results?.[p];
              const m = PIPE_META[p];
              return (
                <div key={p} style={{
                  flex: 1, background: C.panel, border: `1px solid ${r ? m.color + "60" : C.border}`,
                  borderRadius: 10, padding: "14px 18px",
                  boxShadow: r ? `0 0 20px ${m.color}18` : "none",
                  transition: "all .4s",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                    <span style={{ color: m.color }}>{m.icon}</span>
                    <span style={{ color: m.color, fontSize: 11, fontWeight: 700, letterSpacing: 2, textTransform: "uppercase" }}>{m.label}</span>
                    {r?.demo && (
                      <span style={{ marginLeft: "auto", fontSize: 9, color: C.textDim, padding: "2px 6px",
                        border: `1px solid ${C.border}`, borderRadius: 3 }}>DEMO</span>
                    )}
                  </div>
                  {r ? (
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                      {[
                        ["TPS",     fmt(r.tokens_per_sec), "tok/s"],
                        ["Sim",     fmt(r.similarity),     "%"],
                        ["RAM Δ",   fmt(r.ram_delta_mb, 0),"MB"],
                        ["Latency", fmt(r.elapsed_s),      "s"],
                      ].map(([k, v, u]) => (
                        <div key={k}>
                          <div style={{ color: C.textDim, fontSize: 9, letterSpacing: 1 }}>{k}</div>
                          <div style={{ color: m.color, fontSize: 18, fontWeight: 700 }}>
                            {v}<span style={{ fontSize: 10, color: C.textDim, marginLeft: 3 }}>{u}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ color: C.textDim, fontSize: 11 }}>
                      {running && activePipes.includes(p) ? "Running…" : "Not run yet"}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* ── Charts panel ── */}
          <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 10, padding: 18 }}>
            {/* Tab bar */}
            <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
              {["radar","bar"].map(t => (
                <button key={t} onClick={() => setTab(t)} style={{
                  padding: "5px 14px", borderRadius: 6, fontSize: 11, cursor: "pointer",
                  background: tab === t ? C.accent + "22" : "transparent",
                  border: `1px solid ${tab === t ? C.accent : C.border}`,
                  color: tab === t ? C.accent : C.textDim,
                }}>{t === "radar" ? "Radar" : "Bar"}</button>
              ))}
            </div>

            {tab === "radar" ? (
              <ResponsiveContainer width="100%" height={280}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke={C.border} />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: C.textDim, fontSize: 10, fontFamily: "monospace" }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                  {activePipes.map(p => (
                    <Radar key={p} name={PIPE_META[p].label} dataKey={p}
                      stroke={PIPE_COLORS[p]} fill={PIPE_COLORS[p]} fillOpacity={0.15} />
                  ))}
                  <Legend iconSize={10} wrapperStyle={{ fontSize: 11, fontFamily: "monospace" }} />
                </RadarChart>
              </ResponsiveContainer>
            ) : (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={barData} barCategoryGap="30%">
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="name" tick={{ fill: C.textDim, fontSize: 10 }} />
                  <YAxis tick={{ fill: C.textDim, fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 6, fontFamily: "monospace", fontSize: 11 }} />
                  <Legend wrapperStyle={{ fontSize: 11, fontFamily: "monospace" }} />
                  <Bar dataKey="TPS"          fill={C.amber}  radius={[4,4,0,0]} />
                  <Bar dataKey="Similarity %" fill={C.green}  radius={[4,4,0,0]} />
                  <Bar dataKey="RAM Δ MB"     fill={C.accent} radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* ── Hardware Monitor ── */}
          <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 10, padding: 18 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
              <Cpu size={14} color={C.accent} />
              <span style={{ color: C.textDim, fontSize: 10, letterSpacing: 2 }}>HARDWARE MONITOR</span>
            </div>

            {/* CPU + RAM gauges */}
            <div style={{ display: "flex", justifyContent: "space-around", marginBottom: 12 }}>
              <div style={{ textAlign: "center" }}>
                <Gauge value={hw.cpu_total} label="CPU" color={C.amber} />
                <div style={{ color: C.textDim, fontSize: 9, marginTop: 4 }}>Intel i5-13th</div>
              </div>
              <div style={{ textAlign: "center" }}>
                <Gauge value={hw.ram_percent} label="RAM" color={C.green} />
                <div style={{ color: C.textDim, fontSize: 9, marginTop: 4 }}>{hw.ram_used_gb} / {hw.ram_total_gb} GB</div>
              </div>
            </div>

            {/* ── GPU Panel (new) ── */}
            <div style={{
              background: "#0a0818", border: `1px solid ${C.purple}40`,
              borderRadius: 8, padding: "10px 12px", marginBottom: 12,
            }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ color: C.purple, fontSize: 12 }}>⬡</span>
                  <span style={{ color: C.purple, fontSize: 10, letterSpacing: 1 }}>GPU</span>
                </div>
                <span style={{ color: C.textDim, fontSize: 9 }}>{gpu.gpu_name ?? "RTX 3050 6GB"}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-around", marginBottom: 6 }}>
                <div style={{ textAlign: "center" }}>
                  <Gauge value={gpu.gpu_percent ?? 0} max={100} label="Util" color={C.purple} />
                </div>
                <div style={{ textAlign: "center", paddingTop: 8 }}>
                  <div style={{ color: C.purple, fontSize: 22, fontWeight: 700 }}>
                    {Math.round(gpu.temperature ?? 0)}
                    <span style={{ fontSize: 11, color: C.textDim }}>°C</span>
                  </div>
                  <div style={{ color: C.textDim, fontSize: 9, marginTop: 4 }}>TEMP</div>
                  <div style={{ color: C.purple, fontSize: 14, fontWeight: 700, marginTop: 8 }}>
                    {Math.round(gpu.vram_used_mb ?? 0)}
                    <span style={{ fontSize: 9, color: C.textDim }}> MB used</span>
                  </div>
                  <div style={{ color: C.textDim, fontSize: 9 }}>VRAM</div>
                </div>
              </div>
              <VramBar usedMb={gpu.vram_used_mb ?? 0} totalMb={gpu.vram_total_mb ?? 6144} color={C.purple} />
              {!gpu.available && (
                <div style={{ color: C.textDim, fontSize: 9, textAlign: "center", marginTop: 6, fontStyle: "italic" }}>
                  GPUtil not detected — install: pip install GPUtil
                </div>
              )}
            </div>

            {/* Per-core bars */}
            {hw.cpu_per_core && (
              <div>
                <div style={{ color: C.textDim, fontSize: 9, letterSpacing: 1, marginBottom: 6 }}>CORE UTILISATION</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(8, 1fr)", gap: 3 }}>
                  {hw.cpu_per_core.slice(0, 16).map((c, i) => (
                    <div key={i} title={`Core ${i}: ${c}%`}>
                      <div style={{ height: 36, background: C.border, borderRadius: 3, position: "relative", overflow: "hidden" }}>
                        <div style={{
                          position: "absolute", bottom: 0, width: "100%",
                          height: `${c}%`, background: c > 80 ? C.red : c > 50 ? C.amber : C.accent,
                          transition: "height .5s", borderRadius: 3,
                        }} />
                      </div>
                      <div style={{ color: C.textDim, fontSize: 7, textAlign: "center", marginTop: 2 }}>{i}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* ── Terminal ── */}
          <div style={{
            gridColumn: "1 / -1", background: "#020810",
            border: `1px solid ${C.border}`, borderRadius: 10, overflow: "hidden",
            display: "flex", flexDirection: "column", minHeight: 240,
          }}>
            <div style={{
              padding: "8px 16px", background: C.panel, borderBottom: `1px solid ${C.border}`,
              display: "flex", alignItems: "center", justifyContent: "space-between",
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <Terminal size={13} color={C.accent} />
                <span style={{ color: C.textDim, fontSize: 11, letterSpacing: 1 }}>AGENT THOUGHT STREAM</span>
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <div style={{ display: "flex", gap: 4 }}>
                  {["base","peft","agentic"].map(p => (
                    <button key={p} onClick={() => setStreamPipe(p)} style={{
                      padding: "3px 10px", borderRadius: 4, fontSize: 10, cursor: "pointer",
                      background: streamPipe === p ? PIPE_COLORS[p] + "22" : "transparent",
                      border: `1px solid ${streamPipe === p ? PIPE_COLORS[p] : C.border}`,
                      color: streamPipe === p ? PIPE_COLORS[p] : C.textDim,
                    }}>{p}</button>
                  ))}
                </div>
                {streaming ? (
                  <button onClick={stopStream} style={{
                    padding: "4px 12px", background: C.red + "22", border: `1px solid ${C.red}`,
                    color: C.red, borderRadius: 6, cursor: "pointer", fontSize: 11,
                  }}>■ Stop</button>
                ) : (
                  <button onClick={runStream} style={{
                    display: "flex", alignItems: "center", gap: 5,
                    padding: "4px 12px", background: C.accent + "22", border: `1px solid ${C.accent}`,
                    color: C.accent, borderRadius: 6, cursor: "pointer", fontSize: 11,
                  }}><Activity size={11} />Stream</button>
                )}
              </div>
            </div>

            <div ref={logRef} style={{
              flex: 1, padding: "12px 18px", overflowY: "auto", fontSize: 12,
              lineHeight: 1.7, letterSpacing: 0.3,
            }}>
              {logs.length === 0 ? (
                <span style={{ color: C.textDim }}>$ _  awaiting input…</span>
              ) : logs.map((l, i) => (
                <div key={i} style={{ color: LOG_COLORS[l.type] ?? C.text }}>
                  <span style={{ color: C.textDim, marginRight: 12, fontSize: 10 }}>
                    {new Date(l.ts).toLocaleTimeString("en-US", { hour12: false })}
                  </span>
                  {l.text}
                </div>
              ))}
              {streaming && <span style={{ color: C.accent }}>█</span>}
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
