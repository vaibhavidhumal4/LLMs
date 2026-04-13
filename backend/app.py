
import streamlit as st
import model_engine as engine
import os, time, base64
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RockFlow-DL | AI Composer",
    page_icon="🎸",
    layout="centered",
)
 
# ── Custom CSS: dark rock theme ───────────────────────────────────────────────
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Metal+Mania&family=Rajdhani:wght@400;600;700&display=swap');
 
/* Dark background */
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0a !important;
    color: #e8e8e8 !important;
}
[data-testid="stHeader"] { background: transparent !important; }
 
/* Animated gradient title */
.rock-title {
    font-family: 'Metal Mania', cursive;
    font-size: 3rem;
    background: linear-gradient(90deg, #ff3c00, #ff8800, #ff3c00);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: flame 2s ease infinite;
    text-align: center;
    margin-bottom: 0;
}
@keyframes flame {
    0%   { background-position: 0% }
    50%  { background-position: 100% }
    100% { background-position: 0% }
}
.rock-sub {
    font-family: 'Rajdhani', sans-serif;
    text-align: center;
    color: #888;
    font-size: 1rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0;
}
.divider {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #ff3c00, transparent);
    margin: 1.5rem 0;
}
 
/* Cards */
.info-card {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #ff3c00;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-family: 'Rajdhani', sans-serif;
}
.info-card b { color: #ff6a00; }
 
/* Generate button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #ff3c00, #cc2200) !important;
    color: white !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.7rem 2.5rem !important;
    width: 100%;
    transition: all 0.2s ease;
    box-shadow: 0 0 20px rgba(255, 60, 0, 0.4);
}
div[data-testid="stButton"] > button:hover {
    box-shadow: 0 0 35px rgba(255, 60, 0, 0.8) !important;
    transform: translateY(-2px);
}
 
/* Slider */
[data-testid="stSlider"] > div > div > div { background: #ff3c00 !important; }
 
/* Audio player */
audio {
    width: 100%;
    border-radius: 8px;
    background: #1a1a1a;
    margin-top: 0.5rem;
}
 
/* Expander */
[data-testid="stExpander"] {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
}
 
/* Success box */
[data-testid="stAlert"] { border-left: 4px solid #ff3c00 !important; background: #1a0a00 !important; }
 
/* Metrics */
[data-testid="stMetric"] {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 0.8rem;
}
[data-testid="stMetricValue"] { color: #ff6a00 !important; font-family: 'Rajdhani', sans-serif !important; }
</style>
""")
 
# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="rock-title">🎸 RockFlow-DL</p>', unsafe_allow_html=True)
st.markdown('<p class="rock-sub">Deep Learning · AI Rock Composer · M.Tech GenAI Project</p>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)
 
# ── Syllabus integration accordion ────────────────────────────────────────────
with st.expander("📚 Syllabus Integration Details", expanded=False):
    st.markdown("""
    <div class='info-card'>
      <b>Topic 1 — Feedforward Networks:</b> Dense layers map extracted features to note probabilities.
    </div>
    <div class='info-card'>
      <b>Topic 2 — Convolutional Networks (CNN 1D):</b> Conv1D layers scan note sequences to detect rhythmic motifs and melodic patterns.
    </div>
    <div class='info-card'>
      <b>Topic 3 — Recurrent Networks (LSTM):</b> Stacked LSTMs model long-range temporal dependencies in guitar and bass sequences.
    </div>
    <div class='info-card'>
      <b>Topic 4 — Optimization & Regularization:</b> BatchNormalization stabilizes training; Dropout (0.3) prevents overfitting on music patterns.
    </div>
    <div class='info-card'>
      <b>Topic 5 — Deployment Pipeline:</b> End-to-end pipeline: architecture → generation → MIDI synthesis → WAV render → Streamlit UI.
    </div>
    """, unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
 
# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    bpm = st.slider("🥁 Tempo (BPM)", min_value=120, max_value=220, value=160, step=5,
                    help="Higher BPM = more aggressive feel")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    at_bpm = 4 * 30 / (bpm / 60)    # bars in 30 sec
    st.metric("⏱ Approx. Duration", f"{int(4*30/(bpm/60))} bars ≈ 30 sec")
 
st.markdown("<br>", unsafe_allow_html=True)
generate = st.button("⚡ GENERATE ROCK TRACK")
 
# ── Generation ────────────────────────────────────────────────────────────────
if generate:
    progress = st.progress(0, text="Initializing neural architecture...")
    time.sleep(0.4)
 
    progress.progress(20, text="Building CNN + LSTM model (Topics 2, 3, 4)...")
    model = engine.build_rock_model()
    time.sleep(0.5)
 
    progress.progress(45, text="Composing rock sections: Intro → Verse → Chorus → Bridge...")
    midi_path = engine.generate_midi(bpm=bpm)
    time.sleep(0.3)
 
    progress.progress(70, text="Rendering MIDI → WAV via FluidSynth...")
    wav_path = engine.midi_to_wav(midi_path)
    time.sleep(0.3)
 
    progress.progress(100, text="Done!")
    time.sleep(0.3)
    progress.empty()
 
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.success("🤘 Your AI rock track is ready — crank it up!")
 
    # ── In-browser audio player ───────────────────────────────────────────────
    if wav_path and os.path.exists(wav_path):
        st.markdown("### 🔊 Play Your Track")
        with open(wav_path, "rb") as wav_file:
            st.audio(wav_file.read(), format="audio/wav")
        # Also offer MIDI download
        with open(midi_path, "rb") as mid_file:
            st.download_button("📥 Download MIDI", mid_file, "ai_rock.mid",
                               mime="audio/midi", use_container_width=True)
    else:
        # FluidSynth not available → MIDI only
        st.warning("⚠️ WAV render unavailable (FluidSynth missing). Download MIDI and play in any DAW.")
        with open(midi_path, "rb") as mid_file:
            st.download_button("📥 Download Rock MIDI", mid_file, "ai_rock.mid",
                               mime="audio/midi", use_container_width=True)
 
    # ── Track info ────────────────────────────────────────────────────────────
    st.markdown("### 🎼 Track Structure")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Intro",   "8 bars")
    c2.metric("Verse",   "8 bars")
    c3.metric("Chorus",  "8 bars")
    c4.metric("Bridge",  "4 bars")
    c5.metric("Outro",   "2 bars")
 
    st.markdown(f"""
    <div class='info-card'>
      🎸 <b>Guitar:</b> Power chords, pentatonic minor riffs, alternate picking, palm-mute bridge<br>
      🎵 <b>Bass:</b> Root-note quarter groove locked to guitar sections<br>
      🥁 <b>BPM:</b> {bpm} | <b>Key:</b> E minor pentatonic | <b>Instruments:</b> Electric Guitar + Bass
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#444;font-size:0.8rem;">RockFlow-DL · M.Tech GenAI · CNN + LSTM + FluidSynth Pipeline</p>', unsafe_allow_html=True)
