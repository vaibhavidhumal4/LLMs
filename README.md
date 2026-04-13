# LLM Evaluation & Benchmarking Dashboard
### M.Tech Final Year Project — Intel i5 13th Gen / RTX 3050 6 GB / 16 GB RAM

```
llm-bench/
├── backend/
│   ├── main.py             ← FastAPI server (GPU-optimised)
│   ├── model_engine.py     ← RockFlow MIDI + TF model (CUDA-aware)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx         ← React dashboard (GPU monitor panel added)
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## Prerequisites (Windows — one-time setup)

### 1. Install CUDA Toolkit (for RTX 3050)
Download CUDA 12.x from https://developer.nvidia.com/cuda-downloads  
Choose: Windows → x86_64 → 11 → exe (local)

### 2. Install Ollama (handles GPU offloading automatically)
```
winget install Ollama.Ollama
```
Or download from https://ollama.com/download/windows

Ollama auto-detects your RTX 3050 via CUDA — no extra config needed.

### 3. Pull a model
```bash
# Recommended for 6 GB VRAM — fits entirely on GPU:
ollama pull llama3.2:3b

# Optional — also fits in 6 GB VRAM (Q4_K_M ~4.7 GB):
ollama pull llama3:8b
```

---

## Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Verify GPU is visible
python -c "import GPUtil; [print(g.name) for g in GPUtil.getGPUs()]"

# Start the backend
python main.py
# or:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend runs at **http://localhost:8000**

### GPU tuning environment variables (optional)
```bash
# Default is 35 layers on GPU (fits llama3.2:3b fully)
set GPU_LAYERS=35

# For llama3:8b on 6 GB VRAM use 28-32 layers
set GPU_LAYERS=28

# Use 12 CPU threads (i5-13th has 16 logical cores)
set CPU_THREADS=12

# Switch model
set OLLAMA_MODEL=llama3:8b
```

---

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Dashboard at **http://localhost:5173**

---

## Key API endpoints

| Method | Path           | Description                              |
|--------|----------------|------------------------------------------|
| POST   | `/benchmark`   | Run all pipelines, return metrics        |
| GET    | `/hw`          | CPU, RAM + GPU stats (polled every 2 s)  |
| WS     | `/ws/stream`   | Streaming token-by-token thought process |
| GET    | `/health`      | Ollama status + GPU layer config         |

---

## GPU Optimisations applied (vs original Mac build)

| Setting         | Mac (original)     | This build (RTX 3050)        | Rationale                          |
|-----------------|--------------------|------------------------------|------------------------------------|
| `num_gpu`       | 0 (CPU only)       | 35 (full model on VRAM)      | RTX 3050 has 6 GB — fits 3B model  |
| `main_gpu`      | N/A                | 0                            | Single GPU system                  |
| `f16_kv`        | disabled           | enabled                      | FP16 KV cache 2x faster on CUDA   |
| `num_thread`    | 8 (i9 cores)       | 12 (i5-13th P+E cores)       | i5-13600K has 14 cores / 20 threads|
| `low_vram`      | N/A                | False                        | 6 GB sufficient for 3B model      |
| `LLAMA_METAL`   | off (Mac CPU)      | N/A (Ollama handles CUDA)    | Ollama auto-selects CUDA backend   |
| Expected TPS    | ~18-22 tok/s       | ~40-55 tok/s                 | ~2-3x speedup from GPU offload    |

---

## Hardware Monitor (new panel)
The dashboard now shows a **GPU monitor** alongside CPU/RAM:
- GPU utilisation % (live gauge)
- VRAM used / 6144 MB (live bar)
- GPU temperature (°C)
- Top-bar quick stats: GPU% and VRAM MB

GPU stats use the `GPUtil` library. If it shows "GPUtil not detected", run:
```bash
pip install GPUtil
```

---

## Demo mode
If Ollama is not running or the model is missing, the backend returns
realistic pre-scripted answers with simulated GPU-speed TPS numbers
(~40-48 tok/s) so the entire dashboard is functional for demos.

---

## RockFlow-DL (Streamlit app)
```bash
# From backend/ directory
pip install streamlit music21
streamlit run app.py
```
The TensorFlow model in `model_engine.py` will automatically use CUDA
on RTX 3050 if TensorFlow-GPU is installed:
```bash
pip install tensorflow[and-cuda]
```
