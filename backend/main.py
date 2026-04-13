"""
LLM Evaluation & Benchmarking Dashboard — FastAPI Backend
GPU-Optimised: Intel i5 13th Gen + RTX 3050 6GB + 16GB RAM

Pipelines:
  base    → zero-shot via Ollama
  peft    → LoRA fine-tuned GPT-2 Medium (loads adapter from lora_adapter/)
  agentic → Real ReAct agent with tool execution on inventory/supplier CSVs
"""

import asyncio
import json
import time
import os
import sys
import httpx
from pathlib import Path
from typing import Optional

import psutil
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

LORA_AVAILABLE = Path("lora_adapter/adapter_config.json").exists()
if LORA_AVAILABLE:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from react_agent import run_react_agent, SUPPLY_CHAIN_SCENARIO

OLLAMA_BASE  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME   = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
GPU_LAYERS   = int(os.getenv("GPU_LAYERS", "35"))
CPU_THREADS  = int(os.getenv("CPU_THREADS", "12"))

REFERENCE_ANSWER = (
    "To optimise the supply chain: implement ABC-XYZ inventory classification, "
    "reduce lead times via dual-sourcing and supplier SLAs, apply statistical "
    "safety stock formula SS=Z*sigma*sqrt(LT), use EOQ for order sizing, "
    "target stockout below 5%, OTIF above 92%, inventory turnover above 10x."
)

SUPPLY_CHAIN_PROMPT = (
    "A logistics company faces: 35% stockout rate, 28-day avg lead time, "
    "$2.1M/yr carrying costs, 67% on-time delivery rate. "
    "Provide a structured improvement plan covering inventory, procurement, "
    "distribution, and KPIs. Be concise and specific."
)

_lora_model     = None
_lora_tokenizer = None

app = FastAPI(title="LLM Benchmark API — GPU + LoRA + ReAct", version="4.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def get_gpu_stats() -> dict:
    if not GPUTIL_AVAILABLE:
        return {"available": False, "gpu_percent": 0, "vram_used_mb": 0,
                "vram_total_mb": 6144, "vram_percent": 0,
                "gpu_name": "RTX 3050 6GB", "temperature": 0}
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return {"available": False, "gpu_percent": 0, "vram_used_mb": 0,
                    "vram_total_mb": 6144, "vram_percent": 0,
                    "gpu_name": "RTX 3050 6GB", "temperature": 0}
        g = gpus[0]
        return {
            "available":    True,
            "gpu_percent":  round(g.load * 100, 1),
            "vram_used_mb": round(g.memoryUsed, 0),
            "vram_total_mb":round(g.memoryTotal, 0),
            "vram_percent": round(g.memoryUtil * 100, 1),
            "gpu_name":     g.name,
            "temperature":  round(g.temperature, 0),
        }
    except Exception:
        return {"available": False, "gpu_percent": 0, "vram_used_mb": 0,
                "vram_total_mb": 6144, "vram_percent": 0,
                "gpu_name": "RTX 3050 6GB", "temperature": 0}


def get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1_048_576


def get_cpu_percent() -> float:
    return psutil.cpu_percent(interval=0.1)


def cosine_sim(a: str, b: str) -> float:
    if not SKLEARN_AVAILABLE or not a.strip() or not b.strip():
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return len(sa & sb) / max(len(sa | sb), 1)
    try:
        vec = TfidfVectorizer().fit_transform([a, b])
        return float(cosine_similarity(vec[0], vec[1])[0][0])
    except Exception:
        return 0.0


async def ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def list_ollama_models() -> list:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


async def run_base_pipeline(temperature: float, top_p: float, max_tokens: int) -> dict:
    ram_before = get_ram_mb()
    t0 = time.perf_counter()
    payload = {
        "model":  MODEL_NAME,
        "prompt": SUPPLY_CHAIN_PROMPT,
        "stream": False,
        "options": {
            "temperature":  temperature,
            "top_p":        top_p,
            "num_predict":  max_tokens,
            "num_thread":   CPU_THREADS,
            "num_gpu":      GPU_LAYERS,
            "main_gpu":     0,
            "f16_kv":       True,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{OLLAMA_BASE}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        elapsed = time.perf_counter() - t0
        text    = data.get("response", "").strip()
        n_eval  = data.get("eval_count", 1)
        tps     = n_eval / elapsed if elapsed > 0 else 0
        return {
            "text":           text,
            "tokens_per_sec": round(tps, 2),
            "ram_delta_mb":   round(get_ram_mb() - ram_before, 1),
            "elapsed_s":      round(elapsed, 2),
            "similarity":     round(cosine_sim(text, REFERENCE_ANSWER) * 100, 1),
            "gpu":            get_gpu_stats(),
            "demo":           False,
        }
    except Exception as e:
        return {
            "text":           f"Ollama error: {e}",
            "tokens_per_sec": 0,
            "ram_delta_mb":   0,
            "elapsed_s":      round(time.perf_counter() - t0, 2),
            "similarity":     0,
            "gpu":            get_gpu_stats(),
            "demo":           True,
        }


async def run_lora_pipeline(temperature: float, top_p: float, max_tokens: int) -> dict:
    global _lora_model, _lora_tokenizer

    ram_before = get_ram_mb()
    t0 = time.perf_counter()

    if not LORA_AVAILABLE:
        return {
            "text":           "LoRA adapter not found. Run train_lora.py first.",
            "tokens_per_sec": 0,
            "ram_delta_mb":   0,
            "elapsed_s":      0,
            "similarity":     0,
            "gpu":            get_gpu_stats(),
            "demo":           False,
        }

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if _lora_model is None:
            _lora_tokenizer = AutoTokenizer.from_pretrained("lora_adapter")
            _lora_tokenizer.pad_token = _lora_tokenizer.eos_token
            base = AutoModelForCausalLM.from_pretrained(
                "gpt2-medium",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            _lora_model = PeftModel.from_pretrained(base, "lora_adapter")
            _lora_model.eval()

        formatted = (
            f"### Instruction:\n{SUPPLY_CHAIN_PROMPT}\n\n"
            f"### Response:\n"
        )
        inputs = _lora_tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=300,
        ).to(device)

        with torch.no_grad():
            output = _lora_model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 400),
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.15,
                pad_token_id=_lora_tokenizer.eos_token_id,
                eos_token_id=_lora_tokenizer.eos_token_id,
            )

        text = _lora_tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        elapsed = time.perf_counter() - t0
        token_count = output.shape[1] - inputs["input_ids"].shape[1]
        tps = token_count / elapsed if elapsed > 0 else 0

        return {
            "text":           text,
            "tokens_per_sec": round(tps, 2),
            "ram_delta_mb":   round(get_ram_mb() - ram_before, 1),
            "elapsed_s":      round(elapsed, 2),
            "similarity":     round(cosine_sim(text, REFERENCE_ANSWER) * 100, 1),
            "gpu":            get_gpu_stats(),
            "demo":           False,
        }
    except Exception as e:
        return {
            "text":           f"LoRA inference error: {e}",
            "tokens_per_sec": 0,
            "ram_delta_mb":   round(get_ram_mb() - ram_before, 1),
            "elapsed_s":      round(time.perf_counter() - t0, 2),
            "similarity":     0,
            "gpu":            get_gpu_stats(),
            "demo":           False,
        }


async def run_agentic_pipeline(temperature: float, top_p: float, max_tokens: int) -> dict:
    ram_before = get_ram_mb()
    t0 = time.perf_counter()

    loop = asyncio.get_event_loop()
    final_answer = await loop.run_in_executor(
        None, lambda: run_react_agent(SUPPLY_CHAIN_SCENARIO, verbose=False)
    )

    if not final_answer:
        final_answer = "ReAct agent did not reach a Final Answer within step limit."

    elapsed = time.perf_counter() - t0
    words   = len(final_answer.split())
    tps     = round(words / elapsed * 1.3, 2) if elapsed > 0 else 0

    return {
        "text":           final_answer,
        "tokens_per_sec": tps,
        "ram_delta_mb":   round(get_ram_mb() - ram_before, 1),
        "elapsed_s":      round(elapsed, 2),
        "similarity":     round(cosine_sim(final_answer, REFERENCE_ANSWER) * 100, 1),
        "gpu":            get_gpu_stats(),
        "demo":           False,
    }


class BenchmarkRequest(BaseModel):
    temperature: float     = 0.7
    top_p:       float     = 0.9
    max_tokens:  int       = 512
    pipelines:   list[str] = ["base", "peft", "agentic"]


@app.post("/benchmark")
async def benchmark(req: BenchmarkRequest):
    results = {}
    for pipeline in req.pipelines:
        if pipeline == "base":
            results[pipeline] = await run_base_pipeline(req.temperature, req.top_p, req.max_tokens)
        elif pipeline == "peft":
            results[pipeline] = await run_lora_pipeline(req.temperature, req.top_p, req.max_tokens)
        elif pipeline == "agentic":
            results[pipeline] = await run_agentic_pipeline(req.temperature, req.top_p, req.max_tokens)

    vm = psutil.virtual_memory()
    return {
        "status":  "ok",
        "model":   f"base:{MODEL_NAME} | peft:gpt2-medium+lora | agentic:react+tools",
        "results": results,
        "system":  {
            "cpu_percent":  get_cpu_percent(),
            "ram_percent":  vm.percent,
            "ram_total_gb": round(vm.total / 1e9, 1),
            "gpu":          get_gpu_stats(),
        },
    }


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        raw         = await websocket.receive_text()
        req         = json.loads(raw)
        temperature = req.get("temperature", 0.7)
        top_p       = req.get("top_p", 0.9)
        max_tokens  = req.get("max_tokens", 512)
        pipeline    = req.get("pipeline", "agentic")

        await websocket.send_json({"type": "start", "pipeline": pipeline})

        if pipeline == "peft":
            result = await run_lora_pipeline(temperature, top_p, max_tokens)
            for word in result["text"].split(" "):
                await websocket.send_json({"type": "token", "token": word + " "})
                await asyncio.sleep(0.02)

        elif pipeline == "agentic":
            from react_agent import (
                TOOLS, parse_action, call_tool, query_ollama
            )
            context = [SUPPLY_CHAIN_SCENARIO]
            for step in range(8):
                model_output = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: query_ollama(context)
                )
                if not model_output:
                    break
                for word in model_output.split(" "):
                    await websocket.send_json({"type": "token", "token": word + " "})
                    await asyncio.sleep(0.01)
                if "Final Answer:" in model_output:
                    break
                tool_name, arg = parse_action(model_output)
                if tool_name:
                    observation = call_tool(tool_name, arg)
                    obs_msg = f"\n\nObservation:\n{observation}\n\n"
                    await websocket.send_json({"type": "token", "token": obs_msg})
                    context.append(model_output)
                    context.append(f"Observation: {observation}")
                else:
                    context.append(model_output)

        else:
            ollama_ok = await ollama_available()
            if not ollama_ok:
                await websocket.send_json({"type": "token", "token": "Ollama not available."})
            else:
                payload = {
                    "model": MODEL_NAME, "prompt": SUPPLY_CHAIN_PROMPT, "stream": True,
                    "options": {
                        "temperature": temperature, "top_p": top_p,
                        "num_predict": max_tokens, "num_thread": CPU_THREADS,
                        "num_gpu": GPU_LAYERS, "main_gpu": 0, "f16_kv": True,
                    },
                }
                async with httpx.AsyncClient(timeout=300) as client:
                    async with client.stream("POST", f"{OLLAMA_BASE}/api/generate", json=payload) as resp:
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                chunk = json.loads(line)
                                token = chunk.get("response", "")
                                if token:
                                    await websocket.send_json({"type": "token", "token": token})
                                if chunk.get("done"):
                                    break
                            except json.JSONDecodeError:
                                continue

        vm = psutil.virtual_memory()
        await websocket.send_json({
            "type":        "done",
            "cpu_percent": get_cpu_percent(),
            "ram_percent": vm.percent,
            "gpu":         get_gpu_stats(),
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/hw")
def hardware():
    vm = psutil.virtual_memory()
    return {
        "cpu_total":    psutil.cpu_percent(),
        "cpu_per_core": psutil.cpu_percent(percpu=True),
        "ram_used_gb":  round(vm.used / 1e9, 2),
        "ram_total_gb": round(vm.total / 1e9, 1),
        "ram_percent":  vm.percent,
        "gpu":          get_gpu_stats(),
    }


@app.get("/health")
async def health():
    ollama_ok = await ollama_available()
    models    = await list_ollama_models() if ollama_ok else []
    return {
        "status":           "ok",
        "ollama":           ollama_ok,
        "active_model":     MODEL_NAME,
        "available_models": models,
        "lora_adapter":     LORA_AVAILABLE,
        "react_tools":      list(["check_stockout_risk", "analyse_suppliers",
                                  "calculate_safety_stock", "calculate_eoq",
                                  "get_carrying_cost", "get_reorder_alerts"]),
        "gpu_layers":       GPU_LAYERS,
        "gpu":              get_gpu_stats(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
