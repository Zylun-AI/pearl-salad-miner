import json
import os
import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .model_zyi import ZYIEngine


HERE = Path(__file__).resolve().parent
CFG_PATH = HERE / "config.json"

with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = json.load(f)

OUTPUT_DIR = Path(
    os.environ.get(
        "ZYI_OUTPUT_DIR",
        "/app/outputs",
    )
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


app = FastAPI(
    title="ZYI 1.6 Base 180M API",
    version="beta",
)


# ============================================================
# REQUEST
# ============================================================

class GenerateRequest(BaseModel):

    prompt: str = Field(
        min_length=1,
        max_length=int(
            CFG.get(
                "max_prompt_chars",
                500,
            )
        ),
    )

    steps: int | None = None

    cfg: float | None = None

    seed: int | None = None


# ============================================================
# STATE
# ============================================================

jobs = OrderedDict()

jobs_lock = threading.Lock()

engine_lock = threading.Lock()

engine = None

last_submit = {}


# ============================================================
# TOKEN
# ============================================================

def require_proxy_token(
    provided: str | None,
):

    expected = os.environ.get(
        "ZYI_PROXY_TOKEN",
        "",
    ).strip()

    if not expected:
        return

    if not secrets.compare_digest(
        provided or "",
        expected,
    ):

        raise HTTPException(
            status_code=401,
            detail="Proxy token invalido.",
        )


# ============================================================
# ENGINE
# ============================================================

def get_engine():

    global engine

    with engine_lock:

        if engine is None:

            print(
                "[API] Carregando ZYI..."
            )

            engine = ZYIEngine(
                CFG_PATH
            )

            print(
                "[API] ZYI carregado."
            )

    return engine


# ============================================================
# LIMPEZA DE IMAGENS
# ============================================================

def trim_outputs():

    keep = int(
        CFG.get(
            "output_keep",
            30,
        )
    )

    files = sorted(
        OUTPUT_DIR.glob("*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for path in files[keep:]:

        try:
            path.unlink()

        except Exception:
            pass


# ============================================================
# JOB
# ============================================================

def run_job(
    job_id,
    prompt,
    steps,
    cfg_scale,
    seed,
):

    with jobs_lock:

        jobs[job_id][
            "status"
        ] = "loading_model"

        jobs[job_id][
            "started_at"
        ] = time.time()

    try:

        eng = get_engine()

        with jobs_lock:

            jobs[job_id][
                "status"
            ] = "generating"

        output_path = (
            OUTPUT_DIR
            / f"{job_id}.png"
        )

        t0 = time.time()

        eng.generate(
            prompt=prompt,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            output_path=output_path,
        )

        elapsed = (
            time.time() - t0
        )

        with jobs_lock:

            jobs[job_id][
                "status"
            ] = "done"

            jobs[job_id][
                "finished_at"
            ] = time.time()

            jobs[job_id][
                "generation_seconds"
            ] = round(
                elapsed,
                2,
            )

            jobs[job_id][
                "image"
            ] = (
                f"/v1/jobs/"
                f"{job_id}/image"
            )

        trim_outputs()

    except Exception as exc:

        print(
            "[JOB ERROR]",
            repr(exc),
        )

        with jobs_lock:

            jobs[job_id][
                "status"
            ] = "error"

            jobs[job_id][
                "error"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            jobs[job_id][
                "finished_at"
            ] = time.time()


# ============================================================
# FILA
# ============================================================

def next_job():

    with jobs_lock:

        for job_id, job in jobs.items():

            if (
                job["status"]
                == "queued"
            ):

                job[
                    "status"
                ] = "starting"

                return (
                    job_id,
                    dict(job),
                )

    return None


def worker_loop():

    print(
        "[API] Worker iniciado."
    )

    while True:

        item = next_job()

        if item is None:

            time.sleep(
                0.25
            )

            continue

        job_id, job = item

        run_job(
            job_id,
            job["prompt"],
            job["steps"],
            job["cfg"],
            job["seed"],
        )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():

    threading.Thread(
        target=worker_loop,
        daemon=True,
        name="zyi-worker",
    ).start()

    print(
        "=" * 60
    )

    print(
        "ZYI 1.6 BASE 180M"
    )

    print(
        "Salad Cloud API"
    )

    print(
        "=" * 60
    )

    print(
        "CUDA:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

        mem = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
            / 1024**3
        )

        print(
            f"VRAM: {mem:.2f} GB"
        )

    print(
        "Proxy token:",
        (
            "ATIVO"
            if os.environ.get(
                "ZYI_PROXY_TOKEN"
            )
            else "DESATIVADO"
        ),
    )


# ============================================================
# HEALTH
#
# Sem token para o Salad conseguir verificar o container.
# ============================================================

@app.get("/health")
def health():

    gpu = None

    if torch.cuda.is_available():

        props = (
            torch.cuda
            .get_device_properties(0)
        )

        gpu = {
            "name": props.name,

            "vram_total_gb": round(
                props.total_memory
                / 1024**3,
                2,
            ),

            "vram_allocated_gb": round(
                torch.cuda
                .memory_allocated()
                / 1024**3,
                2,
            ),
        }

    with jobs_lock:

        queued = sum(
            1
            for j in jobs.values()
            if j["status"]
            in {
                "queued",
                "starting",
            }
        )

        busy = any(
            j["status"]
            in {
                "loading_model",
                "generating",
            }
            for j in jobs.values()
        )

    return {
        "ok": True,
        "model":
            "ZYI-1.6-BASE-180M-BETA",
        "cuda":
            torch.cuda.is_available(),
        "gpu":
            gpu,
        "engine_loaded":
            engine is not None,
        "busy":
            busy,
        "queued":
            queued,
    }


# ============================================================
# GENERATE
# ============================================================

@app.post("/v1/generate")
def generate(
    req: GenerateRequest,
    request: Request,
    x_zyi_proxy_token:
        str | None
        = Header(default=None),
):

    require_proxy_token(
        x_zyi_proxy_token
    )

    ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    now = time.time()

    rate = float(
        CFG.get(
            "rate_limit_seconds",
            5,
        )
    )

    previous = (
        last_submit.get(
            ip,
            0.0,
        )
    )

    if (
        now - previous
        < rate
    ):

        raise HTTPException(
            status_code=429,
            detail=(
                f"Espere "
                f"{rate:.0f}s."
            ),
        )

    prompt = (
        req.prompt
        .strip()
    )

    default_steps = int(
        os.environ.get(
            "ZYI_DEFAULT_STEPS",
            CFG.get(
                "default_steps",
                8,
            ),
        )
    )

    max_steps = int(
        CFG.get(
            "max_steps",
            20,
        )
    )

    steps = int(
        req.steps
        if req.steps is not None
        else default_steps
    )

    steps = max(
        1,
        min(
            steps,
            max_steps,
        ),
    )

    default_cfg = float(
        os.environ.get(
            "ZYI_DEFAULT_CFG",
            CFG.get(
                "default_cfg",
                1.0,
            ),
        )
    )

    cfg_scale = float(
        req.cfg
        if req.cfg is not None
        else default_cfg
    )

    cfg_scale = max(
        1.0,
        min(
            cfg_scale,
            float(
                CFG.get(
                    "max_cfg",
                    4.5,
                )
            ),
        ),
    )

    seed = int(
        req.seed
        if req.seed is not None
        else secrets.randbelow(
            2_147_483_647
        )
    )

    with jobs_lock:

        active = sum(
            1
            for j in jobs.values()
            if j["status"]
            in {
                "queued",
                "starting",
                "loading_model",
                "generating",
            }
        )

        max_queue = int(
            CFG.get(
                "max_queue",
                4,
            )
        )

        if active >= max_queue:

            raise HTTPException(
                status_code=503,
                detail=(
                    "Fila cheia. "
                    "Tente novamente."
                ),
            )

        job_id = (
            secrets.token_hex(
                10
            )
        )

        jobs[job_id] = {
            "job_id":
                job_id,

            "status":
                "queued",

            "prompt":
                prompt,

            "steps":
                steps,

            "cfg":
                cfg_scale,

            "seed":
                seed,

            "created_at":
                now,
        }

        while len(jobs) > 100:

            jobs.popitem(
                last=False
            )

    last_submit[ip] = now

    return {
        "job_id":
            job_id,

        "status":
            "queued",

        "seed":
            seed,
    }


# ============================================================
# STATUS
# ============================================================

@app.get(
    "/v1/jobs/{job_id}"
)
def job_status(
    job_id: str,
    x_zyi_proxy_token:
        str | None
        = Header(default=None),
):

    require_proxy_token(
        x_zyi_proxy_token
    )

    with jobs_lock:

        job = jobs.get(
            job_id
        )

        if job is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Job nao encontrado."
                ),
            )

        return dict(job)


# ============================================================
# IMAGE
# ============================================================

@app.get(
    "/v1/jobs/{job_id}/image"
)
def job_image(
    job_id: str,
    x_zyi_proxy_token:
        str | None
        = Header(default=None),
):

    require_proxy_token(
        x_zyi_proxy_token
    )

    with jobs_lock:

        job = jobs.get(
            job_id
        )

        if job is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Job nao encontrado."
                ),
            )

        if (
            job.get("status")
            != "done"
        ):

            raise HTTPException(
                status_code=409,
                detail=(
                    "Imagem ainda "
                    "nao esta pronta."
                ),
            )

    path = (
        OUTPUT_DIR
        / f"{job_id}.png"
    )

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Imagem nao encontrada."
            ),
        )

    return FileResponse(
        path,
        media_type="image/png",
        filename=(
            f"zyi-{job_id}.png"
        ),
    )
