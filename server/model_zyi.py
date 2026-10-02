import gc
import json
import math
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from PIL import Image

from diffusers import AutoencoderKL

from huggingface_hub import (
    hf_hub_download,
)

from transformers import (
    AutoTokenizer,
    T5EncoderModel,
)


# ============================================================
# RMS NORM
# ============================================================

class RMSNorm(nn.Module):

    def __init__(
        self,
        dim,
        eps=1e-6,
        affine=True,
    ):

        super().__init__()

        self.eps = eps

        self.weight = (
            nn.Parameter(
                torch.ones(dim)
            )
            if affine
            else None
        )

    def forward(
        self,
        x,
    ):

        out = (
            x
            * torch.rsqrt(
                x.pow(2)
                .mean(
                    -1,
                    keepdim=True,
                )
                + self.eps
            )
        )

        if self.weight is not None:

            out = (
                out
                * self.weight
            )

        return out


# ============================================================
# TIMESTEP
# ============================================================

class TimestepEmbedder(
    nn.Module
):

    def __init__(
        self,
        dim,
        freq_dim=256,
    ):

        super().__init__()

        self.freq_dim = (
            freq_dim
        )

        self.mlp = (
            nn.Sequential(
                nn.Linear(
                    freq_dim,
                    dim,
                ),

                nn.SiLU(),

                nn.Linear(
                    dim,
                    dim,
                ),
            )
        )

    def forward(
        self,
        t,
    ):

        half = (
            self.freq_dim // 2
        )

        freqs = torch.exp(
            -math.log(
                10000.0
            )
            * torch.arange(
                half,
                device=t.device,
                dtype=torch.float32,
            )
            / half
        )

        args = (
            (
                t * 1000.0
            )[:, None].float()
            * freqs[None]
        )

        emb = torch.cat(
            [
                torch.cos(args),
                torch.sin(args),
            ],
            dim=-1,
        )

        dtype = next(
            self.mlp.parameters()
        ).dtype

        return self.mlp(
            emb.to(dtype)
        )


# ============================================================
# MODULATE
# ============================================================

def modulate(
    x,
    shift,
    scale,
):

    return (
        x
        * (
            1
            + scale.unsqueeze(1)
        )
        + shift.unsqueeze(1)
    )


# ============================================================
# BLOCK
# ============================================================

class ZYIBlock(nn.Module):

    def __init__(
        self,
        dim,
        heads,
        mlp_ratio,
    ):

        super().__init__()

        self.norm1 = RMSNorm(
            dim,
            affine=False,
        )

        self.self_attn = (
            nn.MultiheadAttention(
                dim,
                heads,
                batch_first=True,
            )
        )

        self.norm_ca = RMSNorm(
            dim
        )

        self.cross_attn = (
            nn.MultiheadAttention(
                dim,
                heads,
                batch_first=True,
            )
        )

        self.norm2 = RMSNorm(
            dim,
            affine=False,
        )

        hidden = int(
            dim
            * mlp_ratio
        )

        self.mlp = (
            nn.Sequential(
                nn.Linear(
                    dim,
                    hidden,
                ),

                nn.GELU(
                    approximate="tanh"
                ),

                nn.Linear(
                    hidden,
                    dim,
                ),
            )
        )

        self.ada = (
            nn.Sequential(
                nn.SiLU(),

                nn.Linear(
                    dim,
                    6 * dim,
                ),
            )
        )

    def forward(
        self,
        x,
        t_emb,
        ctx,
        ctx_mask,
    ):

        (
            sh1,
            sc1,
            g1,
            sh2,
            sc2,
            g2,
        ) = (
            self.ada(
                t_emb
            ).chunk(
                6,
                dim=-1,
            )
        )

        h = modulate(
            self.norm1(x),
            sh1,
            sc1,
        )

        h, _ = (
            self.self_attn(
                h,
                h,
                h,
                need_weights=False,
            )
        )

        x = (
            x
            + g1.unsqueeze(1)
            * h
        )

        h = self.norm_ca(
            x
        )

        h, _ = (
            self.cross_attn(
                h,
                ctx,
                ctx,
                key_padding_mask=(
                    ~ctx_mask
                ),
                need_weights=False,
            )
        )

        x = x + h

        h = modulate(
            self.norm2(x),
            sh2,
            sc2,
        )

        x = (
            x
            + g2.unsqueeze(1)
            * self.mlp(h)
        )

        return x


# ============================================================
# ZYI DiT
# ============================================================

class ZYIDiT(nn.Module):

    def __init__(
        self,
        c,
    ):

        super().__init__()

        self.c = c

        self.grid = (
            c.latent_size
            // c.patch_size
        )

        n_tokens = (
            self.grid ** 2
        )

        out_dim = (
            c.latent_channels
            * c.patch_size ** 2
        )

        self.patchify = nn.Conv2d(
            c.latent_channels,
            c.hidden_dim,
            kernel_size=c.patch_size,
            stride=c.patch_size,
        )

        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                n_tokens,
                c.hidden_dim,
            )
        )

        self.t_embed = (
            TimestepEmbedder(
                c.hidden_dim
            )
        )

        self.text_proj = (
            nn.Sequential(
                nn.Linear(
                    c.text_emb_dim,
                    c.hidden_dim,
                ),

                RMSNorm(
                    c.hidden_dim
                ),
            )
        )

        self.blocks = (
            nn.ModuleList(
                [
                    ZYIBlock(
                        c.hidden_dim,
                        c.heads,
                        c.mlp_ratio,
                    )

                    for _ in range(
                        c.depth
                    )
                ]
            )
        )

        self.final_norm = (
            RMSNorm(
                c.hidden_dim,
                affine=False,
            )
        )

        self.final_ada = (
            nn.Sequential(
                nn.SiLU(),

                nn.Linear(
                    c.hidden_dim,
                    2
                    * c.hidden_dim,
                ),
            )
        )

        self.final_proj = (
            nn.Linear(
                c.hidden_dim,
                out_dim,
            )
        )


    def unpatchify(
        self,
        x,
    ):

        b = x.shape[0]

        p = (
            self.c.patch_size
        )

        g = self.grid

        ch = (
            self.c.latent_channels
        )

        x = x.reshape(
            b,
            g,
            g,
            p,
            p,
            ch,
        )

        x = torch.einsum(
            "bhwpqc->bchpwq",
            x,
        )

        return x.reshape(
            b,
            ch,
            g * p,
            g * p,
        )


    def forward(
        self,
        x,
        t,
        text_emb,
        text_mask,
    ):

        h = (
            self.patchify(x)
            .flatten(2)
            .transpose(1, 2)
        )

        h = (
            h
            + self.pos_emb
        )

        t_emb = (
            self.t_embed(t)
        )

        ctx = (
            self.text_proj(
                text_emb
            )
        )

        for block in self.blocks:

            h = block(
                h,
                t_emb,
                ctx,
                text_mask,
            )

        (
            shift,
            scale,
        ) = (
            self.final_ada(
                t_emb
            ).chunk(
                2,
                dim=-1,
            )
        )

        h = modulate(
            self.final_norm(h),
            shift,
            scale,
        )

        h = self.final_proj(
            h
        )

        return (
            self.unpatchify(h)
        )


# ============================================================
# STATE DICT
# ============================================================

def clean_state_dict(
    state,
):

    cleaned = {}

    for key, value in state.items():

        if key.startswith(
            "_orig_mod."
        ):

            key = key[
                len(
                    "_orig_mod."
                ):
            ]

        cleaned[key] = value

    return cleaned


# ============================================================
# ENGINE
# ============================================================

class ZYIEngine:

    def __init__(
        self,
        config_path,
    ):

        self.config_path = Path(
            config_path
        )

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as f:

            self.settings = (
                json.load(f)
            )

        self.lock = (
            threading.Lock()
        )

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA nao disponivel "
                "no container Salad."
            )

        self.device = (
            torch.device(
                "cuda:0"
            )
        )

        self.text_device = (
            torch.device(
                self.settings.get(
                    "text_encoder_device",
                    "cpu",
                )
            )
        )

        self.vae_device = (
            torch.device(
                self.settings.get(
                    "vae_device",
                    "cpu",
                )
            )
        )

        self.dtype = (
            torch.float16
        )


        # ====================================================
        # CPU
        # ====================================================

        cpu_threads = int(
            os.environ.get(
                "ZYI_CPU_THREADS",
                self.settings.get(
                    "cpu_threads",
                    4,
                ),
            )
        )

        torch.set_num_threads(
            max(
                1,
                cpu_threads,
            )
        )

        try:

            torch.set_num_interop_threads(
                max(
                    1,
                    min(
                        2,
                        cpu_threads,
                    ),
                )
            )

        except RuntimeError:
            pass


        # ====================================================
        # GTX 10XX / PASCAL
        # ====================================================

        try:

            torch.backends.cuda.enable_flash_sdp(
                False
            )

            torch.backends.cuda.enable_mem_efficient_sdp(
                False
            )

            torch.backends.cuda.enable_math_sdp(
                True
            )

        except Exception:
            pass

        torch.backends.cudnn.benchmark = (
            True
        )


        # ====================================================
        # MODEL CONFIG
        # ====================================================

        self.c = SimpleNamespace(

            latent_channels=4,

            latent_size=32,

            patch_size=2,

            hidden_dim=512,

            depth=31,

            heads=8,

            mlp_ratio=4,

            text_emb_dim=768,

            text_max_len=32,

            vae_scale=0.18215,
        )


        self._load_all()


    # ========================================================
    # DOWNLOAD CHECKPOINT
    # ========================================================

    def _checkpoint_path(
        self,
    ):

        repo_id = (
            os.environ.get(
                "MODEL_REPO",
                self.settings.get(
                    "model_repo",
                    (
                        "caikybaldo999/"
                        "ZYI-1.6-BASE-180M-BETA"
                    ),
                ),
            )
        )

        model_dir = Path(
            os.environ.get(
                "MODEL_DIR",
                self.settings.get(
                    "model_dir",
                    "/models/zyi",
                ),
            )
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        env_file = (
            os.environ.get(
                "MODEL_FILE",
                "",
            ).strip()
        )

        configured_file = (
            self.settings.get(
                "model_file",
                "",
            )
        )

        candidates = [
            env_file,

            configured_file,

            "zyi_1_6_base_180m_beta.pt",

            "final_ema.pt",

            "final.pt",

            "last.pt",
        ]

        candidates = [
            x
            for x in candidates
            if x
        ]

        # remover duplicados
        candidates = list(
            dict.fromkeys(
                candidates
            )
        )

        token = (
            os.environ.get(
                "HF_TOKEN"
            )
            or None
        )

        # primeiro verifica local
        for filename in candidates:

            local = (
                model_dir
                / filename
            )

            if local.exists():

                print(
                    "[ZYI] Checkpoint local:",
                    local,
                )

                return local


        # depois tenta HF
        errors = []

        for filename in candidates:

            print(
                "[ZYI] Tentando baixar:",
                repo_id,
                filename,
            )

            try:

                downloaded = (
                    hf_hub_download(
                        repo_id=repo_id,

                        filename=filename,

                        repo_type="model",

                        token=token,

                        local_dir=str(
                            model_dir
                        ),
                    )
                )

                print(
                    "[ZYI] Download:",
                    downloaded,
                )

                return Path(
                    downloaded
                )

            except Exception as exc:

                errors.append(
                    (
                        filename,
                        str(exc),
                    )
                )

        raise RuntimeError(
            "Nao encontrei checkpoint "
            f"no repo {repo_id}. "
            f"Tentados: {candidates}. "
            f"Erros: {errors[:2]}"
        )


    # ========================================================
    # GPU STATS
    # ========================================================

    def _gpu_stats(
        self,
    ):

        p = (
            torch.cuda
            .get_device_properties(0)
        )

        return {
            "name":
                p.name,

            "total_gb":
                p.total_memory
                / 1024**3,

            "allocated_gb":
                torch.cuda
                .memory_allocated()
                / 1024**3,

            "reserved_gb":
                torch.cuda
                .memory_reserved()
                / 1024**3,
        }


    # ========================================================
    # LOAD
    # ========================================================

    def _load_all(
        self,
    ):

        checkpoint_path = (
            self._checkpoint_path()
        )

        print(
            "[ZYI] GPU:",
            torch.cuda
            .get_device_name(0),
        )

        print(
            "[ZYI] Checkpoint:",
            checkpoint_path,
        )

        checkpoint = torch.load(
            str(
                checkpoint_path
            ),
            map_location="cpu",
            weights_only=False,
        )


        # ====================================================
        # EMA PRIORIDADE
        # ====================================================

        if (
            isinstance(
                checkpoint,
                dict,
            )
            and "ema"
            in checkpoint
        ):

            print(
                "[ZYI] Usando EMA."
            )

            state = (
                checkpoint["ema"]
            )

        elif (
            isinstance(
                checkpoint,
                dict,
            )
            and "model"
            in checkpoint
        ):

            print(
                "[ZYI] Usando model."
            )

            state = (
                checkpoint["model"]
            )

        elif (
            isinstance(
                checkpoint,
                dict,
            )
            and any(
                key.startswith(
                    "blocks."
                )
                or key.startswith(
                    "_orig_mod.blocks."
                )

                for key
                in checkpoint
            )
        ):

            state = checkpoint

        else:

            raise RuntimeError(
                "Formato de checkpoint "
                "ZYI nao reconhecido."
            )


        state = (
            clean_state_dict(
                state
            )
        )


        # ====================================================
        # DiT
        # ====================================================

        print(
            "[ZYI] Criando "
            "ZYI DiT 180M..."
        )

        self.net = (
            ZYIDiT(
                self.c
            )
            .half()
        )

        missing, unexpected = (
            self.net.load_state_dict(
                state,
                strict=False,
            )
        )

        if (
            missing
            or unexpected
        ):

            print(
                "[ZYI] Missing:",
                missing[:20],
            )

            print(
                "[ZYI] Unexpected:",
                unexpected[:20],
            )

            raise RuntimeError(
                "Checkpoint nao corresponde "
                "a arquitetura ZYI 180M."
            )

        del state
        del checkpoint

        gc.collect()


        # ====================================================
        # GPU
        # ====================================================

        self.net = (
            self.net
            .to(
                self.device
            )
            .eval()
        )

        self.net.requires_grad_(
            False
        )

        torch.cuda.empty_cache()


        # ====================================================
        # TEXT ENCODER CPU
        # ====================================================

        text_repo = (
            os.environ.get(
                "TEXT_ENCODER_REPO",
                self.settings[
                    "text_encoder_repo"
                ],
            )
        )

        print(
            "[ZYI] Carregando T5 CPU:",
            text_repo,
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                text_repo
            )
        )

        self.text_encoder = (
            T5EncoderModel
            .from_pretrained(
                text_repo,

                torch_dtype=torch.float32,

                low_cpu_mem_usage=True,
            )
            .to(
                self.text_device
            )
            .eval()
        )

        self.text_encoder.requires_grad_(
            False
        )


        # ====================================================
        # VAE CPU
        # ====================================================

        vae_repo = (
            os.environ.get(
                "VAE_REPO",
                self.settings[
                    "vae_repo"
                ],
            )
        )

        print(
            "[ZYI] Carregando VAE CPU:",
            vae_repo,
        )

        self.vae = (
            AutoencoderKL
            .from_pretrained(
                vae_repo,

                torch_dtype=torch.float32,

                low_cpu_mem_usage=True,
            )
            .to(
                self.vae_device
            )
            .eval()
        )

        self.vae.requires_grad_(
            False
        )


        # ====================================================
        # NULL CONDITION
        # ====================================================

        (
            self.null_emb_cpu,
            self.null_mask_cpu,
        ) = (
            self._encode_text_cpu(
                [""]
            )
        )

        stats = (
            self._gpu_stats()
        )

        print(
            "[ZYI] READY | "
            f"{stats['name']} | "
            f"VRAM total="
            f"{stats['total_gb']:.2f} GB | "
            f"alloc="
            f"{stats['allocated_gb']:.2f} GB"
        )


    # ========================================================
    # TEXT
    # ========================================================

    @torch.inference_mode()
    def _encode_text_cpu(
        self,
        captions,
    ):

        tokens = (
            self.tokenizer(
                captions,

                padding="max_length",

                truncation=True,

                max_length=(
                    self.c.text_max_len
                ),

                return_tensors="pt",
            )
        )

        ids = (
            tokens.input_ids
            .to(
                self.text_device
            )
        )

        attention = (
            tokens.attention_mask
            .to(
                self.text_device
            )
        )

        output = (
            self.text_encoder(
                input_ids=ids,

                attention_mask=attention,
            )
        )

        return (
            output
            .last_hidden_state
            .float()
            .cpu(),

            attention
            .bool()
            .cpu(),
        )


    # ========================================================
    # CONDITION -> GPU
    # ========================================================

    def _to_gpu_condition(
        self,
        embeddings,
        mask,
    ):

        embeddings = (
            embeddings.to(
                self.device,

                dtype=self.dtype,

                non_blocking=False,
            )
        )

        mask = mask.to(
            self.device,
            non_blocking=False,
        )

        return (
            embeddings,
            mask,
        )


    # ========================================================
    # GENERATE
    # ========================================================

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        steps,
        cfg_scale,
        seed,
        output_path,
    ):

        with self.lock:

            start = (
                time.perf_counter()
            )

            try:

                # ============================================
                # TEXT
                # ============================================

                (
                    cond_cpu,
                    cond_mask_cpu,
                ) = (
                    self._encode_text_cpu(
                        [prompt]
                    )
                )

                (
                    cond,
                    cond_mask,
                ) = (
                    self._to_gpu_condition(
                        cond_cpu,
                        cond_mask_cpu,
                    )
                )


                # ============================================
                # UNCONDITIONAL
                # ============================================

                uncond = None
                uncond_mask = None

                if (
                    cfg_scale
                    > 1.000001
                ):

                    (
                        uncond,
                        uncond_mask,
                    ) = (
                        self._to_gpu_condition(
                            self.null_emb_cpu,
                            self.null_mask_cpu,
                        )
                    )


                # ============================================
                # NOISE
                # ============================================

                generator = (
                    torch.Generator(
                        device=self.device
                    )
                    .manual_seed(
                        int(seed)
                    )
                )

                x = torch.randn(
                    1,
                    self.c.latent_channels,
                    self.c.latent_size,
                    self.c.latent_size,

                    device=self.device,

                    generator=generator,

                    dtype=self.dtype,
                )

                dt = (
                    1.0
                    / float(steps)
                )


                # ============================================
                # RECTIFIED FLOW
                #
                # treino:
                #
                # xt = (1-t)x0 + t*noise
                #
                # target = noise - x0
                #
                # geração:
                #
                # t = 1 -> 0
                # x = x - dt * velocity
                # ============================================

                for i in range(
                    int(steps)
                ):

                    t = torch.full(
                        (1,),

                        1.0
                        - i * dt,

                        device=self.device,

                        dtype=torch.float32,
                    )


                    # ========================================
                    # COND
                    # ========================================

                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    ):

                        velocity_cond = (
                            self.net(
                                x,
                                t,
                                cond,
                                cond_mask,
                            )
                        )


                    # ========================================
                    # CFG
                    #
                    # SEQUENCIAL:
                    # não cria batch 2.
                    # ========================================

                    if (
                        cfg_scale
                        <= 1.000001
                    ):

                        velocity = (
                            velocity_cond
                        )

                    else:

                        with torch.autocast(
                            device_type="cuda",
                            dtype=torch.float16,
                        ):

                            velocity_uncond = (
                                self.net(
                                    x,
                                    t,
                                    uncond,
                                    uncond_mask,
                                )
                            )

                        velocity = (
                            velocity_uncond
                            + float(
                                cfg_scale
                            )
                            * (
                                velocity_cond
                                - velocity_uncond
                            )
                        )


                    # ========================================
                    # EULER
                    # ========================================

                    x = (
                        x
                        - dt
                        * velocity
                    ).to(
                        dtype=self.dtype
                    )


                torch.cuda.synchronize()


                # ============================================
                # LIBERAR CONDITION
                # ============================================

                del cond
                del cond_mask
                del cond_cpu
                del cond_mask_cpu

                if uncond is not None:

                    del uncond
                    del uncond_mask


                # ============================================
                # GPU -> CPU
                # ============================================

                latent = (
                    x.float().cpu()
                    / self.c.vae_scale
                )

                del x

                torch.cuda.empty_cache()


                # ============================================
                # VAE CPU
                # ============================================

                latent = latent.to(
                    self.vae_device
                )

                decoded = (
                    self.vae.decode(
                        latent
                    ).sample
                )

                image = (
                    (
                        decoded
                        .float()
                        .clamp(
                            -1,
                            1,
                        )
                        + 1
                    )
                    * 127.5
                )

                image = (
                    image
                    .round()
                    .to(
                        torch.uint8
                    )
                )

                array = (
                    image[0]
                    .permute(
                        1,
                        2,
                        0,
                    )
                    .cpu()
                    .numpy()
                )


                # ============================================
                # SAVE
                # ============================================

                output_path = Path(
                    output_path
                )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                Image.fromarray(
                    array
                ).save(
                    output_path
                )


                elapsed = (
                    time.perf_counter()
                    - start
                )

                print(
                    "[ZYI] Gerado | "
                    f"{elapsed:.1f}s | "
                    f"steps={steps} | "
                    f"cfg={cfg_scale:.2f} | "
                    f"seed={seed}"
                )

                return output_path


            # =================================================
            # OOM
            # =================================================

            except torch.cuda.OutOfMemoryError as exc:

                torch.cuda.empty_cache()

                gc.collect()

                raise RuntimeError(
                    "CUDA OOM. GTX 1050 Ti "
                    "possui apenas 4 GB. "
                    "Use CFG=1.0."
                ) from exc


            finally:

                gc.collect()

                torch.cuda.empty_cache()
