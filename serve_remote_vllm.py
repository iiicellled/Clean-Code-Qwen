from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


# vLLM 0.6.6 may internally instantiate the slow Qwen2Tokenizer with older
# Transformers combinations. That class can miss this property, which vLLM reads.
if not hasattr(PreTrainedTokenizer, "all_special_tokens_extended"):
    PreTrainedTokenizer.all_special_tokens_extended = property(lambda self: list(self.all_special_tokens))
_LEGACY_VLLM_DTYPE = os.environ.pop("VLLM_DTYPE", None)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

load_dotenv()


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(default=768, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


class Settings:
    base_model_path = Path(os.getenv("BASE_MODEL_PATH", "./outputs/qwen-coder-simplifier-dpo-merged")).resolve()
    lora_adapter_path = Path(os.getenv("LORA_ADAPTER_PATH", "./output_models/qwen-coder-simplifier-dpo-lora")).resolve()
    enable_lora = os.getenv("ENABLE_LORA", "false").strip().lower() in {"1", "true", "yes", "on"}
    tokenizer_path = Path(os.getenv("TOKENIZER_PATH", str(base_model_path))).resolve()
    served_model_name = os.getenv("SERVED_MODEL_NAME", "qwen-coder-simplifier-dpo-merged")
    api_key = os.getenv("REMOTE_API_KEY") or None
    status_file = Path(os.getenv("STATUS_FILE", "./serve_status.json")).resolve()

    # Tesla V100 does not support bfloat16. float16 is the safest default for vLLM on V100.
    vllm_dtype = os.getenv("SERVE_DTYPE", _LEGACY_VLLM_DTYPE or os.getenv("TORCH_DTYPE", "float16"))
    tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))
    max_model_len = int(os.getenv("MAX_MODEL_LEN", "4096"))
    min_new_tokens = int(os.getenv("MIN_NEW_TOKENS", "1"))
    max_lora_rank = int(os.getenv("MAX_LORA_RANK", "64"))
    enforce_eager = os.getenv("VLLM_ENFORCE_EAGER", "false").strip().lower() in {"1", "true", "yes", "on"}
    trust_remote_code = os.getenv("TRUST_REMOTE_CODE", "true").strip().lower() in {"1", "true", "yes", "on"}


settings = Settings()
app = FastAPI(title="Remote Qwen Coder vLLM Server", version="1.1.0")
_tokenizer: Any | None = None
_llm: LLM | None = None
_lora_request: LoRARequest | None = None
_engine_lock = threading.Lock()
_status: dict[str, Any] = {
    "stage": "booted",
    "message": "Remote server module imported; vLLM engine is not loaded yet.",
    "request_id": None,
    "model_loaded": False,
    "last_error": None,
    "updated_at": time.time(),
}


def safe_cuda_status() -> dict[str, Any]:
    """Return GPU status without touching torch.cuda before vLLM forks workers."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"cuda_status": "nvidia-smi not found"}
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        gpus = []
        for index, line in enumerate(result.stdout.splitlines()):
            name, used_mb, total_mb = [part.strip() for part in line.split(",", 2)]
            gpus.append(
                {
                    "index": index,
                    "name": name,
                    "memory_used_gb": round(float(used_mb) / 1024, 3),
                    "memory_total_gb": round(float(total_mb) / 1024, 3),
                }
            )
        return {"cuda_available": bool(gpus), "gpus": gpus}
    except Exception as exc:
        return {"cuda_status_error": f"{type(exc).__name__}: {exc}"}


def write_status_file(snapshot: dict[str, Any]) -> None:
    try:
        settings.status_file.parent.mkdir(parents=True, exist_ok=True)
        settings.status_file.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"Failed to write status file {settings.status_file}: {exc}", flush=True)


def current_status(include_cuda: bool = True) -> dict[str, Any]:
    data = dict(_status)
    data.update(
        {
            "base_model_path": str(settings.base_model_path),
            "base_model_exists": settings.base_model_path.exists(),
            "lora_adapter_path": str(settings.lora_adapter_path),
            "lora_adapter_exists": settings.lora_adapter_path.exists(),
            "enable_lora": settings.enable_lora,
            "tokenizer_path": str(settings.tokenizer_path),
            "tokenizer_exists": settings.tokenizer_path.exists(),
            "served_model_name": settings.served_model_name,
            "status_file": str(settings.status_file),
            "pid": os.getpid(),
            "backend": "vllm",
            "vllm_dtype": settings.vllm_dtype,
            "tensor_parallel_size": settings.tensor_parallel_size,
            "gpu_memory_utilization": settings.gpu_memory_utilization,
            "max_model_len": settings.max_model_len,
            "min_new_tokens": settings.min_new_tokens,
            "max_lora_rank": settings.max_lora_rank,
        }
    )
    if include_cuda:
        data.update(safe_cuda_status())
    return data


def set_status(stage: str, message: str, request_id: str | None = None, **extra: Any) -> None:
    _status.update(
        {
            "stage": stage,
            "message": message,
            "request_id": request_id if request_id is not None else _status.get("request_id"),
            "model_loaded": _llm is not None,
            "updated_at": time.time(),
            **extra,
        }
    )
    snapshot = current_status(include_cuda=True)
    write_status_file(snapshot)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {stage}: {message}", flush=True)


def _check_auth(authorization: str | None) -> None:
    if not settings.api_key:
        return
    expected = f"Bearer {settings.api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _adapter_rank(adapter_path: Path) -> int | None:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return None
    try:
        return int(json.loads(config_path.read_text(encoding="utf-8")).get("r"))
    except Exception:
        return None


def load_engine(request_id: str) -> tuple[Any, LLM, LoRARequest | None]:
    global _tokenizer, _llm, _lora_request
    if _tokenizer is not None and _llm is not None and (not settings.enable_lora or _lora_request is not None):
        set_status("model_ready", "vLLM engine is already loaded.", request_id, **safe_cuda_status())
        return _tokenizer, _llm, _lora_request

    with _engine_lock:
        if _tokenizer is not None and _llm is not None and (not settings.enable_lora or _lora_request is not None):
            set_status("model_ready", "vLLM engine is already loaded.", request_id, **safe_cuda_status())
            return _tokenizer, _llm, _lora_request

        set_status("checking_paths", "Checking model, tokenizer, and optional LoRA adapter paths.", request_id)
        if not settings.base_model_path.exists():
            raise RuntimeError(f"BASE_MODEL_PATH does not exist: {settings.base_model_path}")
        if settings.enable_lora and not settings.lora_adapter_path.exists():
            raise RuntimeError(f"LORA_ADAPTER_PATH does not exist: {settings.lora_adapter_path}")
        if not settings.tokenizer_path.exists():
            raise RuntimeError(f"TOKENIZER_PATH does not exist: {settings.tokenizer_path}")

        adapter_rank = _adapter_rank(settings.lora_adapter_path) if settings.enable_lora else None
        max_lora_rank = max(settings.max_lora_rank, adapter_rank or 0)

        set_status("loading_tokenizer", f"Loading fast tokenizer from {settings.tokenizer_path}.", request_id)
        tokenizer = AutoTokenizer.from_pretrained(
            settings.tokenizer_path,
            trust_remote_code=settings.trust_remote_code,
            local_files_only=True,
            use_fast=True,
        )
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("Fast tokenizer is required for vLLM 0.6.6; set TOKENIZER_PATH to the base Qwen tokenizer directory.")

        set_status("loading_vllm_engine", f"Loading vLLM engine from {settings.base_model_path}.", request_id, **safe_cuda_status())
        llm = LLM(
            model=str(settings.base_model_path),
            tokenizer=str(settings.tokenizer_path),
            skip_tokenizer_init=True,
            trust_remote_code=settings.trust_remote_code,
            dtype=settings.vllm_dtype,
            tensor_parallel_size=settings.tensor_parallel_size,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            max_model_len=settings.max_model_len,
            enable_lora=settings.enable_lora,
            max_lora_rank=max_lora_rank,
            enforce_eager=settings.enforce_eager,
        )
        lora_request = LoRARequest(settings.served_model_name, 1, str(settings.lora_adapter_path)) if settings.enable_lora else None

        _tokenizer = tokenizer
        _llm = llm
        _lora_request = lora_request
        set_status(
            "model_ready",
            "vLLM engine loaded successfully.",
            request_id,
            last_error=None,
            adapter_rank=adapter_rank,
            **safe_cuda_status(),
        )
        return tokenizer, llm, lora_request


@app.on_event("startup")
def startup_status() -> None:
    set_status("startup", "FastAPI startup completed; vLLM engine is not loaded yet.")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": "vllm",
        "model_loaded": _llm is not None,
        "served_model_name": settings.served_model_name,
    }


@app.get("/debug/ping")
def debug_ping() -> dict[str, Any]:
    return {
        "ok": True,
        "pid": os.getpid(),
        "time": time.time(),
        "message": "FastAPI process is alive.",
        "status_file": str(settings.status_file),
    }


@app.get("/debug/status")
def debug_status(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_auth(authorization)
    data = current_status(include_cuda=True)
    write_status_file(data)
    return data


def build_prompt(tokenizer: Any, messages: list[Message]) -> str:
    return tokenizer.apply_chat_template(
        [{"role": message.role, "content": message.content} for message in messages],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_answer(request: ChatCompletionRequest, request_id: str) -> tuple[str, int | None]:
    tokenizer, llm, lora_request = load_engine(request_id)
    set_status("building_prompt", "Applying chat template.", request_id, **safe_cuda_status())
    prompt = build_prompt(tokenizer, request.messages)
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_tokens = len(prompt_token_ids)

    set_status(
        "generating",
        f"Generating up to {request.max_tokens} tokens with vLLM.",
        request_id,
        prompt_tokens=prompt_tokens,
        **safe_cuda_status(),
    )
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=max(request.temperature, 1e-5),
        top_p=request.top_p,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
        min_tokens=settings.min_new_tokens,
    )
    generate_kwargs = {"lora_request": lora_request} if lora_request is not None else {}
    outputs = llm.generate(
        prompts=None,
        sampling_params=sampling_params,
        prompt_token_ids=[prompt_token_ids],
        **generate_kwargs,
    )
    completion = outputs[0].outputs[0]
    generated_token_ids = list(completion.token_ids or [])
    raw_decoded = tokenizer.decode(generated_token_ids, skip_special_tokens=False) if generated_token_ids else ""
    answer = (completion.text or "").strip()
    if not answer and generated_token_ids:
        answer = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
    set_status(
        "generated",
        "vLLM generation finished; decoding output tokens.",
        request_id,
        finish_reason=getattr(completion, "finish_reason", None),
        generated_tokens=len(generated_token_ids),
        generated_token_ids_head=generated_token_ids[:32],
        raw_completion_text=(completion.text or "")[:500],
        raw_decoded_text=raw_decoded[:500],
        decoded_answer_preview=answer[:500],
    )
    generated_tokens = len(generated_token_ids)
    return answer, generated_tokens


def sse_event(payload: dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def stream_chat_completion(request: ChatCompletionRequest, request_id: str, started_at: float):
    try:
        answer, generated_tokens = generate_answer(request, request_id)
        yield sse_event(
            {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model or settings.served_model_name,
                "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": None}],
            }
        )
        set_status(
            "completed",
            "Streaming request completed successfully.",
            request_id,
            elapsed_seconds=round(time.time() - started_at, 3),
            generated_tokens=generated_tokens,
            last_error=None,
            **safe_cuda_status(),
        )
        yield sse_event(
            {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model or settings.served_model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        yield sse_event("[DONE]")
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        set_status(
            "failed",
            error_text,
            request_id,
            elapsed_seconds=round(time.time() - started_at, 3),
            last_error=error_text,
            traceback=traceback.format_exc()[-4000:],
            **safe_cuda_status(),
        )
        yield sse_event({"error": error_text})
        yield sse_event("[DONE]")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(
    request: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
) -> ChatCompletionResponse | StreamingResponse:
    _check_auth(authorization)
    request_id = uuid.uuid4().hex[:12]
    started_at = time.time()
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request, request_id, started_at),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        set_status("request_received", f"Received chat request with {len(request.messages)} messages.", request_id)
        answer, generated_tokens = generate_answer(request, request_id)
        set_status(
            "completed",
            "Request completed successfully.",
            request_id,
            elapsed_seconds=round(time.time() - started_at, 3),
            generated_tokens=generated_tokens,
            last_error=None,
            **safe_cuda_status(),
        )
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        set_status(
            "failed",
            error_text,
            request_id,
            elapsed_seconds=round(time.time() - started_at, 3),
            last_error=error_text,
            traceback=traceback.format_exc()[-4000:],
            **safe_cuda_status(),
        )
        raise HTTPException(status_code=500, detail=f"Inference failed: {error_text}") from exc

    return ChatCompletionResponse(
        id=f"chatcmpl-{request_id}",
        created=int(time.time()),
        model=request.model or settings.served_model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=answer),
                finish_reason="stop",
            )
        ],
    )

















