"""
Ollama Audio Pipeline — FastAPI Server
오디오 파일을 받아 ffmpeg로 변환 후 gemma4:e4b에 전달, 스트리밍 응답 반환
"""

import asyncio
import base64
import json
import subprocess
import tempfile
import os
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# ───────────────────────────── 설정 ─────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL   = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

# gemma4가 요구하는 오디오 스펙
AUDIO_SAMPLE_RATE = 16000   # 16kHz
AUDIO_CHANNELS    = 1       # mono
AUDIO_FORMAT      = "s16le" # 16-bit PCM (ffmpeg용)

app = FastAPI(
    title="Ollama Audio Pipeline",
    description="오디오 → ffmpeg 변환 → gemma4:e4b 분석 → 스트리밍 응답",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────────── 유틸 ─────────────────────────────

def convert_to_wav(input_path: str, output_path: str) -> None:
    """
    ffmpeg로 어떤 오디오 포맷이든 gemma4 요구 스펙(16kHz mono WAV)으로 변환
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(AUDIO_SAMPLE_RATE),   # 샘플레이트 16kHz
        "-ac", str(AUDIO_CHANNELS),       # 모노
        "-c:a", "pcm_s16le",              # 16-bit PCM
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 변환 실패:\n{result.stderr}")


def audio_to_base64(wav_path: str) -> str:
    """WAV 파일을 base64 문자열로 인코딩"""
    with open(wav_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def stream_ollama(
    model: str,
    prompt: str,
    audio_b64: str,
) -> AsyncGenerator[str, None]:
    """
    Ollama /api/chat 엔드포인트에 오디오 + 프롬프트를 전달하고
    스트리밍으로 토큰을 yield
    """
    payload = {
        "model": model,
        "stream": True,
        "options": {"num_ctx": 8192},
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [audio_b64],   # Ollama는 images 필드로 오디오도 받음
            }
        ],
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Ollama 오류: {body.decode()}",
                )
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue


# ───────────────────────────── 엔드포인트 ─────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    """서버 + Ollama 연결 상태 확인"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
        return {
            "status": "ok",
            "ollama": "connected",
            "available_models": models,
            "target_model": DEFAULT_MODEL,
            "model_ready": DEFAULT_MODEL in models,
        }
    except Exception as e:
        return {"status": "ok", "ollama": "disconnected", "error": str(e)}


@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(..., description="분석할 오디오 파일 (mp3, wav, m4a, webm 등)"),
    prompt: str = Form(
        default="이 오디오를 분석해줘. 무슨 내용인지, 화자의 톤이나 감정, 주요 내용을 설명해줘.",
        description="오디오에 대해 모델에게 할 질문",
    ),
    model: str = Form(default=DEFAULT_MODEL, description="사용할 Ollama 모델"),
):
    """
    오디오 파일을 업로드하면 gemma4가 분석해서 스트리밍으로 응답을 반환합니다.

    - **file**: 오디오 파일 (mp3, wav, m4a, ogg, webm 등 ffmpeg가 지원하는 모든 포맷)
    - **prompt**: 오디오에 대해 묻고 싶은 내용
    - **model**: 사용할 모델 (기본값: gemma4:e4b)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. 업로드 파일 저장
        ext = Path(file.filename).suffix or ".tmp"
        input_path  = os.path.join(tmpdir, f"input{ext}")
        output_path = os.path.join(tmpdir, "output.wav")

        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        # 2. ffmpeg 변환 (16kHz mono WAV)
        try:
            convert_to_wav(input_path, output_path)
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))

        # 3. base64 인코딩
        audio_b64 = audio_to_base64(output_path)

        # 4. WAV 파일 크기 로그
        wav_size_kb = os.path.getsize(output_path) / 1024
        print(f"[audio] {file.filename} → WAV {wav_size_kb:.1f}KB | model={model}")

    # 5. Ollama 스트리밍 응답
    async def generate():
        yield f"data: [분석 시작 — {file.filename} → {model}]\n\n"
        async for token in stream_ollama(model, prompt, audio_b64):
            # SSE 형식으로 전달
            yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(default="Korean", description="주 언어 힌트"),
    model: str = Form(default=DEFAULT_MODEL),
):
    """
    오디오를 텍스트로 전사(transcription)하는 단축 엔드포인트.
    /analyze의 프롬프트를 전사에 특화시킨 버전.
    """
    prompt = (
        f"다음 오디오를 {language}로 정확하게 전사해줘. "
        "말한 내용만 그대로 텍스트로 출력하고, 부가 설명은 하지 마."
    )

    # file을 재사용하기 위해 내부적으로 analyze 로직 재호출
    with tempfile.TemporaryDirectory() as tmpdir:
        ext = Path(file.filename).suffix or ".tmp"
        input_path  = os.path.join(tmpdir, f"input{ext}")
        output_path = os.path.join(tmpdir, "output.wav")

        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        try:
            convert_to_wav(input_path, output_path)
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))

        audio_b64 = audio_to_base64(output_path)

    async def generate():
        async for token in stream_ollama(model, prompt, audio_b64):
            yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)