"""
Ollama Audio Pipeline - FastAPI Server
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

# 설정
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLMA_MODEL", "gemma4:e4b")

# gemma4가 요구하는 오디오 스펙
AUDIO_SAMPLE_RATE = 16000 # 16kHz
AUDIO_CHANNELS = 1 # mono
AUDIO_FORMAT = "S16LE" # 16-bit PCM(ffmpeg 용)

app = FastAPI(
    title = "Ollama Audio Pipeline",
    description = "오디오 -> ffmpeg 변환 -> gemma4:e4b 분석 -> 스트리밍 응답",
    version = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)



