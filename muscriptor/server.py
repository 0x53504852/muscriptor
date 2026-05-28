"""FastAPI server exposing transcription as an SSE event stream.

POST /transcribe with a wav file (multipart/form-data field `file`) returns
`text/event-stream` where each event's data is a JSON-encoded NoteStart /
NoteEnd dict (same shape as `muscriptor.main._event_to_dict`).
"""

import dataclasses
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from muscriptor.events import NoteEndEvent, NoteStartEvent
from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES
from muscriptor.utils.audio import _read_wav_file


class _TranscriberLike(Protocol):
    def transcribe(
        self,
        audio: str | Path | tuple[torch.Tensor, int],
        instrument_group: str | None = None,
    ) -> Iterator[NoteStartEvent | NoteEndEvent]: ...


def event_to_dict(ev: NoteStartEvent | NoteEndEvent) -> dict:
    if isinstance(ev, NoteStartEvent):
        return {"type": "start", **dataclasses.asdict(ev)}
    return {
        "type": "end",
        "end_time": ev.end_time,
        "start_event_index": ev.start_event_index,
    }


def create_app(model: _TranscriberLike, web_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="muscriptor")

    @app.get("/instruments")
    async def list_instruments():
        return {"instruments": list(MT3_FULL_PLUS_GROUP_NAMES.keys())}

    @app.post("/transcribe")
    async def transcribe(
        file: UploadFile = File(...),
        instruments: list[str] = Form(default_factory=list),
    ):
        data = await file.read()
        try:
            wav, sr = _read_wav_file(io.BytesIO(data))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid WAV: {e}")

        unknown = [n for n in instruments if n not in MT3_FULL_PLUS_GROUP_NAMES]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown instrument name(s): {', '.join(unknown)}",
            )
        instrument_group: str | None = (
            " ".join(str(MT3_FULL_PLUS_GROUP_NAMES[n]) for n in instruments)
            if instruments
            else None
        )

        def gen():
            for ev in model.transcribe((wav, sr), instrument_group=instrument_group):
                payload = json.dumps(event_to_dict(ev))
                yield f"data: {payload}\n\n"

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if web_dir is not None:
        web_path = Path(web_dir)
        if web_path.is_dir():
            app.mount("/", StaticFiles(directory=web_path, html=True), name="web")

    return app
