"""Streams a live dense RGB-D SLAM run to the browser.

GET  /             the 3D viewer (index.html)
GET  /api/status   sequence, device and whether a run is in progress
GET  /api/run      newline-delimited JSON events for one SLAM run:
                   {"type": "start", ...}, one {"type": "frame", ...} per frame,
                   then {"type": "done", ...} with the mesh's URL and trajectory error
GET  /api/mesh/ID  the reconstructed surface of a run as a packed binary mesh (see pack_mesh)

The GPU runs one SLAM pass at a time. A request that arrives while one is in progress gets a
replay of the most recent completed run (marked "replay": true) instead of waiting.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from collections import OrderedDict
from typing import Any, AsyncIterator, Callable, Iterator

import numpy as np
import open3d as o3d
import open3d.core as o3c
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from PIL import Image

import slam

SEQUENCE_DIR = Path(os.environ.get("SLAM_SEQUENCE_DIR", "/data/rgbd_dataset_freiburg3_long_office_household"))
STRIDE = int(os.environ.get("SLAM_STRIDE", "2"))
PORT = int(os.environ.get("PORT", "8080"))
INDEX_HTML = Path(os.environ.get("SLAM_INDEX_HTML", str(Path(__file__).with_name("index.html"))))

# Per-frame payload budget: points from every Nth frame, plus a thumbnail every Mth frame.
POINTS_EVERY = 3
POINTS_PER_FRAME = 1600
THUMB_EVERY = 3
THUMB_SIZE = (160, 120)
# Meshes of the most recent runs, so a replay can still fetch the surface it points at.
KEPT_MESHES = 3
GT_POINTS = 600

log = logging.getLogger("slam")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def pack_points(points: np.ndarray, colors: np.ndarray) -> dict[str, str | int]:
    """Positions as little-endian int16 millimetres and colours as uint8 RGB, base64 encoded.

    Millimetre int16 covers +/-32 m, far beyond an indoor sequence, at a third of float32's size.
    """
    mm = np.clip(np.round(points * 1000.0), -32767, 32767).astype("<i2")
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    return {
        "count": int(len(points)),
        "xyz": base64.b64encode(mm.tobytes()).decode(),
        "rgb": base64.b64encode(rgb.tobytes()).decode(),
    }


def pack_mesh(positions: np.ndarray, colors: np.ndarray, faces: np.ndarray) -> bytes:
    """Little-endian binary mesh, every section starting on a 4-byte boundary:

    uint32 vertex count, uint32 triangle count, int16[3V] positions in millimetres,
    uint8[3V] RGB colours, uint32[3T] vertex indices.
    """

    def pad4(b: bytes) -> bytes:
        return b + b"\0" * (-len(b) % 4)

    header = np.array([len(positions), len(faces)], dtype="<u4").tobytes()
    mm = np.clip(np.round(positions * 1000.0), -32767, 32767).astype("<i2").tobytes()
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8).tobytes()
    return header + pad4(mm) + pad4(rgb) + faces.astype("<u4").tobytes()


def thumbnail(path: Path) -> str:
    img = Image.open(path).convert("RGB").resize(THUMB_SIZE)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def device() -> o3c.Device:
    return o3c.Device("CUDA:0") if o3d.core.cuda.is_available() else o3c.Device("CPU:0")


def run_slam(seq: slam.Sequence, store_mesh: Callable[[bytes], str]) -> Iterator[dict[str, Any]]:
    """One full SLAM pass over `seq`, as JSON-ready events. The reconstructed mesh is handed to
    `store_mesh`, which returns the id it can be fetched under."""
    dev = device()
    run = slam.DenseSlam(seq, dev)
    n = len(seq.timestamps)
    yield {
        "type": "start",
        "sequence": seq.name,
        "frames": n,
        "stride": STRIDE,
        "device": str(dev),
        "intrinsics": seq.intrinsics,
        "image_size": [640, 480],
    }
    t0 = time.perf_counter()
    slam_seconds = 0.0
    for r in run.frames():
        slam_seconds += r.seconds
        event: dict[str, Any] = {
            "type": "frame",
            "i": r.index,
            "pose": [round(float(v), 5) for v in r.pose.reshape(-1)],
            "fitness": round(r.fitness, 3),
            "ms": round(r.seconds * 1000, 1),
        }
        if r.index % POINTS_EVERY == 0:
            event["points"] = pack_points(*run.frame_points(r.index, POINTS_PER_FRAME))
        if r.index % THUMB_EVERY == 0:
            event["thumb"] = thumbnail(seq.color_files[r.index])
        yield event

    t_mesh = time.perf_counter()
    positions, colors, faces = run.mesh()
    mesh_id = store_mesh(pack_mesh(positions, colors, faces))
    estimated = np.array([p[:3, 3] for p in run.poses])
    truth = slam.ground_truth_trajectory(seq, len(estimated))
    done: dict[str, Any] = {
        "type": "done",
        "frames": n,
        "slam_seconds": round(slam_seconds, 2),
        "mesh_seconds": round(time.perf_counter() - t_mesh, 2),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "mesh": {"url": f"api/mesh/{mesh_id}", "vertices": len(positions), "triangles": len(faces)},
    }
    if truth is not None:
        aligned, ate = slam.align_and_ate(estimated, truth)
        step = max(1, len(aligned) // GT_POINTS)
        done["ate_cm"] = round(ate * 100, 2)
        done["path_m"] = round(float(np.sum(np.linalg.norm(np.diff(truth, axis=0), axis=1))), 2)
        done["ground_truth"] = [[round(float(v), 4) for v in p] for p in aligned[::step]]
    yield done


state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    seq = slam.load_sequence(SEQUENCE_DIR, STRIDE)
    log.info("loaded %s: %d frames, device %s", seq.name, len(seq.timestamps), device())
    state.update(sequence=seq, lock=threading.Lock(), last_run=None, meshes=OrderedDict())
    # Warm up the CUDA kernels on a short prefix so the first visitor's run starts at full speed.
    warm = slam.Sequence(
        **{
            **seq.__dict__,
            "timestamps": seq.timestamps[:5],
            "depth_files": seq.depth_files[:5],
            "color_files": seq.color_files[:5],
        }
    )
    await asyncio.to_thread(lambda: [e for e in run_slam(warm, lambda _: "warmup") if e["type"] == "done"])
    log.info("warm-up done")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text())


@app.get("/api/status")
async def status() -> JSONResponse:
    seq: slam.Sequence = state["sequence"]
    lock: threading.Lock = state["lock"]
    return JSONResponse(
        {"sequence": seq.name, "frames": len(seq.timestamps), "device": str(device()), "busy": lock.locked()}
    )


def store_mesh(blob: bytes) -> str:
    meshes: OrderedDict[str, bytes] = state["meshes"]
    mesh_id = f"{time.time_ns():x}"
    meshes[mesh_id] = blob
    while len(meshes) > KEPT_MESHES:
        meshes.popitem(last=False)
    return mesh_id


@app.get("/api/mesh/{mesh_id}")
async def mesh(mesh_id: str) -> Response:
    blob = state["meshes"].get(mesh_id)
    if blob is None:
        raise HTTPException(404, "mesh expired; run SLAM again")
    return Response(blob, media_type="application/octet-stream", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/run")
async def run() -> StreamingResponse:
    lock: threading.Lock = state["lock"]
    seq: slam.Sequence = state["sequence"]
    # Unbounded: a viewer that disconnects mid-run must not block the SLAM thread, which then
    # could never release the GPU lock. One run is at most a few tens of MB of events.
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def live() -> None:
        recorded: list[str] = []
        try:
            for event in run_slam(seq, store_mesh):
                line = json.dumps(event, separators=(",", ":")) + "\n"
                recorded.append(line)
                asyncio.run_coroutine_threadsafe(queue.put(line), loop).result()
            state["last_run"] = recorded
        finally:
            lock.release()
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    if lock.acquire(blocking=False):
        threading.Thread(target=live, daemon=True).start()

        async def body() -> AsyncIterator[str]:
            while (line := await queue.get()) is not None:
                yield line

        return StreamingResponse(body(), media_type="application/x-ndjson")

    last: list[str] | None = state["last_run"]
    if last is None:
        return StreamingResponse(iter([json.dumps({"type": "busy"}) + "\n"]), media_type="application/x-ndjson")

    async def replay() -> AsyncIterator[str]:
        # Pace the replay like a live run so it reads the same way in the viewer.
        for line in last:
            yield line.replace('"type":"start"', '"type":"start","replay":true', 1)
            await asyncio.sleep(0.02)

    return StreamingResponse(replay(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
