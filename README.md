# Live RGB-D SLAM on a Chalk GPU

Watch a GPU work out where a depth camera is and build a 3D map of the room it's in, both at
once, frame by frame. This is SLAM: simultaneous localization and mapping.

The camera is the handheld Kinect from the TUM RGB-D benchmark's
`fr3/long_office_household` sequence, a 22 m loop around two office desks. Every frame:

1. **Localize.** The new depth image is aligned to a raycast of the map built so far
   (frame-to-model ICP), which gives the camera's pose.
2. **Map.** The frame is fused into a TSDF voxel grid at that pose, so the map gets denser and
   cleaner as the camera moves.

The browser shows the map growing, the camera's path and its view. When the run ends, marching
cubes turns the fused TSDF into a coloured triangle mesh of the office. The viewer swaps the raw
points for that mesh and compares the tracked path with the dataset's motion-capture ground truth.

| File | Role |
| --- | --- |
| `slam.py` | Sequence loading, the track-then-integrate loop (Open3D's tensor SLAM pipeline), and trajectory error. |
| `server.py` | FastAPI app. `GET /api/run` streams one SLAM run as newline-delimited JSON; `GET /api/mesh/<id>` serves the reconstructed mesh. |
| `index.html` | three.js viewer: streamed points, camera path and frustum, camera feed, tracking stats. |
| `deploy.py` | Builds the GPU image with the sequence baked in and deploys a `chalkcompute.ScalingGroup`. |

## Run it

You need [`uv`](https://docs.astral.sh/uv/) and a Chalk environment that can schedule an L4 GPU.
Credentials come from your existing `chalk login`.

```bash
./deploy.py
```

Open the printed `web_url` and press **Start mapping**. The first build downloads the 1.5 GB
sequence into the image, so it takes a while. Code changes after that redeploy from cached layers.

To run it without a GPU, download and extract the sequence, then start the server on CPU. The
same code runs there, at about 10 frames per second.

```bash
SLAM_SEQUENCE_DIR=./rgbd_dataset_freiburg3_long_office_household python server.py
```

On one NVIDIA L4, tracking and fusing a 640×480 frame takes about 15 ms (around 67 frames per
second). The full sequence at stride 2 (1,244 frames) maps in about 18 s of GPU time, with an RMS
absolute trajectory error of 6.8 cm over the 22 m path.

## Event stream

`GET /api/run` returns one JSON object per line:

| Event | Fields |
| --- | --- |
| `start` | `sequence`, `frames`, `stride`, `device`, `intrinsics`; `replay: true` when replaying |
| `frame` | `i`, `pose` (row-major 4×4 camera-to-map), `fitness`, `ms`, and every few frames `points` and a JPEG `thumb` |
| `done` | `mesh` (`url`, `vertices`, `triangles`), `ate_cm`, `path_m`, aligned `ground_truth` positions, timings |

Points are base64: positions as little-endian int16 millimetres, colours as uint8 RGB. The mesh at
`mesh.url` is one binary blob: uint32 vertex and triangle counts, int16 millimetre positions, uint8
RGB colours, and uint32 triangle indices, each section padded to 4 bytes. The extraction steps:

- **Marching cubes:** on this sequence it yields about 5M triangles.
- **Simplification:** 2 cm vertex clustering brings that to a few hundred thousand, which a browser
  renders smoothly.
- **Cleanup:** small disconnected fragments are dropped.

## Notes

- **One run at a time.** The SLAM state lives on one GPU. A visitor who arrives mid-run gets a
  replay of the last completed run, labelled as such, instead of waiting.
- **No loop closure.** Open3D's dense SLAM tracks frame to model but never revisits its past
  poses, so error grows with distance. On this sequence it stays at a few centimetres over 22 m.
  Skip more frames (`--stride 4`) and the error grows several-fold, because the camera moves too
  far between frames for ICP to stay locked on.
- **Open3D's CUDA build** needs `libcudart.so.12`, which comes from the `nvidia-cuda-runtime-cu12`
  wheel and is added to `LD_LIBRARY_PATH` at startup. It also needs EGL, OpenMP, libusb and TBB
  from apt, even for headless use.

## License and credits

The code is Apache-2.0. [Open3D](https://www.open3d.org) is MIT. The RGB-D sequence is from the
[TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) by Sturm et al.,
licensed CC BY 4.0.

## Local authentication

The SDK reads credentials, API host, and the selected environment from your existing
`chalk login` in `~/.chalk.yml`, including project-scoped and default logins.
For unattended runs, `CHALK_CLIENT_ID` and `CHALK_CLIENT_SECRET` remain supported.
The examples use the current environment selected by your Chalk login.
