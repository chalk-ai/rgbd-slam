#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Deploy live dense RGB-D SLAM (server.py + slam.py + index.html) as a GPU ScalingGroup.

The image bakes in the TUM RGB-D `fr3/long_office_household` sequence, so a run reads frames
from local disk and the GPU is the only bottleneck.

Credentials come from your existing `chalk login` (or the CHALK_* env vars):
    ./deploy.py [--env-id <environment>] [--gpu nvidia-l4] [--stride 2]
Then open the printed URL and press "Start mapping".
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEQUENCE = "rgbd_dataset_freiburg3_long_office_household"
SEQUENCE_URL = f"https://cvg.cit.tum.de/rgbd/dataset/freiburg3/{SEQUENCE}.tgz"

# Open3D's CUDA build loads libcudart from the pip wheel, which is not on the loader path.
# nvidia.* are namespace packages (no __file__), so the path comes from __path__.
ENTRYPOINT = [
    "bash",
    "-c",
    "libs=$(python -c 'import nvidia.cuda_runtime.lib as l; print(l.__path__[0])') || exit 1; "
    + "export LD_LIBRARY_PATH=$libs; exec python /app/server.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="rgbd-slam")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--gpu", default="nvidia-l4")
    # Every 2nd frame (15 fps) tracks the full loop without losing the camera.
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--ready-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    if args.env_id:
        os.environ["CHALK_ENVIRONMENT"] = args.env_id
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    image = (
        Image.debian_slim(python_version="3.12")
        # Open3D's Python module links against EGL/GL, OpenMP, libusb and TBB.
        .apt_install(["libegl1", "libgl1", "libgomp1", "libusb-1.0-0", "libtbb12"])
        .pip_install(
            [
                "open3d==0.20.0",
                "nvidia-cuda-runtime-cu12==12.*",
                "numpy<3",
                "pillow==11.*",
                "fastapi==0.116.1",
                "uvicorn[standard]==0.35.0",
            ]
        )
        # Stream the 1.5 GB sequence straight into the image so a replica never downloads it.
        .run_commands(
            'mkdir -p /data && python -c "import tarfile, urllib.request; '
            + f"tarfile.open(fileobj=urllib.request.urlopen('{SEQUENCE_URL}'), mode='r|gz').extractall('/data')\""
        )
        # libOpen3D also links libidn2, which python:*-slim does not ship. It sits in its own
        # layer after the sequence download so adding it does not re-download 1.5 GB.
        .apt_install(["libidn2-0"])
        .add_local_file(str(HERE / "slam.py"), "/app/slam.py", strategy="copy")
        .add_local_file(str(HERE / "server.py"), "/app/server.py", strategy="copy")
        # strategy="copy" caps the image's copied files at 32 KiB in total, so the viewer goes in
        # through a volume. It mounts its own directory, so it cannot shadow the copied /app.
        .add_local_file(str(HERE / "index.html"), "/web/index.html", strategy="volume")
        .workdir("/app")
    )

    sg = ScalingGroup(
        image=image,
        name=args.name,
        gpu=args.gpu,
        cpu="4",
        memory="16Gi",
        env={
            "SLAM_SEQUENCE_DIR": f"/data/{SEQUENCE}",
            "SLAM_STRIDE": str(args.stride),
            "SLAM_INDEX_HTML": "/web/index.html",
        },
        entrypoint=ENTRYPOINT,
        port=8080,
        readiness_probe=ReadinessProbe.http("/api/status"),
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    try:
        sg.deploy(build_timeout=2400.0, ready_timeout=args.ready_timeout)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
