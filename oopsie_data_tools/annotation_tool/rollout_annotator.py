"""Minimal rollout-side client for the minimal annotation server.

The rollout annnotator is responsible for:
- waiting for a language instruction
- recording step data
- after rollout: saving MP4s, show them in the web UI, wait for Save
- write annotation into episode HDF5 via EpisodeRecorder (failure_annotation attr)
"""

from __future__ import annotations

import atexit
import datetime
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

from oopsie_data_tools.annotation_tool.episode_recorder import EpisodeRecorder
from oopsie_data_tools.utils.robot_profile.robot_profile import RobotProfile

logger = logging.getLogger(__name__)


class WebRolloutAnnotator:
    def __init__(
        self,
        robot_profile: RobotProfile,
        data_root_dir: Path | str,
        operator_name: str,
        annotator_name: str | None = None,
        port: int = 5001,
        wait_for_annotation: bool = True,
        open_browser: bool = True,
        resume_session_name: str | None = None,
    ) -> None:
        self.robot_profile = robot_profile
        self.data_root_dir = Path(data_root_dir)
        self.data_root_dir = self.data_root_dir.resolve()
        self.operator_name = operator_name.strip()
        self.annotator_name = annotator_name.strip() if annotator_name is not None else self.operator_name
        self.port = port
        self.wait_for_annotation = wait_for_annotation
        self.open_browser = open_browser
        self._proc: subprocess.Popen | None = None
        self._active_recorder = EpisodeRecorder(
            robot_profile=self.robot_profile,
            data_root_dir=self.data_root_dir,
            resume_session_name=resume_session_name,
            operator_name=self.operator_name,
        )

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start(self, startup_timeout_s: float = 30.0) -> None:
        """Start the annotation server unless something is already on the port.

        Raises:
            RuntimeError: If the server does not come up within ``startup_timeout_s``.
                Continuing regardless would leave the rollout loop polling an address that
                will never answer, with no output to explain why.
        """
        self.data_root_dir.mkdir(parents=True, exist_ok=True)

        if not self._is_port_open():
            self._spawn_server(startup_timeout_s)

        if self.open_browser:
            webbrowser.open(f"http://localhost:{self.port}/")

    def _spawn_server(self, startup_timeout_s: float) -> None:
        self._proc = subprocess.Popen(
            [
                # Not "python": inside a venv that may not exist, or may be a different
                # interpreter than the one running this rollout.
                sys.executable,
                "-m",
                "oopsie_data_tools.annotation_tool.annotator_server",
                "--samples-dir",
                str(self.data_root_dir),
                "--port",
                str(self.port),
                "--annotator-name",
                self.annotator_name,
                "--with-rollouts",
                "--no-browser",
            ],
            # stderr is kept so a failure to start is visible; discarding it was why a
            # server that died on startup looked identical to one that was merely slow.
            stdout=subprocess.DEVNULL,
        )
        atexit.register(self.stop)

        deadline = time.monotonic() + startup_timeout_s
        while time.monotonic() < deadline:
            if self._is_port_open():
                break
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"Annotation server exited immediately with code {self._proc.returncode}. "
                    "Its error output is above."
                )
            time.sleep(0.25)
        else:
            self.stop()
            raise RuntimeError(
                f"Annotation server did not start on port {self.port} within "
                f"{startup_timeout_s:g}s. Pass a larger startup_timeout_s if this machine is "
                "slow, or check whether the port is taken by something else."
            )

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            # Reap it, so a long rollout loop does not leave zombies behind.
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5)
        self._proc = None

    def _is_port_open(self) -> bool:
        with socket.socket() as s:
            return s.connect_ex(("localhost", self.port)) == 0

    # ------------------------------------------------------------------
    # Task coordination
    # ------------------------------------------------------------------

    def wait_for_task(self) -> str:
        """Block until the browser submits a language instruction, then claim it.

        Polling errors are expected and transient (the server may still be starting), so
        they are logged rather than raised — but they are no longer swallowed silently: a
        server that has died used to leave this loop spinning forever with no output at all.
        """
        consecutive_failures = 0
        while True:
            try:
                state = self._api_get("/api/task/state")
                consecutive_failures = 0
                if (
                    isinstance(state, dict)
                    and state.get("status") == "pending"
                    and state.get("pending_instruction")
                ):
                    instruction = str(state["pending_instruction"])
                    self._api_post("/api/task/start", {"instruction": instruction})
                    return instruction
            except Exception as e:
                consecutive_failures += 1
                # ~5s of failures, then every ~30s, so a dead server is visible without
                # flooding the console during a normal wait.
                if consecutive_failures == 10 or consecutive_failures % 60 == 0:
                    logger.warning(
                        "Cannot reach the annotation server at http://localhost:%s (%s). "
                        "Still waiting; is it still running?",
                        self.port,
                        e,
                    )
            time.sleep(0.5)

    # ------------------------------------------------------------------
    # Optional recording passthrough
    # ------------------------------------------------------------------

    def reset_episode_recorder(self) -> None:
        if self._active_recorder is not None:
            self._active_recorder.reset_episode_recorder()

    def record_step(self, observation: dict[str, Any], action: dict[str, Any]) -> None:
        if self._active_recorder is not None:
            self._active_recorder.record_step(observation=observation, action=action)

    # ------------------------------------------------------------------
    # Post-rollout workflow
    # ------------------------------------------------------------------

    def finish_rollout(
        self,
        instruction: str,
        recorder: EpisodeRecorder | None = None,
    ) -> dict[str, Any] | None:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Instruction must be a non-empty string, got {instruction!r}")

        # Every use below must go through active_recorder. Three of them used to reach for
        # self._active_recorder instead, so passing recorder= validated and stamped the
        # wrong object — and raised AttributeError before reaching the guard right below
        # whenever self._active_recorder was None, which is the case that guard is for.
        active_recorder = recorder or self._active_recorder
        if active_recorder is None:
            raise ValueError("EpisodeRecorder is required (pass recorder=...).")
        sample_id = active_recorder.save_fname

        data = {
            "language_instruction": instruction,
            "metadata": {
                "episode_id": sample_id,
                "operator_name": self.operator_name,
            },
        }
        # Raises if invalid, so an unwritable episode is never saved.
        active_recorder._validate_pre_save(data)

        # 1. Save videos under the recorder's per-session folder
        video_paths = active_recorder._save_videos()
        video_urls = {
            cam: self._video_url_from_abs_path(Path(p))
            for cam, p in video_paths.items()
        }
        data["video_paths"] = video_paths

        # 2. Save the episode HDF5 straight away, before the annotation arrives, and patch
        # the annotation in later. The file is not submittable until then, but writing it
        # now lets the UI show the videos without waiting for the human.
        h5_path = active_recorder.save(data)

        # 3. Wait for annotation from human annotator
        logger.info(
            "Waiting for annotation from human annotator at http://localhost:%s ...", self.port
        )
        self._api_post(
            "/api/task/annotating",
            {
                "sample_id": sample_id,
                "video_urls": video_urls,
                "language_instruction": instruction,
            },
        )
        annotation: dict[str, Any] | None = None
        if self.wait_for_annotation:
            annotation = self._wait_for_annotation(sample_id)
        # Reset UI to idle regardless of save/write outcome
        self._api_post("/api/task/done", {})

        # 4. Write annotation into episode HDF5 in-place
        if annotation:
            # Mirror the old behavior of also writing annotation into a sidecar JSON
            # to keep it co-located with the MP4s.
            sample_json = active_recorder.session_dir / f"{sample_id}.json"
            metadata = {
                "language_instruction": instruction,
                "camera_names": list(video_paths.keys()),
                "timestamp": datetime.datetime.now().isoformat(),
                "annotation": {
                    k: v
                    for k, v in annotation.items()
                    if k not in {"annotated_at", "annotator", "source"}
                },
            }
            fd, tmp = tempfile.mkstemp(
                dir=active_recorder.session_dir, suffix=".json.tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                Path(tmp).replace(sample_json)
            except Exception:
                try:
                    Path(tmp).unlink(missing_ok=True)
                except Exception:
                    pass
                raise

            # Patch the already-saved episode HDF5 in-place
            EpisodeRecorder.patch_h5_failure_annotation(Path(h5_path), annotation)

        return annotation

    def _wait_for_annotation(self, sample_id: str) -> dict[str, Any]:
        while True:
            data = self._api_get(f"/api/annotations/{sample_id}")
            if not isinstance(data, dict):
                time.sleep(0.5)
                continue
            if data.get("__annotation_skipped__"):
                return {}
            inner = (
                data.get("annotation")
                if isinstance(data.get("annotation"), dict)
                else None
            )
            if inner and inner.get("__annotation_skipped__"):
                return {}
            if data:
                # minimal server returns the annotation dict directly
                return inner if inner is not None else data
            time.sleep(0.5)

    def _video_url_from_abs_path(self, abs_path: Path) -> str:
        rel = abs_path.resolve().relative_to(self.data_root_dir.resolve()).as_posix()
        return f"/videos-path/{urllib.parse.quote(rel, safe='/')}"

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _api_get(self, path: str) -> Any:
        url = f"http://localhost:{self.port}{path}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read())

    def _api_post(self, path: str, data: dict[str, Any]) -> Any:
        url = f"http://localhost:{self.port}{path}"
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
