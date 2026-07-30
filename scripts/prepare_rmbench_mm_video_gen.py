#!/usr/bin/env python3
"""Build SenseNova-U1 mm_video_gen JSONL annotations from raw RMBench tasks."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

EPISODE_FILE_PATTERN = re.compile(r"^episode(\d+)$")
EPISODE_ANNOTATION_PATTERN = re.compile(r"^episode_(\d+)$")
DEMO_DIR_PATTERN = re.compile(r"^demo_clean(?:_200)?$")


@dataclass(frozen=True)
class Segment:
    """An inclusive frame interval with its subtask instruction."""

    start_frame: int
    end_frame: int
    text: str
    subtask_index: int


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int


@dataclass
class ConversionStats:
    tasks: int = 0
    demos: int = 0
    episodes: int = 0
    samples: int = 0
    skipped_episodes: int = 0


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def is_demo_root(path: Path) -> bool:
    return (
        path.is_dir()
        and DEMO_DIR_PATTERN.fullmatch(path.name) is not None
        and (path / "video").is_dir()
        and (path / "instructions").is_dir()
        and (path / "language_annotation.json").is_file()
    )


def discover_demo_roots(task_root: Path) -> list[Path]:
    demo_roots = sorted(child for child in task_root.iterdir() if is_demo_root(child))
    if not demo_roots:
        raise ValueError(f"{task_root} contains neither demo_clean nor demo_clean_200 data")
    return demo_roots


def is_task_root(path: Path) -> bool:
    return path.is_dir() and any(is_demo_root(child) for child in path.iterdir())


def discover_task_roots(rmbench_root: Path) -> list[Path]:
    """Accept one RMBench task directory or a parent containing task directories."""

    root = rmbench_root.resolve()
    if is_task_root(root):
        return [root]

    task_roots = sorted(child for child in root.iterdir() if child.is_dir() and is_task_root(child))
    if not task_roots:
        raise ValueError(f"{root} is neither an RMBench task nor a directory containing RMBench tasks")
    return task_roots


def discover_episode_videos(demo_root: Path) -> dict[int, Path]:
    videos = {}
    video_root = demo_root / "video"
    for path in video_root.glob("episode*.mp4"):
        match = EPISODE_FILE_PATTERN.fullmatch(path.stem)
        if match:
            videos[int(match.group(1))] = path
    if not videos:
        raise ValueError(f"no episodeN.mp4 files found under {video_root}")
    return dict(sorted(videos.items()))


def load_language_annotations(demo_root: Path) -> dict[str, Any]:
    annotation_path = demo_root / "language_annotation.json"
    annotations = load_json(annotation_path)
    if not isinstance(annotations, dict):
        raise ValueError(f"expected a JSON object in {annotation_path}")
    return annotations


def validate_annotation_episodes(annotations: dict[str, Any], videos: dict[int, Path]) -> None:
    annotation_ids = set()
    invalid_keys = []
    for key in annotations:
        match = EPISODE_ANNOTATION_PATTERN.fullmatch(key)
        if match:
            annotation_ids.add(int(match.group(1)))
        else:
            invalid_keys.append(key)

    if invalid_keys:
        raise ValueError(f"invalid episode keys in language_annotation.json: {sorted(invalid_keys)}")

    video_ids = set(videos)
    missing_annotations = sorted(video_ids - annotation_ids)
    missing_videos = sorted(annotation_ids - video_ids)
    if missing_annotations or missing_videos:
        raise ValueError(
            "videos and language annotations do not match: "
            f"missing annotations={missing_annotations}, missing videos={missing_videos}"
        )


def load_global_task(demo_root: Path, episode_id: int, rng: random.Random) -> str:
    """Choose one non-empty global task from instructions/episodeN.json's seen list."""

    instruction_path = demo_root / "instructions" / f"episode{episode_id}.json"
    data = load_json(instruction_path)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {instruction_path}")

    seen = data.get("seen")
    if not isinstance(seen, list):
        raise ValueError(f"{instruction_path} must contain a seen list")

    choices = [text.strip() for text in seen if isinstance(text, str) and text.strip()]
    if not choices:
        raise ValueError(f"{instruction_path} contains no non-empty text in seen")
    return rng.choice(choices)


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "reading RMBench video metadata requires opencv-python; "
            "install it with `python -m pip install opencv-python`"
        ) from exc
    return cv2


def probe_video(video_path: Path, fps_override: float | None = None) -> VideoInfo:
    """Read FPS and frame count without decoding the full video."""

    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"failed to open video: {video_path}")
        detected_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()

    fps = float(fps_override) if fps_override is not None else detected_fps
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{video_path} has invalid FPS {fps}")
    if frame_count < 2:
        raise ValueError(f"{video_path} must contain at least two frames, got {frame_count}")
    return VideoInfo(fps=fps, frame_count=frame_count)


def segments_from_language_annotations(
    annotations: dict[str, Any],
    episode_id: int,
    episode_length: int,
) -> list[Segment]:
    """Convert one episode's [subtask, duration] entries into frame intervals."""

    episode_key = f"episode_{episode_id}"
    raw_segments = annotations.get(episode_key)
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"{episode_key} has no language segments")

    segments = []
    current_frame = 0
    for subtask_index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, list) or len(raw_segment) != 2:
            raise ValueError(f"invalid segment {raw_segment!r} in {episode_key}")

        text, duration = raw_segment
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"segment {subtask_index} in {episode_key} has an empty instruction")
        if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
            raise ValueError(f"segment {subtask_index} in {episode_key} has invalid duration {duration!r}")

        end_frame = current_frame + duration - 1
        segments.append(Segment(current_frame, end_frame, text.strip(), subtask_index))
        current_frame = end_frame + 1

    if current_frame != episode_length:
        raise ValueError(
            f"{episode_key} annotations cover {current_frame} frames, but the video contains {episode_length}"
        )
    return segments


def build_human_prompt(global_task: str, subtask: str) -> str:
    return f"Global task: {global_task}\nCurrent subtask: {subtask}"


def build_samples_for_episode(
    *,
    segments: list[Segment],
    video_path_for_json: str,
    task_name: str,
    demo_name: str,
    episode_id: int,
    global_task: str,
    video_info: VideoInfo,
) -> list[dict[str, Any]]:
    samples = []
    for segment in segments:
        samples.append(
            {
                "video": video_path_for_json,
                "clip": [
                    round(segment.start_frame / video_info.fps, 8),
                    round(segment.end_frame / video_info.fps, 8),
                ],
                "conversations": [
                    {
                        "from": "human",
                        "value": build_human_prompt(global_task, segment.text),
                    },
                    {"from": "gpt", "value": ""},
                ],
                "task": task_name,
                "demo": demo_name,
                "episode_id": episode_id,
                "subtask_index": segment.subtask_index,
                "clip_index": 0,
            }
        )
    return samples


def convert_task(
    *,
    task_root: Path,
    rmbench_root: Path,
    rng: random.Random,
    fps_override: float | None,
    skip_invalid: bool,
    stats: ConversionStats,
) -> list[dict[str, Any]]:
    task_name = task_root.name
    task_samples = []
    for demo_root in discover_demo_roots(task_root):
        videos = discover_episode_videos(demo_root)
        annotations = load_language_annotations(demo_root)
        validate_annotation_episodes(annotations, videos)
        stats.demos += 1

        for episode_id, video_path in videos.items():
            try:
                video_info = probe_video(video_path, fps_override)
                global_task = load_global_task(demo_root, episode_id, rng)
                segments = segments_from_language_annotations(annotations, episode_id, video_info.frame_count)
                task_samples.extend(
                    build_samples_for_episode(
                        segments=segments,
                        video_path_for_json=video_path.relative_to(rmbench_root).as_posix(),
                        task_name=task_name,
                        demo_name=demo_root.name,
                        episode_id=episode_id,
                        global_task=global_task,
                        video_info=video_info,
                    )
                )
                stats.episodes += 1
            except Exception as exc:
                if not skip_invalid:
                    raise
                stats.skipped_episodes += 1
                print(f"warning: skipping {task_name}/{demo_root.name} episode {episode_id}: {exc}")

    stats.tasks += 1
    stats.samples += len(task_samples)
    return task_samples


def write_jsonl_atomic(path: Path, samples: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            for sample in samples:
                file.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SenseNova-U1 mm_video_gen JSONL from raw RMBench tasks.")
    parser.add_argument(
        "--rmbench-root",
        type=Path,
        required=True,
        help="An RMBench task directory, or the parent directory containing multiple tasks.",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-meta", type=Path, required=True)
    parser.add_argument(
        "--fps",
        type=float,
        help="Override video FPS; by default each MP4's metadata is used.",
    )
    parser.add_argument("--seed", type=int, help="Seed for choosing a global task from each seen list.")
    parser.add_argument("--repeat-time", type=float, default=1.0)
    parser.add_argument("--dataset-name", default="rmbench_mm_video_gen")
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid episodes instead of stopping at the first error.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rmbench_root = args.rmbench_root.resolve()
    output_jsonl = args.output_jsonl.resolve()
    output_meta = args.output_meta.resolve()

    if not rmbench_root.is_dir():
        raise ValueError(f"--rmbench-root is not a directory: {rmbench_root}")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        raise ValueError("--fps must be a positive finite number")
    if args.repeat_time <= 0:
        raise ValueError("--repeat-time must be positive")
    if not args.dataset_name:
        raise ValueError("--dataset-name cannot be empty")
    if output_jsonl == output_meta:
        raise ValueError("--output-jsonl and --output-meta must be different files")

    stats = ConversionStats()
    samples = []
    rng = random.Random(args.seed)
    for task_root in discover_task_roots(rmbench_root):
        samples.extend(
            convert_task(
                task_root=task_root,
                rmbench_root=rmbench_root,
                rng=rng,
                fps_override=args.fps,
                skip_invalid=args.skip_invalid,
                stats=stats,
            )
        )

    if not samples:
        raise ValueError("conversion produced no valid mm_video_gen samples")

    write_jsonl_atomic(output_jsonl, samples)
    write_json_atomic(
        output_meta,
        {
            args.dataset_name: {
                "root": str(rmbench_root),
                "annotation": str(output_jsonl),
                "repeat_time": args.repeat_time,
                "length": len(samples),
                "task": "video_gen",
                "data_type": "rmbench_mm_video_gen",
            }
        },
    )

    print(
        "completed: "
        f"tasks={stats.tasks}, demos={stats.demos}, episodes={stats.episodes}, samples={stats.samples}, "
        f"skipped_episodes={stats.skipped_episodes}"
    )
    print(f"JSONL: {output_jsonl}")
    print(f"meta:  {output_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
