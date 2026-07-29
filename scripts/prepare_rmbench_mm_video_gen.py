#!/usr/bin/env python3
"""Build SenseNova-U1 mm_video_gen JSONL annotations from RMBench LeRobot datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

HEAD_CAMERA_KEY = "observation.image.head_camera"
PARQUET_COLUMNS = {
    "episode_index",
    "episode_id",
    "frame_index",
    "timestamp",
    "subtask",
    "global_task",
    "subtask_end",
}


@dataclass(frozen=True)
class Segment:
    """An inclusive frame interval with its language instruction."""

    start_frame: int
    end_frame: int
    text: str
    subtask_index: int


@dataclass(frozen=True)
class VideoReference:
    """A physical LeRobot video and the episode's offset within it."""

    path: Path
    timestamp_offset: float = 0.0
    timestamp_end: float | None = None


@dataclass
class ConversionStats:
    tasks: int = 0
    episodes: int = 0
    samples: int = 0
    skipped_episodes: int = 0
    skipped_single_frame_segments: int = 0
    fallback_segmented_episodes: int = 0


def _scalar(value: Any) -> Any:
    """Unwrap common one-element containers emitted by Arrow/NumPy."""

    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _scalar(value[0])
    return value


def _text(value: Any) -> str:
    value = _scalar(value)
    return "" if value is None else str(value)


def discover_dataset_roots(lerobot_root: Path) -> list[Path]:
    """Accept either one LeRobot dataset or a parent containing task datasets."""

    root = lerobot_root.resolve()
    if (root / "meta" / "info.json").is_file():
        return [root]

    dataset_roots = sorted(
        child for child in root.iterdir() if child.is_dir() and (child / "meta" / "info.json").is_file()
    )
    if not dataset_roots:
        raise ValueError(f"{root} is neither a LeRobot dataset nor a directory containing LeRobot task datasets")
    return dataset_roots


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def load_jsonl_by_episode(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}

    episodes: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "episode_index" not in item:
                raise ValueError(f"{path}:{line_number} has no episode_index")
            episodes[int(_scalar(item["episode_index"]))] = item
    return episodes


def select_video_key(info: dict[str, Any]) -> str:
    features = info.get("features", {})
    video_keys = [key for key, spec in features.items() if isinstance(spec, dict) and spec.get("dtype") == "video"]
    if HEAD_CAMERA_KEY in video_keys:
        return HEAD_CAMERA_KEY
    head_camera_candidates = [key for key in video_keys if "head_camera" in key]
    if len(head_camera_candidates) == 1:
        return head_camera_candidates[0]
    raise ValueError(f"could not uniquely identify the head-camera video feature; found {video_keys}")


def _import_pyarrow_parquet():
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "reading LeRobot Parquet metadata requires pyarrow; install it with `python -m pip install pyarrow`"
        ) from exc
    return parquet


def read_episode_rows(dataset_root: Path) -> dict[int, list[dict[str, Any]]]:
    """Read only the columns needed to create video-generation annotations."""

    parquet_files = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise ValueError(f"no Parquet files found under {dataset_root / 'data'}")

    parquet = _import_pyarrow_parquet()
    episodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for parquet_path in parquet_files:
        schema_names = set(parquet.read_schema(parquet_path).names)
        columns = sorted(PARQUET_COLUMNS.intersection(schema_names))
        if "episode_index" not in columns and "episode_id" not in columns:
            raise ValueError(f"{parquet_path} has neither episode_index nor episode_id")
        if "subtask" not in columns:
            raise ValueError(f"{parquet_path} has no subtask column")

        table = parquet.read_table(parquet_path, columns=columns)
        for row in table.to_pylist():
            episode_index = int(_scalar(row.get("episode_index", row.get("episode_id"))))
            episodes[episode_index].append(row)

    for rows in episodes.values():
        rows.sort(
            key=lambda row: (
                int(_scalar(row["frame_index"]))
                if row.get("frame_index") is not None
                else float(_scalar(row.get("timestamp", 0)))
            )
        )
    return dict(episodes)


def segments_from_language_annotations(
    annotations: dict[str, Any],
    episode_id: int,
    episode_length: int,
) -> list[Segment]:
    episode_key = f"episode_{episode_id}"
    if episode_key not in annotations:
        raise ValueError(f"language annotations have no {episode_key}")

    raw_segments = annotations[episode_key]
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"{episode_key} has no language segments")

    segments = []
    current_frame = 0
    for subtask_index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, (list, tuple)) or len(raw_segment) != 2:
            raise ValueError(f"invalid segment {raw_segment!r} in {episode_key}")
        text, duration = raw_segment
        duration = int(duration)
        if duration <= 0:
            raise ValueError(f"segment {subtask_index} in {episode_key} has invalid duration {duration}")
        text = _text(text).strip()
        if not text:
            raise ValueError(f"segment {subtask_index} in {episode_key} has an empty instruction")

        end_frame = current_frame + duration - 1
        segments.append(Segment(current_frame, end_frame, text, subtask_index))
        current_frame = end_frame + 1

    if current_frame != episode_length:
        raise ValueError(
            f"{episode_key} annotations cover {current_frame} frames, but LeRobot contains {episode_length}"
        )
    return segments


def recover_segments_from_rows(rows: list[dict[str, Any]]) -> list[Segment]:
    """Best-effort recovery from the end-window signal stored in LeRobot."""

    if not rows:
        return []

    texts = [_text(row.get("subtask")).strip() for row in rows]
    if any(not text for text in texts):
        raise ValueError("episode contains frames with an empty subtask")
    end_flags = [bool(_scalar(row.get("subtask_end", False))) for row in rows]

    segments = []
    start = 0
    for index in range(len(rows) - 1):
        text_changes = texts[index] != texts[index + 1]
        leaves_end_window = end_flags[index] and not end_flags[index + 1]
        if text_changes or leaves_end_window:
            segments.append(Segment(start, index, texts[start], len(segments)))
            start = index + 1
    segments.append(Segment(start, len(rows) - 1, texts[start], len(segments)))
    return segments


def split_segment(segment: Segment, fps: float, max_num_frame: int) -> list[tuple[int, int, int]]:
    """Split an interval so both 1 FPS and 2 FPS training sampling remain valid."""

    if max_num_frame < 2:
        raise ValueError(f"max_num_frame must be at least 2, got {max_num_frame}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid dataset FPS: {fps}")
    if segment.end_frame <= segment.start_frame:
        return []

    max_frame_span = math.floor((max_num_frame - 1) * fps / 2)
    if max_frame_span < 1:
        raise ValueError(f"max_num_frame={max_num_frame} is too small for dataset FPS {fps}")

    clips = []
    clip_start = segment.start_frame
    clip_index = 0
    while clip_start < segment.end_frame:
        clip_end = min(segment.end_frame, clip_start + max_frame_span)
        clips.append((clip_start, clip_end, clip_index))
        clip_start = clip_end
        clip_index += 1
    return clips


def build_human_prompt(global_task: str, subtask: str) -> str:
    global_task = global_task.strip()
    subtask = subtask.strip()
    if global_task:
        return f"Global task: {global_task}\nCurrent subtask: {subtask}"
    return f"Task: {subtask}"


def _first_present(mapping: dict[str, Any], keys: Iterable[str], default: Any) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def resolve_video_reference(
    dataset_root: Path,
    info: dict[str, Any],
    episode_meta: dict[str, Any],
    episode_index: int,
    video_key: str,
) -> VideoReference:
    """Resolve LeRobot v2 episode videos and v3 shared video files."""

    chunk_size = int(info.get("chunks_size", info.get("chunk_size", 1000)))
    episode_chunk = episode_index // chunk_size
    videos = episode_meta.get("videos", {})
    video_meta = videos.get(video_key, {}) if isinstance(videos, dict) else {}
    if not isinstance(video_meta, dict):
        video_meta = {}

    video_chunk = int(
        _first_present(
            video_meta,
            ("video_chunk_index", "chunk_index"),
            _first_present(episode_meta, ("video_chunk_index", "chunk_index"), episode_chunk),
        )
    )
    video_file = int(
        _first_present(
            video_meta,
            ("video_file_index", "file_index"),
            _first_present(episode_meta, ("video_file_index", "file_index"), episode_index),
        )
    )
    template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    format_values = {
        "episode_chunk": episode_chunk,
        "episode_index": episode_index,
        "video_key": video_key,
        "video_chunk": video_chunk,
        "video_file": video_file,
        "chunk_index": video_chunk,
        "file_index": video_file,
    }
    try:
        relative_path = Path(str(template).format(**format_values))
    except KeyError as exc:
        raise ValueError(f"unsupported placeholder {exc} in video_path template {template!r}") from exc

    video_path = dataset_root / relative_path
    if not video_path.is_file():
        candidates = sorted((dataset_root / "videos").glob(f"**/{video_key}/episode_{episode_index:06d}.mp4"))
        if len(candidates) == 1:
            video_path = candidates[0]
        else:
            raise ValueError(f"video for episode {episode_index} does not exist: {video_path}")

    timestamp_offset = float(_first_present(video_meta, ("from_timestamp", "start_timestamp"), 0.0))
    raw_timestamp_end = _first_present(video_meta, ("to_timestamp", "end_timestamp"), None)
    timestamp_end = float(raw_timestamp_end) if raw_timestamp_end is not None else None
    return VideoReference(video_path, timestamp_offset, timestamp_end)


def build_samples_for_episode(
    *,
    rows: list[dict[str, Any]],
    segments: list[Segment],
    video: VideoReference,
    video_path_for_json: str,
    task_name: str,
    episode_index: int,
    episode_id: int,
    fps: float,
    max_num_frame: int,
    stats: ConversionStats,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError(f"episode {episode_index} contains no rows")

    global_task = _text(rows[0].get("global_task")).strip()
    samples = []
    for segment in segments:
        clips = split_segment(segment, fps, max_num_frame)
        if not clips:
            stats.skipped_single_frame_segments += 1
            continue

        for start_frame, end_frame, clip_index in clips:
            clip_start = video.timestamp_offset + start_frame / fps
            clip_end = video.timestamp_offset + end_frame / fps
            if video.timestamp_end is not None and clip_end > video.timestamp_end + 1e-6:
                raise ValueError(f"episode {episode_index} clip end {clip_end} exceeds video end {video.timestamp_end}")
            samples.append(
                {
                    "video": video_path_for_json,
                    "clip": [round(clip_start, 8), round(clip_end, 8)],
                    "conversations": [
                        {
                            "from": "human",
                            "value": build_human_prompt(global_task, segment.text),
                        },
                        {"from": "gpt", "value": ""},
                    ],
                    "task": task_name,
                    "episode_id": episode_id,
                    "subtask_index": segment.subtask_index,
                    "clip_index": clip_index,
                }
            )
    return samples


def _annotation_path(rmbench_data_root: Path | None, task_name: str) -> Path | None:
    if rmbench_data_root is None:
        return None
    return rmbench_data_root / task_name / "demo_clean" / "language_annotation.json"


def convert_task(
    *,
    dataset_root: Path,
    lerobot_root: Path,
    rmbench_data_root: Path | None,
    max_num_frame: int,
    skip_invalid: bool,
    stats: ConversionStats,
) -> list[dict[str, Any]]:
    info = load_json(dataset_root / "meta" / "info.json")
    fps = float(info.get("fps", 0))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{dataset_root} has invalid FPS {fps}")

    task_name = dataset_root.name
    video_key = select_video_key(info)
    episode_rows = read_episode_rows(dataset_root)
    episode_metadata = load_jsonl_by_episode(dataset_root / "meta" / "episodes.jsonl")

    annotation_path = _annotation_path(rmbench_data_root, task_name)
    annotations = load_json(annotation_path) if annotation_path is not None and annotation_path.is_file() else None
    is_multi_subtask = "global_task" in info.get("features", {})
    if is_multi_subtask and annotations is None:
        print(
            f"warning: {task_name} has no source language_annotation.json; "
            "recovering subtask boundaries from LeRobot end-window flags"
        )

    task_samples = []
    for episode_index in sorted(episode_rows):
        rows = episode_rows[episode_index]
        try:
            episode_id = int(_scalar(rows[0].get("episode_id", episode_index)))
            if annotations is not None and is_multi_subtask:
                segments = segments_from_language_annotations(annotations, episode_id, len(rows))
            else:
                segments = recover_segments_from_rows(rows)
                if is_multi_subtask:
                    stats.fallback_segmented_episodes += 1

            video = resolve_video_reference(
                dataset_root,
                info,
                episode_metadata.get(episode_index, {}),
                episode_index,
                video_key,
            )
            video_path_for_json = video.path.relative_to(lerobot_root).as_posix()
            samples = build_samples_for_episode(
                rows=rows,
                segments=segments,
                video=video,
                video_path_for_json=video_path_for_json,
                task_name=task_name,
                episode_index=episode_index,
                episode_id=episode_id,
                fps=fps,
                max_num_frame=max_num_frame,
                stats=stats,
            )
            task_samples.extend(samples)
            stats.episodes += 1
        except Exception as exc:
            if not skip_invalid:
                raise
            stats.skipped_episodes += 1
            print(f"warning: skipping {task_name} episode {episode_index}: {exc}")

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
    parser = argparse.ArgumentParser(
        description="Generate SenseNova-U1 mm_video_gen JSONL from RMBench LeRobot datasets."
    )
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        required=True,
        help="A LeRobot task dataset, or the parent directory containing multiple task datasets.",
    )
    parser.add_argument(
        "--rmbench-data-root",
        type=Path,
        help="Optional original RMBench data directory containing TASK/demo_clean/language_annotation.json.",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-meta", type=Path, required=True)
    parser.add_argument("--max-num-frame", type=int, default=128)
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
    lerobot_root = args.lerobot_root.resolve()
    rmbench_data_root = args.rmbench_data_root.resolve() if args.rmbench_data_root else None
    output_jsonl = args.output_jsonl.resolve()
    output_meta = args.output_meta.resolve()

    if not lerobot_root.is_dir():
        raise ValueError(f"--lerobot-root is not a directory: {lerobot_root}")
    if rmbench_data_root is not None and not rmbench_data_root.is_dir():
        raise ValueError(f"--rmbench-data-root is not a directory: {rmbench_data_root}")
    if args.max_num_frame < 2:
        raise ValueError("--max-num-frame must be at least 2")
    if args.repeat_time <= 0:
        raise ValueError("--repeat-time must be positive")
    if not args.dataset_name:
        raise ValueError("--dataset-name cannot be empty")
    if output_jsonl == output_meta:
        raise ValueError("--output-jsonl and --output-meta must be different files")

    stats = ConversionStats()
    samples = []
    for dataset_root in discover_dataset_roots(lerobot_root):
        samples.extend(
            convert_task(
                dataset_root=dataset_root,
                lerobot_root=lerobot_root,
                rmbench_data_root=rmbench_data_root,
                max_num_frame=args.max_num_frame,
                skip_invalid=args.skip_invalid,
                stats=stats,
            )
        )

    if not samples:
        raise ValueError("conversion produced no valid mm_video_gen samples")

    write_jsonl_atomic(output_jsonl, samples)
    meta = {
        args.dataset_name: {
            "root": str(lerobot_root),
            "annotation": str(output_jsonl),
            "repeat_time": args.repeat_time,
            "length": len(samples),
            "task": "video_gen",
            "data_type": "rmbench_lerobot_mm_video_gen",
        }
    }
    write_json_atomic(output_meta, meta)

    print(
        "completed: "
        f"tasks={stats.tasks}, episodes={stats.episodes}, samples={stats.samples}, "
        f"skipped_episodes={stats.skipped_episodes}, "
        f"skipped_single_frame_segments={stats.skipped_single_frame_segments}, "
        f"fallback_segmented_episodes={stats.fallback_segmented_episodes}"
    )
    print(f"JSONL: {output_jsonl}")
    print(f"meta:  {output_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
