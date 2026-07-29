import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import prepare_rmbench_mm_video_gen as converter


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_discover_accepts_single_dataset_or_parent(tmp_path):
    single = tmp_path / "battery_try"
    _write_json(single / "meta" / "info.json", {})

    assert converter.discover_dataset_roots(single) == [single.resolve()]
    assert converter.discover_dataset_roots(tmp_path) == [single.resolve()]


def test_language_annotations_preserve_adjacent_identical_segments():
    annotations = {
        "episode_3": [
            ["move the red block", 2],
            ["move the red block", 3],
        ]
    }

    segments = converter.segments_from_language_annotations(annotations, 3, 5)

    assert segments == [
        converter.Segment(0, 1, "move the red block", 0),
        converter.Segment(2, 4, "move the red block", 1),
    ]


def test_language_annotation_length_must_match_episode():
    with pytest.raises(ValueError, match="cover 2 frames"):
        converter.segments_from_language_annotations(
            {"episode_0": [["pick up the block", 2]]},
            episode_id=0,
            episode_length=3,
        )


def test_recover_segments_uses_text_changes_and_end_window_transitions():
    rows = [
        {"subtask": "A", "subtask_end": False},
        {"subtask": "A", "subtask_end": True},
        {"subtask": "A", "subtask_end": False},
        {"subtask": "B", "subtask_end": True},
    ]

    assert converter.recover_segments_from_rows(rows) == [
        converter.Segment(0, 1, "A", 0),
        converter.Segment(2, 2, "A", 1),
        converter.Segment(3, 3, "B", 2),
    ]


def test_split_segment_respects_two_fps_frame_budget():
    segment = converter.Segment(0, 300, "task", 0)

    clips = converter.split_segment(segment, fps=30, max_num_frame=5)

    assert clips == [(0, 60, 0), (60, 120, 1), (120, 180, 2), (180, 240, 3), (240, 300, 4)]
    for start, end, _ in clips:
        assert converter.math.ceil(((end - start) / 30) * 2) + 1 <= 5


def test_resolve_video_reference_supports_shared_video_metadata(tmp_path):
    dataset_root = tmp_path / "battery_try"
    video_path = dataset_root / "videos" / "chunk-002" / converter.HEAD_CAMERA_KEY / "file-004.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()
    info = {
        "chunks_size": 1000,
        "video_path": "videos/chunk-{video_chunk:03d}/{video_key}/file-{video_file:03d}.mp4",
    }
    episode_meta = {
        "videos": {
            converter.HEAD_CAMERA_KEY: {
                "video_chunk_index": 2,
                "video_file_index": 4,
                "from_timestamp": 12.5,
                "to_timestamp": 20.0,
            }
        }
    }

    result = converter.resolve_video_reference(
        dataset_root,
        info,
        episode_meta,
        episode_index=1234,
        video_key=converter.HEAD_CAMERA_KEY,
    )

    assert result == converter.VideoReference(video_path, 12.5, 20.0)


def test_build_samples_emits_training_schema_and_provenance(tmp_path):
    video_path = tmp_path / "episode.mp4"
    video_path.touch()
    stats = converter.ConversionStats()

    samples = converter.build_samples_for_episode(
        rows=[{"global_task": "Arrange the blocks."}] * 4,
        segments=[converter.Segment(0, 3, "Pick up the red block.", 2)],
        video=converter.VideoReference(video_path),
        video_path_for_json="battery_try/videos/episode.mp4",
        task_name="battery_try",
        episode_index=5,
        episode_id=8,
        fps=30,
        max_num_frame=128,
        stats=stats,
    )

    assert samples == [
        {
            "video": "battery_try/videos/episode.mp4",
            "clip": [0.0, 0.1],
            "conversations": [
                {
                    "from": "human",
                    "value": "Global task: Arrange the blocks.\nCurrent subtask: Pick up the red block.",
                },
                {"from": "gpt", "value": ""},
            ],
            "task": "battery_try",
            "episode_id": 8,
            "subtask_index": 2,
            "clip_index": 0,
        }
    ]


def test_main_generates_jsonl_and_meta(monkeypatch, tmp_path):
    lerobot_root = tmp_path / "lerobot_datasets"
    dataset_root = lerobot_root / "battery_try"
    info = {
        "fps": 30,
        "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            converter.HEAD_CAMERA_KEY: {"dtype": "video"},
            "subtask": {"dtype": "string"},
            "global_task": {"dtype": "string"},
            "subtask_end": {"dtype": "bool"},
        },
    }
    _write_json(dataset_root / "meta" / "info.json", info)
    parquet_path = dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True)
    parquet_path.touch()
    video_path = dataset_root / "videos" / "chunk-000" / converter.HEAD_CAMERA_KEY / "episode_000000.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()

    rmbench_root = tmp_path / "RMBench" / "data"
    _write_json(
        rmbench_root / "battery_try" / "demo_clean" / "language_annotation.json",
        {"episode_7": [["First move", 2], ["Second move", 2]]},
    )

    rows = [
        {
            "episode_index": 0,
            "episode_id": 7,
            "frame_index": index,
            "subtask": "First move" if index < 2 else "Second move",
            "global_task": "Complete the battery task.",
            "subtask_end": index in (1, 3),
        }
        for index in range(4)
    ]

    class FakeTable:
        def to_pylist(self):
            return rows

    fake_parquet = SimpleNamespace(
        read_schema=lambda _path: SimpleNamespace(names=list(rows[0])),
        read_table=lambda _path, columns: FakeTable(),
    )
    monkeypatch.setattr(converter, "_import_pyarrow_parquet", lambda: fake_parquet)

    output_jsonl = tmp_path / "output" / "annotations.jsonl"
    output_meta = tmp_path / "output" / "meta.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_rmbench_mm_video_gen.py",
            "--lerobot-root",
            str(lerobot_root),
            "--rmbench-data-root",
            str(rmbench_root),
            "--output-jsonl",
            str(output_jsonl),
            "--output-meta",
            str(output_meta),
            "--max-num-frame",
            "128",
        ],
    )

    assert converter.main() == 0

    samples = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(samples) == 2
    assert samples[0]["clip"] == [0.0, 0.03333333]
    assert samples[1]["clip"] == [0.06666667, 0.1]
    assert all(sample["conversations"][1] == {"from": "gpt", "value": ""} for sample in samples)

    meta = json.loads(output_meta.read_text(encoding="utf-8"))
    assert meta["rmbench_mm_video_gen"] == {
        "root": str(lerobot_root.resolve()),
        "annotation": str(output_jsonl.resolve()),
        "repeat_time": 1.0,
        "length": 2,
        "task": "video_gen",
        "data_type": "rmbench_lerobot_mm_video_gen",
    }
