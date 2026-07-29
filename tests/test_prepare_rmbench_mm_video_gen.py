import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import prepare_rmbench_mm_video_gen as converter


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _create_episode(task_root, episode_id, *, demo_name="demo_clean", seen=None, subtasks=None):
    demo_root = task_root / demo_name
    video = demo_root / "video" / f"episode{episode_id}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.touch()
    _write_json(
        demo_root / "instructions" / f"episode{episode_id}.json",
        {
            "seen": seen or ["Complete the global task."],
            "unseen": ["This text must not be used."],
        },
    )
    _write_json(
        demo_root / "language_annotation.json",
        {f"episode_{episode_id}": subtasks or [["Complete the subtask.", 4]]},
    )
    return video


def test_discover_accepts_single_task_or_parent(tmp_path):
    task_root = tmp_path / "battery_try"
    _create_episode(task_root, 0)

    assert converter.discover_task_roots(task_root) == [task_root.resolve()]
    assert converter.discover_task_roots(tmp_path) == [task_root.resolve()]


def test_episode_discovery_only_requires_videos(tmp_path):
    task_root = tmp_path / "battery_try"
    video = _create_episode(task_root, 7)

    assert converter.discover_episode_videos(task_root / "demo_clean") == {7: video}
    assert not (task_root / "demo_clean" / "data").exists()


def test_discovers_demo_clean_and_demo_clean_200(tmp_path):
    task_root = tmp_path / "cover_blocks"
    _create_episode(task_root, 0)
    _create_episode(task_root, 0, demo_name="demo_clean_200")

    assert converter.discover_demo_roots(task_root) == [
        task_root / "demo_clean",
        task_root / "demo_clean_200",
    ]


def test_video_and_annotation_episode_ids_must_match(tmp_path):
    task_root = tmp_path / "battery_try"
    _create_episode(task_root, 0)
    videos = converter.discover_episode_videos(task_root / "demo_clean")

    with pytest.raises(ValueError, match=r"missing annotations=\[0\], missing videos=\[1\]"):
        converter.validate_annotation_episodes({"episode_1": [["subtask", 4]]}, videos)


def test_global_task_randomly_uses_seen_only(tmp_path):
    task_root = tmp_path / "battery_try"
    _create_episode(
        task_root,
        0,
        seen=["First seen task.", "Second seen task.", "Third seen task."],
    )

    expected = random.Random(123).choice(["First seen task.", "Second seen task.", "Third seen task."])
    actual = converter.load_global_task(task_root / "demo_clean", 0, random.Random(123))

    assert actual == expected
    assert actual != "This text must not be used."


@pytest.mark.parametrize(
    "instruction_data,error",
    [
        ({"unseen": ["No seen text."]}, "must contain a seen list"),
        ({"seen": ["", "   "]}, "contains no non-empty text in seen"),
    ],
)
def test_global_task_requires_non_empty_seen_text(tmp_path, instruction_data, error):
    task_root = tmp_path / "battery_try"
    demo_root = task_root / "demo_clean"
    instruction_path = demo_root / "instructions" / "episode0.json"
    _write_json(instruction_path, instruction_data)

    with pytest.raises(ValueError, match=error):
        converter.load_global_task(demo_root, 0, random.Random(0))


def test_all_subtasks_come_from_language_annotations():
    annotations = {
        "episode_3": [
            ["First annotated subtask.", 2],
            ["Second annotated subtask.", 3],
        ]
    }

    assert converter.segments_from_language_annotations(annotations, 3, 5) == [
        converter.Segment(0, 1, "First annotated subtask.", 0),
        converter.Segment(2, 4, "Second annotated subtask.", 1),
    ]


def test_language_annotation_length_must_match_video():
    with pytest.raises(ValueError, match="video contains 3"):
        converter.segments_from_language_annotations(
            {"episode_0": [["Pick up the block.", 2]]},
            episode_id=0,
            episode_length=3,
        )


def test_split_segment_respects_two_fps_frame_budget():
    segment = converter.Segment(0, 300, "task", 0)

    clips = converter.split_segment(segment, fps=30, max_num_frame=5)

    assert clips == [
        (0, 60, 0),
        (60, 120, 1),
        (120, 180, 2),
        (180, 240, 3),
        (240, 300, 4),
    ]
    for start, end, _ in clips:
        assert converter.math.ceil(((end - start) / 30) * 2) + 1 <= 5


def test_convert_task_keeps_same_episode_ids_from_both_demo_directories(monkeypatch, tmp_path):
    rmbench_root = tmp_path / "data"
    task_root = rmbench_root / "cover_blocks"
    _create_episode(task_root, 0, seen=["Short demo task."])
    _create_episode(task_root, 0, demo_name="demo_clean_200", seen=["Long demo task."])
    monkeypatch.setattr(
        converter,
        "probe_video",
        lambda _path, _fps: converter.VideoInfo(fps=30, frame_count=4),
    )

    samples = converter.convert_task(
        task_root=task_root,
        rmbench_root=rmbench_root,
        rng=random.Random(0),
        fps_override=None,
        max_num_frame=128,
        skip_invalid=False,
        stats=converter.ConversionStats(),
    )

    assert [(sample["demo"], sample["episode_id"], sample["video"]) for sample in samples] == [
        ("demo_clean", 0, "cover_blocks/demo_clean/video/episode0.mp4"),
        ("demo_clean_200", 0, "cover_blocks/demo_clean_200/video/episode0.mp4"),
    ]


def test_main_uses_seen_global_task_and_annotated_subtasks(monkeypatch, tmp_path):
    rmbench_root = tmp_path / "data"
    task_root = rmbench_root / "battery_try"
    _create_episode(
        task_root,
        7,
        seen=["First global task.", "Second global task."],
        subtasks=[["First subtask.", 2], ["Second subtask.", 2]],
    )
    monkeypatch.setattr(
        converter,
        "probe_video",
        lambda _path, _fps: converter.VideoInfo(fps=30, frame_count=4),
    )

    output_jsonl = tmp_path / "output" / "annotations.jsonl"
    output_meta = tmp_path / "output" / "meta.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_rmbench_mm_video_gen.py",
            "--rmbench-root",
            str(rmbench_root),
            "--output-jsonl",
            str(output_jsonl),
            "--output-meta",
            str(output_meta),
            "--max-num-frame",
            "128",
            "--seed",
            "123",
        ],
    )

    assert converter.main() == 0

    samples = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    global_task = random.Random(123).choice(["First global task.", "Second global task."])
    assert samples == [
        {
            "video": "battery_try/demo_clean/video/episode7.mp4",
            "clip": [0.0, 0.03333333],
            "conversations": [
                {
                    "from": "human",
                    "value": f"Global task: {global_task}\nCurrent subtask: First subtask.",
                },
                {"from": "gpt", "value": ""},
            ],
            "task": "battery_try",
            "demo": "demo_clean",
            "episode_id": 7,
            "subtask_index": 0,
            "clip_index": 0,
        },
        {
            "video": "battery_try/demo_clean/video/episode7.mp4",
            "clip": [0.06666667, 0.1],
            "conversations": [
                {
                    "from": "human",
                    "value": f"Global task: {global_task}\nCurrent subtask: Second subtask.",
                },
                {"from": "gpt", "value": ""},
            ],
            "task": "battery_try",
            "demo": "demo_clean",
            "episode_id": 7,
            "subtask_index": 1,
            "clip_index": 0,
        },
    ]

    meta = json.loads(output_meta.read_text(encoding="utf-8"))
    assert meta["rmbench_mm_video_gen"] == {
        "root": str(rmbench_root.resolve()),
        "annotation": str(output_jsonl.resolve()),
        "repeat_time": 1.0,
        "length": 2,
        "task": "video_gen",
        "data_type": "rmbench_mm_video_gen",
    }
