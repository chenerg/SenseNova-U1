import json
import types

import numpy as np
import pytest
import torch
from PIL import Image
from sensenovavl.data import dataset as video_dataset
from sensenovavl.data import multimodal_dataset
from sensenovavl.data.constants import IGNORE_INDEX
from sensenovavl.data.multimodal_dataset import (
    LazySupervisedDataset,
    build_video_gen_prompt,
    expand_video_gen_frames,
)


class _FakeBatch:
    def __init__(self, frames):
        self.frames = frames

    def asnumpy(self):
        return self.frames


class _FakeVideoReader:
    frame_count = 10
    fps = 2.0
    last_indices = None

    def __init__(self, _path, num_threads=1):
        assert num_threads == 1
        self.frames = np.arange(self.frame_count * 2 * 2 * 3, dtype=np.uint8).reshape(self.frame_count, 2, 2, 3)

    def __len__(self):
        return self.frame_count

    def get_avg_fps(self):
        return self.fps

    def get_batch(self, indices):
        type(self).last_indices = list(indices)
        return _FakeBatch(self.frames[indices])


@pytest.mark.parametrize(
    ("sample_fps", "expected_slots", "expected_decode_times"),
    [
        (1, [0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 12.0, 13.0, 13.2]),
        (
            2,
            [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.2],
        ),
    ],
)
def test_video_gen_sample_slots_and_decode_times(sample_fps, expected_slots, expected_decode_times):
    slots, decode_times = video_dataset.get_video_gen_sample_times([10.0, 13.2], sample_fps)

    assert slots == expected_slots
    assert decode_times == expected_decode_times


def test_exact_grid_endpoint_has_no_extra_slot():
    slots, decode_times = video_dataset.get_video_gen_sample_times([10.0, 13.0], 2)

    assert slots == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert decode_times[-1] == 13.0


def test_near_grid_endpoint_has_no_extra_slot_from_float_error():
    clip = [13.46666667, 16.966666670000002]

    slots, decode_times = video_dataset.get_video_gen_sample_times(clip, 2)

    assert slots == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    assert decode_times[-1] == pytest.approx(clip[1])


def test_prompt_uses_regular_slot_for_padded_endpoint():
    slots, _ = video_dataset.get_video_gen_sample_times([10.0, 13.2], 1)

    assert build_video_gen_prompt(slots) == (
        "00:00.00]:<image>\n00:01.00]:<image>\n00:02.00]:<image>\n00:03.00]:<image>\n00:04.00]:<image>"
    )


def test_timestamp_decoder_keeps_duplicate_endpoint_indices(monkeypatch):
    monkeypatch.setattr(video_dataset, "VideoReader", _FakeVideoReader)

    frames, _, _, _, slots, decode_times, indices = video_dataset.read_frames_decord_video_gen(
        "unused.mp4",
        clip=[1.0, 4.2],
        sample_fps=1,
        max_num_frames=10,
    )

    assert slots == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert decode_times == [1.0, 2.0, 3.0, 4.0, 4.2]
    assert indices == [2, 4, 6, 8, 8]
    assert _FakeVideoReader.last_indices == indices
    assert len(frames) == len(indices)


@pytest.mark.parametrize(
    ("clip", "sample_fps"),
    [
        (None, 1),
        ([0.0], 1),
        ([-1.0, 1.0], 1),
        ([1.0, 1.0], 1),
        ([2.0, 1.0], 1),
        ([0.0, float("inf")], 1),
        ([0.0, 1.0], 0),
        ([0.0, 1.0], 1.5),
    ],
)
def test_invalid_sampling_configuration_is_rejected(clip, sample_fps):
    with pytest.raises(ValueError):
        video_dataset.get_video_gen_sample_times(clip, sample_fps)


def test_out_of_range_short_and_too_many_videos_are_rejected(monkeypatch):
    monkeypatch.setattr(video_dataset, "VideoReader", _FakeVideoReader)

    with pytest.raises(ValueError, match="outside video duration"):
        video_dataset.read_frames_decord_video_gen("unused.mp4", clip=[0.0, 6.0], sample_fps=1, max_num_frames=20)
    with pytest.raises(ValueError, match="exceeds max_num_frame"):
        video_dataset.read_frames_decord_video_gen("unused.mp4", clip=[0.0, 5.0], sample_fps=2, max_num_frames=10)

    class OneFrameVideoReader(_FakeVideoReader):
        frame_count = 1

    monkeypatch.setattr(video_dataset, "VideoReader", OneFrameVideoReader)
    with pytest.raises(ValueError, match="at least two frames"):
        video_dataset.read_frames_decord_video_gen("unused.mp4", clip=[0.0, 0.5], sample_fps=1, max_num_frames=10)


def test_frame_expansion_and_generation_flags():
    frames, generation_flags, duplicate_flags = expand_video_gen_frames(["F0", "F1", "F2", "F3"])

    assert frames == ["F0", "F1", "F1", "F2", "F2", "F3"]
    assert generation_flags == [False, True, False, True, False, True]
    assert duplicate_flags == [False, False, True, False, True, False]


def _make_video_gen_dataset(monkeypatch, sample_fps):
    dataset = object.__new__(LazySupervisedDataset)
    dataset.root = "/dataset"
    dataset.max_num_frame = 32
    dataset.max_num_frame_gen = 32
    dataset.dynamic_image_version = "fixed"
    dataset.pad2square = False
    dataset.is_train = False
    dataset.image_size = 2
    dataset.dynamic_image_size = False
    dataset.max_pixels_gen = 4096
    dataset.min_pixels_gen = 4
    dataset.max_tokens = 4096
    dataset.patch_size = 1
    dataset.downsample_ratio = 1
    dataset.num_image_token = 1
    dataset.template_name = "sensenovalm2-chat-v3"
    dataset.tokenizer = None
    dataset.ds_name = "video_gen_test"
    dataset.image_context_token_id = 7
    dataset.type_id = 6
    dataset.typeid2type = {6: "mm_video_gen"}
    dataset.worker_id = 0

    captured = {}

    def loader(path, **kwargs):
        captured["path"] = path
        captured["loader_kwargs"] = kwargs
        slots, decode_times = video_dataset.get_video_gen_sample_times(kwargs["clip"], kwargs["sample_fps"])
        logical_frames = [
            Image.new("RGB", (2, 2), color=(32 * index, 32 * index, 32 * index)) for index in range(len(slots))
        ]
        return (
            logical_frames,
            2.0,
            24.0,
            sample_fps,
            slots,
            decode_times,
            list(range(len(slots))),
        )

    def preprocess(
        _template_name,
        data_item,
        _tokenizer,
        num_image_tokens,
        **_kwargs,
    ):
        captured["assistant_prompt"] = data_item["conversations"][1]["value"]
        token_count = sum(num_image_tokens)
        return {
            "input_ids": torch.tensor([[7] * token_count + [9]]),
            "labels": torch.tensor([[1] * (token_count + 1)]),
        }

    dataset.tcs_loader = loader
    dataset.preprocess_function = preprocess
    monkeypatch.setattr(multimodal_dataset.random, "randint", lambda _start, _end: sample_fps)
    monkeypatch.setattr(
        multimodal_dataset,
        "build_transform",
        lambda **_kwargs: lambda image: torch.tensor(np.asarray(image).copy()).permute(2, 0, 1),
    )
    return dataset, captured


@pytest.mark.parametrize("sample_fps", [1, 2])
def test_video_gen_item_masks_ce_and_keeps_future_flow_targets(monkeypatch, sample_fps):
    dataset, captured = _make_video_gen_dataset(monkeypatch, sample_fps)
    data_item = {
        "video": "clip.mp4",
        "clip": [10.0, 12.0],
        "conversations": [
            {"from": "human", "value": "Task: keep walking."},
            {"from": "gpt", "value": ""},
        ],
    }

    result = dataset.video_gen_get_item(data_item)

    assert captured["path"] == "/dataset/clip.mp4"
    assert captured["loader_kwargs"]["sample_fps"] == sample_fps
    slots, _ = video_dataset.get_video_gen_sample_times(data_item["clip"], sample_fps)
    _, expected_generation_flags, expected_duplicate_flags = expand_video_gen_frames(list(range(len(slots))))
    assert captured["assistant_prompt"] == build_video_gen_prompt(slots, duplicate_intermediate=True)
    assert result["image_for_gen_flags"].tolist() == expected_generation_flags
    assert result["image_for_gen_loss_flags"].tolist() == expected_generation_flags
    assert result["is_image_duplicated_for_und_flags"].tolist() == expected_duplicate_flags
    assert result["pixel_values"].shape == (
        len(expected_generation_flags),
        3,
        2,
        2,
    )
    assert torch.equal(result["pixel_values"][1], result["pixel_values"][2])
    assert torch.all(result["labels"] == IGNORE_INDEX)

    generation_targets = result["pixel_values"][result["image_for_gen_flags"]]
    fake_flow_matching_loss = generation_targets.float().square().mean()
    assert generation_targets.shape[0] == len(slots) - 1
    assert fake_flow_matching_loss.item() > 0


def test_video_gen_randomly_crops_contiguous_logical_frames(monkeypatch):
    dataset, captured = _make_video_gen_dataset(monkeypatch, sample_fps=2)
    dataset.max_num_frame_gen = 8

    def randint(start, end):
        if (start, end) == (1, 2):
            return 2
        assert (start, end) == (0, 5)
        return 3

    monkeypatch.setattr(multimodal_dataset.random, "randint", randint)
    data_item = {
        "video": "clip.mp4",
        "clip": [10.0, 16.0],
        "conversations": [
            {"from": "human", "value": "Task: keep walking."},
            {"from": "gpt", "value": ""},
        ],
    }

    result = dataset.video_gen_get_item(data_item)

    assert captured["loader_kwargs"]["clip"] == [11.5, 15.0]
    assert captured["loader_kwargs"]["max_num_frames"] == 8
    expected_slots = [index / 2 for index in range(8)]
    assert captured["assistant_prompt"] == build_video_gen_prompt(
        expected_slots, duplicate_intermediate=True
    )
    assert result["image_for_gen_flags"].sum().item() == 7
    assert result["pixel_values"].shape[0] == 14


def test_video_gen_get_sample_allows_all_ignored_labels(monkeypatch):
    dataset, _ = _make_video_gen_dataset(monkeypatch, sample_fps=1)
    data_item = {
        "video": "clip.mp4",
        "clip": [10.0, 12.0],
        "conversations": [
            {"from": "human", "value": "Task: keep walking."},
            {"from": "gpt", "value": ""},
        ],
    }

    result = dataset.get_sample(json.dumps(data_item).encode())

    assert result is not None
    assert torch.all(result["labels"] == IGNORE_INDEX)
    assert torch.all(result["type_ids"] == dataset.type_id)


@pytest.mark.parametrize(
    ("task_type", "data_item", "expected_method"),
    [
        (
            "mm_video",
            {"video": "clip.mp4", "conversations": []},
            "video_get_item",
        ),
        (
            "mm_t2i",
            {"image": "image.png", "conversations": []},
            "multi_modal_get_item",
        ),
        (
            "mm_interleave_gen",
            {"images": ["image.png"], "conversations": []},
            "multi_modal_get_item",
        ),
    ],
)
def test_existing_task_dispatch_is_unchanged(task_type, data_item, expected_method):
    dataset = object.__new__(LazySupervisedDataset)
    dataset.type_id = 5
    dataset.typeid2type = {5: task_type}
    dataset.max_tokens = 32
    dataset.ds_name = "regression"
    dataset.worker_id = 0
    dataset.root = None
    calls = []

    def make_result(all_ignored):
        label = IGNORE_INDEX if all_ignored else 1
        return {
            "input_ids": torch.tensor([1]),
            "labels": torch.tensor([label]),
        }

    dataset.video_get_item = types.MethodType(
        lambda self, _item: calls.append("video_get_item") or make_result(False),
        dataset,
    )
    dataset.multi_modal_get_item = types.MethodType(
        lambda self, _item: calls.append("multi_modal_get_item") or make_result(task_type != "mm_video"),
        dataset,
    )
    dataset.pure_text_get_item = types.MethodType(
        lambda self, _item: calls.append("pure_text_get_item") or make_result(False),
        dataset,
    )

    result = dataset.get_sample(json.dumps(data_item).encode())

    assert result is not None
    assert calls == [expected_method]


@pytest.mark.parametrize(
    "invalid_update",
    [
        {"clip": None},
        {"clip": [1.0, 1.0]},
        {"sample_fps": 1},
        {"frame_timestamps": [0.0, 1.0]},
        {"conversations": [{"from": "human", "value": "Task"}]},
        {
            "conversations": [
                {"from": "human", "value": "Task"},
                {"from": "gpt", "value": "00:00.00]:<image>"},
            ]
        },
    ],
)
def test_invalid_video_gen_jsonl_is_safely_skipped(monkeypatch, invalid_update):
    dataset, _ = _make_video_gen_dataset(monkeypatch, sample_fps=1)
    data_item = {
        "video": "clip.mp4",
        "clip": [10.0, 12.0],
        "conversations": [
            {"from": "human", "value": "Task: keep walking."},
            {"from": "gpt", "value": ""},
        ],
    }
    data_item.update(invalid_update)

    assert dataset.get_sample(json.dumps(data_item).encode()) is None
