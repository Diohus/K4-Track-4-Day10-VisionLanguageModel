import json
import re
import pytest
from scripts.score import parse_output
from student.common import yolo_to_box, objects_to_answer, iou_reward, train_subset, LLM_TARGET, VIT_TARGET
from student.reporting import paired_ci, image_counts
from student.run import config_for
from student.grpo import reward_function

GT = [{"label":"pothole", "bbox":[100,100,200,200]}]
GOOD = json.dumps([{"label":"pothole","bbox_2d":[100,100,200,200]}])


@pytest.mark.parametrize("raw,gt,expected", [
    (GOOD,GT,1.), ("broken",GT,0.), ('{}',GT,0.),
    (GOOD.replace("pothole","crack"),GT,.2),
    (json.dumps(json.loads(GOOD)*2),GT,.6),
    ("[]",[],1.), ("[]",GT,.2), (GOOD,[],.2),
    ('[{"label":"pothole","bbox_2d":[0,0,1000,1000]}]',GT,.2),
    ('[{"label":"pothole","bbox_2d":[100,100,100,200]}]',GT,0.),
])
def test_reward(raw, gt, expected):
    assert iou_reward(raw,gt) == pytest.approx(expected)


def test_group_reward_conversation():
    assert reward_function([[{"role":"assistant","content":GOOD}]], [GT]) == [1.]


def test_yolo_roundtrip():
    assert yolo_to_box("crack",[.5,.5,.2,.4])["bbox_2d"] == [400,300,600,700]


def test_boundary_box_positive_area():
    box = yolo_to_box("crack",[1,1,.0001,.0001])["bbox_2d"]
    assert 0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000


@pytest.mark.parametrize("yolo", [[.5,.5,-.1,.2], [float("nan"),.5,.1,.2], [1.1,.5,.1,.2]])
def test_invalid_gt(yolo):
    with pytest.raises(ValueError):
        yolo_to_box("pothole",yolo)


def test_stable_sort_and_empty():
    objects = [{"label":"crack","yolo":[.7,.7,.1,.1]}, {"label":"pothole","yolo":[.2,.2,.1,.1]}]
    assert objects_to_answer(objects) == objects_to_answer(objects[::-1])
    assert json.loads(objects_to_answer(objects))[0]["label"] == "pothole"
    assert objects_to_answer([]) == "[]"


def test_exact_target_modules():
    llm = [f"model.language_model.layers.{i}.{part}" for i in range(36)
           for part in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
                        "mlp.gate_proj","mlp.up_proj","mlp.down_proj"]]
    vit = [f"model.visual.blocks.{i}.{part}" for i in range(24)
           for part in ["attn.qkv","attn.proj","mlp.linear_fc1","mlp.linear_fc2"]]
    assert sum(bool(re.fullmatch(LLM_TARGET,n)) for n in llm+vit) == 252
    assert sum(bool(re.fullmatch(VIT_TARGET,n)) for n in llm+vit) == 96
    assert not re.fullmatch(VIT_TARGET,"model.visual.merger.linear_fc1")


def test_train_subset_is_nested_and_deterministic():
    rows = [{"image_id":str(i)} for i in range(1422)]
    small = train_subset(rows,.1,42)
    assert len(small) == 142
    assert small == train_subset(rows[::-1],.1,42)
    assert small == train_subset(rows,1.,42)[:142]


def test_effective_batch_consistent():
    for setup in ("e1","e2","e3"):
        c = config_for(setup)
        assert c["batch_size"]*c["grad_accum"] == 8 and c["max_steps"] == 300


def test_paired_bootstrap_identical_models():
    entries = [{"image_id":str(i),"session":str(i//2),
                "objects":[{"label":"pothole","yolo":[.15,.15,.1,.1]}]} for i in range(6)]
    pred = {e["image_id"]:GOOD for e in entries}
    assert image_counts(entries[0],GOOD) == (1,0,0)
    assert paired_ci(entries,pred,pred,repeats=100)["delta_ci95"] == [0.,0.]


def test_paired_bootstrap_direction():
    entries = [{"image_id":str(i),"session":str(i//2),
                "objects":[{"label":"pothole","yolo":[.15,.15,.1,.1]}]} for i in range(6)]
    good = {e["image_id"]:GOOD for e in entries}
    empty = {e["image_id"]:"[]" for e in entries}
    assert paired_ci(entries,empty,good,repeats=100)["delta_ci95"] == [1.,1.]


@pytest.mark.parametrize("yolo", [
    [.5,.5,0,.2], [.5,.5,.2,0], [.5,.5,0,0],
    [0,0,0,0], [1,1,0,0], [1,1,0,.2], [1,1,.2,0],
])
def test_zero_area_annotation_serializes_without_changing_source(yolo):
    original = list(yolo)
    obj = {"label": "pothole", "yolo": yolo}
    parsed = parse_output(objects_to_answer([obj]))
    assert parsed is not None and len(parsed) == 1
    x1, y1, x2, y2 = parsed[0]["bbox"]
    assert 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000
    assert yolo == original
    # Scoring must still see the ORIGINAL zero-area ground truth, hence no TP.
    assert image_counts({"objects": [obj]}, objects_to_answer([obj])) == (0,1,1)
