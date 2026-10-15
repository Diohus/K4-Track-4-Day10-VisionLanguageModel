from types import SimpleNamespace
import pytest
torch = pytest.importorskip("torch")
from PIL import Image
from student.modeling import collate, encode_example, logical_numel, is_projector


class FakeProcessor:
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: 9)

    def __init__(self, bad_prefix=False):
        self.bad_prefix = bad_prefix

    def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
        # prompt contains two visual placeholders and an assistant header;
        # full adds answer IDs, stop token 9, then a template newline 10.
        ids = [1,2,2,3]
        if not add_generation_prompt:
            ids = ([0] if self.bad_prefix else [1])+[2,2,3,7,8,9,10]
        return {"input_ids":torch.tensor([ids]), "attention_mask":torch.ones(1,len(ids),dtype=torch.long),
                "mm_token_type_ids":torch.tensor([[0,1,1,0]+[0]*(len(ids)-4)]),
                "pixel_values":torch.ones(8,3), "image_grid_thw":torch.tensor([[1,2,4]])}


@pytest.fixture
def entry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/"data/images").mkdir(parents=True)
    Image.new("RGB",(640,360)).save(tmp_path/"data/images/a.jpg")
    return {"image_path":"images/a.jpg","objects":[]}


def test_mask_keeps_answer_and_end_only(entry):
    encoded = encode_example(entry,FakeProcessor())
    assert encoded["labels"].tolist() == [-100,-100,-100,-100,7,8,9]
    assert encoded["input_ids"].tolist() == [1,2,2,3,7,8,9]
    assert len(encoded["mm_token_type_ids"]) == 7
    assert encoded["pixel_values"].shape == (8,3)


def test_prefix_mismatch_fails(entry):
    with pytest.raises(AssertionError,match="prefix"):
        encode_example(entry,FakeProcessor(bad_prefix=True))


def test_collator_pads_text_but_concatenates_images(entry):
    a = encode_example(entry,FakeProcessor())
    b = {k:(v[:-1] if k in ("input_ids","attention_mask","labels","mm_token_type_ids") else v) for k,v in a.items()}
    batch = collate([a,b],99)
    assert batch["input_ids"].shape == (2,7)
    assert batch["input_ids"][1,-1].item() == 99
    assert batch["labels"][1,-1].item() == -100
    assert batch["attention_mask"][1,-1].item() == 0
    assert batch["mm_token_type_ids"].shape == (2,7)
    assert batch["pixel_values"].shape == (16,3)
    assert batch["image_grid_thw"].shape == (2,3)


def test_unhandled_processor_field_fails(entry):
    item = encode_example(entry,FakeProcessor())
    item["future_field"] = torch.ones(1)
    with pytest.raises(ValueError,match="Unsupported"):
        collate([item],0)


def test_logical_parameter_count_uses_unpacked_shape():
    fake = SimpleNamespace(quant_state=SimpleNamespace(shape=(8,8)), numel=lambda:32)
    assert logical_numel(fake) == 64
    assert logical_numel(torch.ones(3,4)) == 12


def test_projector_includes_deepstack_but_not_vit():
    assert is_projector("base_model.model.model.visual.deepstack_merger_list.0.linear_fc1.weight")
    assert is_projector("base_model.model.model.visual.merger.linear_fc1.weight")
    assert not is_projector("base_model.model.model.visual.blocks.0.mlp.linear_fc1.weight")
