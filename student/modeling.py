"""GPU code, imported only by training/inference commands."""
import json
import re
from pathlib import Path

import torch
from PIL import Image

from student.common import PROMPT, LLM_TARGET, VIT_TARGET, objects_to_answer, dump

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def is_projector(name):
    return "visual.merger." in name or "visual.deepstack_merger_list." in name


def load_base():
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Select T4 on Kaggle.")
    # Transformers 5.x matches full module prefixes, not ancestor basenames.
    # "visual" does not match model.visual.blocks.*; exclude "model.visual".
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["model.visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float16, quantization_config=quant,
        device_map={"": 0}, attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    for n, p in model.named_parameters():
        if "visual." in n or "lm_head" in n:
            assert p.__class__.__name__ != "Params4bit", f"Unexpected quantization: {n}"
    return model, processor


def logical_numel(p):
    # Params4bit packs two values per byte. quant_state.shape is the original shape.
    state = getattr(p, "quant_state", None)
    return int(__import__("math").prod(state.shape)) if state is not None else p.numel()


def architecture(model):
    counts = {"vision_backbone": 0, "projector": 0, "language_and_head": 0}
    for n, p in model.named_parameters():
        key = "projector" if is_projector(n) else ("vision_backbone" if "visual." in n else "language_and_head")
        counts[key] += logical_numel(p)
    return counts


def apply_setup(model, config):
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad = False
    regex = LLM_TARGET
    if config["lora"]["targets"] == "llm+vit":
        regex = f"(?:{LLM_TARGET}|{VIT_TARGET})"
    names = [n for n, _ in model.named_modules() if re.fullmatch(regex, n)]
    expected = 348 if config["lora"]["targets"] == "llm+vit" else 252
    assert len(names) == expected, f"Matched {len(names)}, expected {expected}. Inspect model tree."
    model = get_peft_model(model, LoraConfig(
        r=config["lora"]["r"], lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"], target_modules=regex,
        bias="none", task_type="CAUSAL_LM"))
    if config["train_projector"]:
        for n, p in model.named_parameters():
            if is_projector(n):
                p.requires_grad = True
    # GradScaler cannot unscale fp16 trainable gradients. Keep ALL trainables fp32.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return model, names


def user_message(image):
    return [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]


def encode_example(entry, processor):
    with Image.open(Path("data") / entry["image_path"]) as img:
        image = img.convert("RGB")
    user = user_message(image)
    answer = objects_to_answer(entry["objects"])
    kwargs = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt = processor.apply_chat_template(user, add_generation_prompt=True, **kwargs)
    full = processor.apply_chat_template(
        user + [{"role": "assistant", "content": answer}], add_generation_prompt=False, **kwargs)
    n = prompt["input_ids"].shape[1]
    assert torch.equal(full["input_ids"][0, :n], prompt["input_ids"][0]), "Template prefix differs"
    full.pop("token_type_ids", None)
    full["labels"] = full["input_ids"].clone()
    full["labels"][:, :n] = -100
    assert (full["labels"] != -100).any()
    # Remove trailing template newline after im_end; train exactly answer + stop token.
    end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    ends = (full["input_ids"][0, n:] == end_id).nonzero()
    assert len(ends) > 0, "Missing assistant stop token"
    stop = n + int(ends[0]) + 1
    sequence = {"input_ids", "attention_mask", "labels", "mm_token_type_ids"}
    return {k: (v[0, :stop] if k in sequence else v) for k, v in full.items()}


def collate(batch, pad_token_id):
    result = {}
    pads = {"input_ids": pad_token_id, "attention_mask": 0, "labels": -100, "mm_token_type_ids": 0}
    for k in batch[0]:
        if k in pads:
            result[k] = torch.nn.utils.rnn.pad_sequence(
                [b[k] for b in batch], batch_first=True, padding_value=pads[k])
        elif k in ("pixel_values", "image_grid_thw"):
            result[k] = torch.cat([b[k] for b in batch], dim=0)
        else:
            raise ValueError(f"Unsupported processor field: {k}; review collation explicitly")
    return result


def mask_audit(entry, processor):
    item = encode_example(entry, processor)
    token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_img = int((item["input_ids"] == token_id).sum())
    merge = processor.image_processor.merge_size
    grid_count = int(item["image_grid_thw"].prod(dim=-1).sum()) // merge**2
    assert n_img == grid_count == 220, (n_img, grid_count)
    assert (item["labels"][item["input_ids"] == token_id] == -100).all()
    text = processor.tokenizer.decode(item["labels"][item["labels"] != -100], skip_special_tokens=False)
    assert text == objects_to_answer(entry["objects"]) + "<|im_end|>", repr(text)
    result = {"image_id": entry["image_id"], "image_tokens": n_img,
              "total_tokens": len(item["input_ids"]), "supervised_tokens": int((item["labels"] != -100).sum()),
              "supervised_text": text, "prompt_prefix_verified": True}
    dump("report/diagnostics/mask_audit.json", result)
    print(result)


def save_adapter(model, processor, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    processor.save_pretrained(directory)
    extra = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
             if is_projector(n) and p.requires_grad}
    if extra:
        torch.save(extra, directory / "projector.pt")


def load_adapter(directory, trainable=False):
    from peft import PeftModel
    directory = Path(directory)
    config = json.loads((directory / "run.json").read_text())["config"]
    model, processor = load_base()
    model = PeftModel.from_pretrained(model, directory, is_trainable=trainable)
    if config["train_projector"]:
        path = directory / "projector.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained projector: {path}")
        extra = torch.load(path, map_location="cpu", weights_only=True)
        params = dict(model.named_parameters())
        expected = {n for n in params if is_projector(n)}
        assert set(extra) == expected, "Projector checkpoint names differ"
        with torch.no_grad():
            for n, value in extra.items():
                assert params[n].shape == value.shape
                params[n].data = params[n].data.float()
                params[n].copy_(value)
                params[n].requires_grad = trainable
    return model, processor
