"""Run with python -m student.run; one GPU process per experiment."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

from scripts.labcommon import read_jsonl
from student.common import dump, train_subset, audit_data


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_for(setup, fraction=1.0, batch=1, steps=300):
    import yaml
    config = yaml.safe_load(Path(f"configs/{setup}.yaml").read_text(encoding="utf-8"))
    config.update(batch_size=batch, grad_accum=8//batch, max_steps=steps,
                  train_fraction=fraction, inference_max_new_tokens=512)
    return config


def score(name, split, meta):
    output = f"report/{'evidence' if split == 'test' else 'validation'}/{name}.json"
    cmd = [sys.executable, "scripts/score.py", "--pred", f"work/{name}_{split}.jsonl",
           "--split", split, "--name", name.upper(), "--out", output]
    for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
        cmd += ["--meta", f"{key}={meta.get(key, 0)}"]
    subprocess.run(cmd, check=True)
    return read(output)


def train(name, config, resume=None):
    import torch
    from student.modeling import load_base, apply_setup, architecture, encode_example, collate, mask_audit, save_adapter
    from transformers import set_seed, get_cosine_schedule_with_warmup
    set_seed(config["seed"])
    directory = Path("work") / name
    if (directory / "run.json").exists() and not resume:
        raise FileExistsError(f"{directory} already completed. Use inference or a new --name.")
    directory.mkdir(parents=True, exist_ok=True)
    model, processor = load_base()
    counts = architecture(model)
    model, targets = apply_setup(model, config)
    rows = train_subset(read_jsonl("data/splits/train.jsonl"), config["train_fraction"], config["seed"])
    mask_audit(rows[0], processor)
    params = [p for p in model.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in params)
    optimizer = torch.optim.AdamW(params, lr=float(config["lr"]), weight_decay=0, foreach=False)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, math.ceil(config["max_steps"]*.05), config["max_steps"])
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.)
    dataset_version = read("data/splits/meta.json")["dataset_version"]
    train_ids = [r["image_id"] for r in rows]
    info = {"config": config, "dataset_version": dataset_version, "architecture": counts,
            "matched_modules": targets, "trainable_params": sum(p.numel() for p in params),
            "train_image_ids": train_ids, "gpu": torch.cuda.get_device_name(0),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("torch", "transformers", "peft", "bitsandbytes", "accelerate")}}
    dump(f"report/runs/{name}_config.json", info)
    start_step, previous_minutes, previous_peak = 0, 0., 0.
    if resume:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        cp = Path(resume)
        state = torch.load(cp / "training_state.pt", map_location="cpu", weights_only=False)
        assert state["config"] == config and state["dataset_version"] == dataset_version
        set_peft_model_state_dict(model, load_file(str(cp / "adapter_model.safetensors")))
        if config["train_projector"]:
            extra = torch.load(cp / "projector.pt", map_location="cpu", weights_only=True)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in extra:
                        p.copy_(extra[n])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step, previous_minutes, previous_peak = state["step"], state["minutes"], state["peak"]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.train()
    log_path = Path(f"report/runs/{name}_train.jsonl")
    if not resume:
        log_path.write_text("")
    elif log_path.exists():
        old_logs = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        log_path.write_text("".join(json.dumps(row)+"\n" for row in old_logs if row["step"] <= start_step))
    for step in range(start_step, config["max_steps"]):
        for attempt in range(6):
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            # Deterministic shuffled epochs; reconstructing from step makes resume exact.
            for micro in range(config["grad_accum"]):
                offset = (step*config["grad_accum"] + micro)*config["batch_size"]
                indices = []
                for pos in range(offset, offset+config["batch_size"]):
                    epoch, ix = divmod(pos, len(rows))
                    perm = list(range(len(rows)))
                    random.Random(config["seed"]+epoch).shuffle(perm)
                    indices.append(perm[ix])
                encoded = [encode_example(rows[i], processor) for i in indices]
                batch = {k: v.cuda() for k, v in collate(encoded, processor.tokenizer.pad_token_id).items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}; inspect mask/learning rate")
                loss_value += loss.item()/config["grad_accum"]
                scaler.scale(loss/config["grad_accum"]).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if torch.isfinite(norm):
                break
            scaler.update(new_scale=scaler.get_scale()/2)
            print(f"{name}: gradient overflow, retry step {step+1}, scale={scaler.get_scale()}", flush=True)
        else:
            raise FloatingPointError(f"Persistent nonfinite gradients at step {step+1}")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        event = {"step": step+1, "loss": loss_value, "grad_norm": float(norm),
                 "lr": scheduler.get_last_lr()[0]}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event)+"\n")
        if (step+1) % 10 == 0:
            print(name, event, flush=True)
        if (step+1) % 100 == 0:
            cp = directory / f"checkpoint-{step+1}"
            save_adapter(model, processor, cp)
            torch.cuda.synchronize()
            torch.save({"step": step+1, "config": config, "dataset_version": dataset_version,
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(), "rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                        "minutes": previous_minutes+(time.perf_counter()-started)/60,
                        "peak": max(previous_peak, torch.cuda.max_memory_allocated()/2**30)},
                       cp / "training_state.pt")
    torch.cuda.synchronize()
    info.update(train_minutes=previous_minutes+(time.perf_counter()-started)/60,
                peak_vram_gb=max(previous_peak, torch.cuda.max_memory_allocated()/2**30),
                completed_steps=config["max_steps"])
    save_adapter(model, processor, directory)
    dump(directory / "run.json", info)
    dump(f"report/runs/{name}.json", info)


def infer(name, split, adapter=None):
    import torch
    from PIL import Image
    from student.modeling import load_base, load_adapter, user_message
    if adapter:
        info = read(Path(adapter) / "run.json")
        assert info["dataset_version"] == read("data/splits/meta.json")["dataset_version"]
        model, processor = load_adapter(adapter)
    else:
        model, processor = load_base()
        info = {"trainable_params": 0, "peak_vram_gb": 0, "train_minutes": 0}
    model.eval()
    model.config.use_cache = True
    rows = read_jsonl(f"data/splits/{split}.jsonl")
    output = Path(f"work/{name}_{split}.jsonl")
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        raise FileExistsError(f"{output} exists; preserve it, use a new run name for changed models.")
    started = time.perf_counter()
    truncated = 0
    # Atomic result: a failed inference leaves .partial and is never graded as complete.
    with output.with_suffix(".partial").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            with Image.open(Path("data") / row["image_path"]) as im:
                image = im.convert("RGB")
            inputs = processor.apply_chat_template(
                user_message(image), tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
            inputs.pop("token_type_ids", None)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            tokens = generated[:, inputs["input_ids"].shape[1]:]
            truncated += int(tokens.shape[1] >= 512)
            raw = processor.batch_decode(tokens, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]
            f.write(json.dumps({"image_id": row["image_id"], "raw": raw}, ensure_ascii=False)+"\n")
            if (i+1) % 25 == 0:
                f.flush()
                print(f"{name} {split}: {i+1}/{len(rows)}", flush=True)
    output.with_suffix(".partial").replace(output)
    dump(f"report/diagnostics/{name}_{split}_inference.json",
         {"images": len(rows), "truncated_at_limit": truncated, "max_new_tokens": 512,
          "minutes": (time.perf_counter()-started)/60,
          "prediction_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    score(name, split, info)


def select():
    choices = {n: read(f"report/validation/{n}.json") for n in ("e1", "e2", "e3")}
    best = max(choices, key=lambda n: choices[n]["micro"]["f1"])
    config = read(f"work/{best}/run.json")["config"]
    assert config["max_steps"] == 300 and config["train_fraction"] == 1.
    dump("report/selection.json", {"best": best, "criterion": "validation micro F1, ties prefer simpler setup",
                                 "validation_f1": {n: d["micro"]["f1"] for n, d in choices.items()}})
    print("Best validation setup:", best)
    return best


def ablation():
    best = read("report/selection.json")["best"]
    original = read(f"work/{best}/run.json")["config"]
    smaller = copy.deepcopy(original)
    smaller["train_fraction"] = .1
    assert [k for k in original if original[k] != smaller[k]] == ["train_fraction"]
    # 100% endpoint reuses exactly the chosen SFT run; no redundant GPU training.
    dump("report/ablation.json", {"variable": "train_fraction", "base": best,
                                "values": [0.1, 1.0], "configs": [smaller, original]})
    return smaller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["audit", "train", "infer", "select", "e4", "e4-score-full"])
    parser.add_argument("--setup", choices=["e1", "e2", "e3"], default="e1")
    parser.add_argument("--name")
    parser.add_argument("--adapter")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch", type=int, choices=[1, 2], default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.action == "audit":
        audit_data()
    elif args.action == "train":
        train(args.name or args.setup, config_for(args.setup, batch=args.batch, steps=args.steps), args.resume)
    elif args.action == "infer":
        infer(args.name or args.setup, args.split, args.adapter)
    elif args.action == "select":
        select()
    elif args.action == "e4":
        train("e4_data_size_0.1", ablation(), args.resume)
    elif args.action == "e4-score-full":
        import shutil
        best = read("report/selection.json")["best"]
        shutil.copyfile(f"work/{best}_test.jsonl", "work/e4_data_size_1.0_test.jsonl")
        score("e4_data_size_1.0", "test", read(f"work/{best}/run.json"))


if __name__ == "__main__":
    main()
