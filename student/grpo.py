"""Optional E5; run only after selecting the SFT setup on validation."""
import json
import time
from pathlib import Path

from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes
from student.common import PROMPT, dump, iou_reward
from student.run import read


def reward_function(completions, gt_boxes, **kwargs):
    result = []
    for completion, gt in zip(completions, gt_boxes):
        if isinstance(completion, list):
            content = completion[-1]["content"]
            completion = content if isinstance(content, str) else "".join(
                c.get("text", "") for c in content if c.get("type") == "text")
        result.append(iou_reward(completion, gt))
    return result


def main():
    import torch
    from datasets import Dataset, Image as HFImage
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from student.modeling import load_adapter, save_adapter
    best = read("report/selection.json")["best"]
    source = Path("work") / best
    destination = Path("work/e5")
    if (destination / "run.json").exists():
        raise FileExistsError("E5 already completed")
    source_info = read(source / "run.json")
    set_seed(42)
    model, processor = load_adapter(source, trainable=True)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.enable_input_require_grads()
    model.config.use_cache = False
    messages = [{"role":"user", "content":[{"type":"image"}, {"type":"text","text":PROMPT}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    dataset = Dataset.from_list([
        {"prompt":prompt, "image": str((Path("data")/e["image_path"]).resolve()),
         "gt_boxes":_gt_boxes(e)}
        for e in read_jsonl("data/splits/train.jsonl")
    ]).cast_column("image", HFImage())

    class SaveProjector(TrainerCallback):
        def on_save(self, args, state, control, model=None, **kwargs):
            save_adapter(model, processor, Path(args.output_dir)/f"checkpoint-{state.global_step}")

    # Group size 4 divides effective generation batch 1*4 on one GPU.
    args = GRPOConfig(
        output_dir=str(destination), per_device_train_batch_size=1,
        gradient_accumulation_steps=4, num_generations=4,
        max_completion_length=128, max_steps=60, learning_rate=5e-6, beta=0.,
        use_vllm=False, fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False},
        logging_steps=1, save_steps=20, save_total_limit=2,
        report_to="none", seed=42, remove_unused_columns=False)
    trainer = GRPOTrainer(model=model, args=args, processing_class=processor,
                          reward_funcs=reward_function, train_dataset=dataset,
                          callbacks=[SaveProjector()])
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    info = {
        "config":source_info["config"], "dataset_version":source_info["dataset_version"],
        "initial_adapter":best, "grpo":args.to_dict(), "completed_steps":trainer.state.global_step,
        "trainable_params":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_vram_gb":torch.cuda.max_memory_allocated()/2**30,
        "train_minutes":(time.perf_counter()-started)/60}
    assert trainer.state.global_step == 60
    save_adapter(model, processor, destination)
    dump(destination/"run.json", info)
    dump("report/runs/e5.json", info)
    dump("report/runs/e5_grpo_log.json", trainer.state.log_history)
    print("E5 trained; run student.run infer with --adapter work/e5, then student.reporting.")


if __name__ == "__main__":
    main()
