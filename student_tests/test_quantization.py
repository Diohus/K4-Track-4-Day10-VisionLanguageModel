"""Regression against the installed Transformers matcher, without model weights."""
import ast
from pathlib import Path
import pytest

pytest.importorskip("transformers")
from transformers.quantizers.quantizers_utils import should_convert_module

ROOT = Path(__file__).resolve().parents[1]


def skip_patterns():
    tree = ast.parse((ROOT/"student/modeling.py").read_text(encoding="utf-8"))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "BitsAndBytesConfig")
    return ast.literal_eval(next(k.value for k in call.keywords if k.arg == "llm_int8_skip_modules"))


def test_all_vision_layers_and_projectors_excluded_by_real_matcher():
    names = [f"model.visual.blocks.{i}.{part}" for i in range(24)
             for part in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")]
    names += ["model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2", "lm_head"]
    names += [f"model.visual.deepstack_merger_list.{i}.{part}" for i in range(3)
              for part in ("linear_fc1", "linear_fc2")]
    assert all(not should_convert_module(n, skip_patterns()) for n in names)


def test_language_layers_still_quantized():
    names = [f"model.language_model.layers.{i}.{part}" for i in range(36)
             for part in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                          "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    assert all(should_convert_module(n, skip_patterns()) for n in names)
