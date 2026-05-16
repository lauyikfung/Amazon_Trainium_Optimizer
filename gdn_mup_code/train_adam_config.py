from __future__ import annotations

import sys
import types
from ast import literal_eval
from collections.abc import Sequence
from tokenize import open as open_source
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ConfigDict


class FinewebEduTrainConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    # I/O
    data_path: str = "data"
    out_dir: str = "out"
    resume_dir: str = "out"
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    save_checkpoints: bool = True
    always_save_checkpoint: bool = True
    keep_last_checkpoints: int | None = None
    init_from: str = "scratch"

    # wandb logging
    wandb_log: bool = False
    wandb_project: str = "nanogpt"
    wandb_run_name: str = "gpt"

    # data
    dataset: str = "fineweb-edu"
    tokenizer: str = "gpt2"
    byteoss_vocab: str = "compact"
    gradient_accumulation_steps: int = 5
    batch_size: int = 12
    global_batch_size: int | None = None
    block_size: int = 1024
    data_loader: str = "random"
    stream_packing: str = "auto"
    fit_or_cut_threshold: int = 128
    ignore_doc_start_loss: bool = False

    # reproducibility
    seed: int = 1337
    deterministic: bool = False
    bitwise_deterministic: bool = False
    data_seed: int = 1337
    eval_seed: int = 1338
    data_rng_mode: str = "stateful"

    # model
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    hidden_size: int = 768
    head_dim: int = -1
    tpa_kvrank: int = 2
    tpa_qrank: int = 6
    dropout: float = 0.0
    bias: bool = False
    using_groupnorm: bool = False

    # muP
    mup: bool = False
    hidden_size_base: int = 768
    embedding_lr_multiplier: float = 1.0

    # KV shifting
    use_k_shift: bool = False
    use_v_shift: bool = False

    # initialization / normalization knobs
    embedding_init_std: float = 0.02
    hidden_init_std_factor: float = 0.02
    use_qk_rmsnorm: bool = False
    rope_ratio: float = 1.0
    p_tie_mode: str = "none"
    p_head_dim: int = -1

    # optimizer
    optimizer_name: str = "sgd"
    learning_rate_base: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    zero_stage: int = 0

    # learning rate decay settings
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr_base: float = 6e-5

    # DDP settings
    backend: str = "nccl"

    # scheduler
    schedule: str = "cosine"
    mymup: bool = False

    # model variants
    model_type: str = "gpt_base"
    num_key_value_heads: int | None = None

    # system
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    scale_attn_by_inverse_layer_idx: bool = False


TrainAdamConfigBase = FinewebEduTrainConfig


def _parse_value(raw: str) -> object:
    try:
        return literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


def _is_instance_for_annotation(value: object, annotation: object) -> bool:
    origin = get_origin(annotation)
    if origin is None:
        if annotation is float:
            return isinstance(value, (float, int)) and not isinstance(value, bool)
        if annotation is Any:
            return True
        return isinstance(value, annotation)
    if origin in (list, dict, tuple, set):
        return isinstance(value, origin)
    args = get_args(annotation)
    if origin is type(None):
        return value is None
    if str(origin) == "typing.Union" or origin is types.UnionType:
        return any(_is_instance_for_annotation(value, arg) for arg in args)
    return True


def _coerce_known_value(config_cls: type[FinewebEduTrainConfig], key: str, value: object) -> object:
    annotation = config_cls.model_fields[key].annotation
    if annotation is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if not _is_instance_for_annotation(value, annotation):
        raise TypeError(f"Type mismatch for {key!r}: got {type(value).__name__}, expected {annotation!r}.")
    return value


def _exec_config_file(config_file: str, env: dict[str, object]) -> None:
    print(f"Overriding config with {config_file}:")
    with open_source(config_file) as f:
        config_text = f.read()
    print(config_text)
    exec(compile(config_text, config_file, "exec"), env)


def load_train_config(
    config_cls: type[FinewebEduTrainConfig],
    argv: Sequence[str],
) -> tuple[FinewebEduTrainConfig, dict[str, object]]:
    values = config_cls.model_construct().model_dump()
    extra: dict[str, object] = {}

    for arg in argv:
        if "=" not in arg:
            if arg.startswith("--"):
                raise ValueError(f"Expected a config file path, got {arg!r}.")
            env: dict[str, object] = {
                "FinewebEduTrainConfig": config_cls,
                "TrainAdamConfigBase": TrainAdamConfigBase,
                "CONFIG": None,
            }
            _exec_config_file(arg, env)
            config_obj = env.get("CONFIG")
            if isinstance(config_obj, BaseModel):
                dumped = config_obj.model_dump(exclude_unset=True)
                for key, value in dumped.items():
                    if key in config_cls.model_fields:
                        values[key] = value
                    else:
                        extra[key] = value
            for key, value in env.items():
                if key.startswith("_") or key in {"FinewebEduTrainConfig", "TrainAdamConfigBase", "CONFIG"}:
                    continue
                if key in config_cls.model_fields:
                    values[key] = value
                elif key.isidentifier():
                    extra[key] = value
            continue

        if not arg.startswith("--"):
            raise ValueError(f"Expected an override in the form --name=value, got {arg!r}.")
        key, raw_value = arg[2:].split("=", 1)
        value = _parse_value(raw_value)
        if key in config_cls.model_fields:
            values[key] = _coerce_known_value(config_cls, key, value)
        else:
            extra[key] = value
        print(f"Overriding: {key} = {value}")

    config = config_cls.model_validate(values)
    return config, extra


if __name__ == "__main__":
    cfg, extras = load_train_config(FinewebEduTrainConfig, sys.argv[1:])
    print(cfg.model_dump())
    if extras:
        print({"extra": extras})
