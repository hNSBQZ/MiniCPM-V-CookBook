"""
Quantize MiniCPM-V 4's LLM backbone (LlamaForCausalLM) to 4-bit using GPTQ.
Output is compatible with both transformers and vLLM for inference.

Download the model from https://huggingface.co/openbmb/MiniCPM-V-4
Install AutoGPTQ (e.g. git clone AutoGPTQ then pip install -e .).
cd MiniCPM-V-CookBook
python quantization/gptq/minicpm-v4_gptq_quantize.py
"""
import os
import sys
import json
import shutil
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig


def _ensure_hf_dynamic_module_cache(model_path: str) -> None:
    """Pre-copy all .py files so transitive relative imports are available."""
    from transformers.dynamic_module_utils import (
        HF_MODULES_CACHE,
        TRANSFORMERS_DYNAMIC_MODULE_NAME,
        _sanitize_module_name,
        init_hf_modules,
    )
    init_hf_modules()
    submodule = _sanitize_module_name(Path(model_path).name)
    cache_dir = Path(HF_MODULES_CACHE) / TRANSFORMERS_DYNAMIC_MODULE_NAME / submodule
    cache_dir.mkdir(parents=True, exist_ok=True)
    for py in Path(model_path).glob("*.py"):
        dst = cache_dir / py.name
        if not dst.exists():
            shutil.copy2(str(py), str(dst))


logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "/model/MiniCPM-V-4"
OUTPUT_PATH = "./model/MiniCPM-V-4-gptq-int4"

BITS = 4
GROUP_SIZE = 128
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LEN = 512


def prepare_calibration_data(tokenizer, num_samples, max_length):
    """Prepare calibration data from Alpaca dataset."""
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

    examples = []
    for sample in dataset:
        text = sample.get("text", "") or sample.get("output", "") or sample.get("instruction", "")
        if not text:
            continue
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        examples.append({
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
        })

    logger.info(f"Prepared {len(examples)} calibration samples")
    return examples


def main():
    logger.info(f"Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    logger.info("Preparing calibration data...")
    examples = prepare_calibration_data(tokenizer, NUM_CALIBRATION_SAMPLES, MAX_SEQ_LEN)

    _ensure_hf_dynamic_module_cache(MODEL_PATH)
    logger.info(f"Loading full MiniCPM-V model from {MODEL_PATH}")
    full_model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    llm = full_model.llm
    logger.info(f"Extracted LLM backbone: {llm.__class__.__name__}")
    logger.info(f"LLM config model_type: {llm.config.model_type}")

    original_model_type = llm.config.model_type
    llm.config.model_type = "llama"

    from auto_gptq import BaseQuantizeConfig
    from auto_gptq.modeling.llama import LlamaGPTQForCausalLM

    quantize_config = BaseQuantizeConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
    )
    logger.info(f"Quantize config: bits={BITS}, group_size={GROUP_SIZE}")

    llm.seqlen = MAX_SEQ_LEN

    gptq_model = LlamaGPTQForCausalLM(llm, False, quantize_config)

    logger.info("Starting GPTQ quantization...")
    gptq_model.quantize(examples, batch_size=1)
    logger.info("Quantization complete!")

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    logger.info(f"Saving quantized LLM to {OUTPUT_PATH} (temp)")
    llm_temp_dir = os.path.join(OUTPUT_PATH, "_llm_temp")
    os.makedirs(llm_temp_dir, exist_ok=True)
    gptq_model.save_quantized(llm_temp_dir, use_safetensors=True)

    logger.info("Building full quantized model...")

    for fname in os.listdir(MODEL_PATH):
        src = os.path.join(MODEL_PATH, fname)
        dst = os.path.join(OUTPUT_PATH, fname)
        if fname.startswith("model") and fname.endswith(".safetensors"):
            continue
        if fname in ("config.json", "quantize_config.json", "model.safetensors.index.json"):
            continue
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            logger.info(f"  Copied: {fname}")
        elif os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
            logger.info(f"  Copied dir: {fname}")

    logger.info("Saving combined state dict (quantized LLM + original multimodal)...")
    from safetensors.torch import load_file as safe_load, save_file as safe_save

    quantized_llm_files = [f for f in os.listdir(llm_temp_dir) if f.endswith(".safetensors")]
    llm_state_dict = {}
    for f in quantized_llm_files:
        sd = safe_load(os.path.join(llm_temp_dir, f))
        llm_state_dict.update(sd)
    logger.info(f"  Loaded {len(llm_state_dict)} quantized LLM tensors")

    non_llm_state_dict = {}
    full_model_sd = full_model.state_dict()
    for key, value in full_model_sd.items():
        if not key.startswith("llm."):
            non_llm_state_dict[key] = value
    logger.info(f"  Collected {len(non_llm_state_dict)} non-LLM tensors")

    combined_sd = {}
    for key, value in llm_state_dict.items():
        combined_sd[f"llm.{key}"] = value
    for key, value in non_llm_state_dict.items():
        combined_sd[key] = value
    logger.info(f"  Total combined tensors: {len(combined_sd)}")

    MAX_SHARD_SIZE_BYTES = 4 * 1024 * 1024 * 1024  # 4GB per shard
    sorted_keys = sorted(combined_sd.keys())

    shards = []
    current_shard = {}
    current_size = 0
    weight_map = {}

    for key in sorted_keys:
        tensor = combined_sd[key]
        tensor_size = tensor.nelement() * tensor.element_size()

        if current_size > 0 and current_size + tensor_size > MAX_SHARD_SIZE_BYTES:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor.clone().contiguous()
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    if len(shards) == 1:
        shard_name = "model.safetensors"
        safe_save(shards[0], os.path.join(OUTPUT_PATH, shard_name))
        for key in shards[0]:
            weight_map[key] = shard_name
        logger.info(f"  Saved single shard: {shard_name}")
    else:
        for i, shard in enumerate(shards):
            shard_name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
            safe_save(shard, os.path.join(OUTPUT_PATH, shard_name))
            for key in shard:
                weight_map[key] = shard_name
            logger.info(f"  Saved shard {i+1}/{len(shards)}: {shard_name}")

    index = {
        "metadata": {"total_size": sum(t.nelement() * t.element_size() for t in combined_sd.values())},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUTPUT_PATH, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    logger.info("  Saved weight index")

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    config_dict = config.to_dict()
    config_dict["quantization_config"] = {
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "damp_percent": 0.01,
        "desc_act": False,
        "static_groups": False,
        "sym": True,
        "true_sequential": True,
        "quant_method": "gptq",
        "checkpoint_format": "gptq",
        "model_name_or_path": None,
        "model_file_base_name": None,
    }
    config_dict["model_type"] = original_model_type

    with open(os.path.join(OUTPUT_PATH, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info("  Saved config.json with quantization_config")

    quant_config_dict = quantize_config.to_dict()
    with open(os.path.join(OUTPUT_PATH, "quantize_config.json"), "w") as f:
        json.dump(quant_config_dict, f, indent=2)
    logger.info("  Saved quantize_config.json")

    shutil.rmtree(llm_temp_dir)
    logger.info("  Cleaned up temp directory")

    del combined_sd, llm_state_dict, non_llm_state_dict, full_model_sd
    torch.cuda.empty_cache()

    logger.info(f"Done! Quantized model saved to: {OUTPUT_PATH}")
    logger.info(f"Quantization: W{BITS}A16 GPTQ (weight-only {BITS}-bit, activation fp16)")
    logger.info("Compatible with: transformers (via optimum) and vLLM")


if __name__ == "__main__":
    main()
