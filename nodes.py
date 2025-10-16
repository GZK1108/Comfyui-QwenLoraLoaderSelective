from __future__ import annotations

import logging
from typing import Dict

import comfy.lora
import comfy.lora_convert
from comfy import utils
import folder_paths

from comfy_api.latest import io


def _split_filters(raw: str) -> list[str]:
    if not raw:
        return []
    tokens = [token.strip() for token in raw.replace("\n", ",").split(",")]
    return [token for token in tokens if token]


def _filter_mapping(mapping: Dict[str, str], filters: list[str]) -> Dict[str, str]:
    if not filters:
        return mapping
    filtered: Dict[str, str] = {}
    for logical_key, param_name in mapping.items():
        if any(token in logical_key or token in param_name for token in filters):
            filtered[logical_key] = param_name
    return filtered


class SelectiveLoraLoader(io.ComfyNode):
    """加载 LoRA 并按关键字筛选目标 diffusion 层。"""

    _cache: Dict[str, Dict] = {}

    @classmethod
    def _available_loras(cls) -> list[str]:
        try:
            return sorted(folder_paths.get_filename_list("loras"))
        except Exception as exc:  # pragma: no cover
            logging.warning("Failed to enumerate LoRA files: %s", exc)
            return []

    @classmethod
    def define_schema(cls) -> io.Schema:
        lora_options = cls._available_loras() or ["<none>"]
        return io.Schema(
            node_id="LoadLoraQwenImage",
            display_name="LoadLoraQwenImage (Selective Qwen Image layers from ai-toolkit)",
            category="loaders",
            inputs=[
                io.Model.Input("model", optional=True),
                io.Combo.Input("lora_name", options=lora_options),
                io.Float.Input("strength_model", default=1.0, min=-10.0, max=10.0, step=0.05),
                io.String.Input(
                    "layer_filter",
                    multiline=True,
                    default="",
                    lazy=True,
                    tooltip="多关键字用逗号或换行分隔，匹配目标权重路径（如 transformer_blocks.0）。留空表示应用全部层。",
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        lora_name: str,
        strength_model: float,
        layer_filter: str,
    ) -> io.NodeOutput:
        if not lora_name or lora_name == "<none>":
            return io.NodeOutput(model)

        if strength_model == 0.0:
            return io.NodeOutput(model)

        filters = _split_filters(layer_filter)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if lora_path in cls._cache:
            lora_state = cls._cache[lora_path]
        else:
            logging.info("Loading LoRA: %s", lora_name)
            raw_state = utils.load_torch_file(lora_path, safe_load=True)
            lora_state = comfy.lora_convert.convert_lora(raw_state)
            cls._cache[lora_path] = lora_state

        if model is not None:
            base_model = getattr(model, "model", model)
            model_map = _filter_mapping(comfy.lora.model_lora_keys_unet(base_model), filters)
            if model_map:
                patches = comfy.lora.load_lora(lora_state, model_map, log_missing=False)
                target_model = model.clone() if hasattr(model, "clone") else model
                target_model.add_patches(patches, strength_model)
                model = target_model
            else:
                logging.info("No diffusion model weights matched filter for LoRA %s", lora_name)

        return io.NodeOutput(model)