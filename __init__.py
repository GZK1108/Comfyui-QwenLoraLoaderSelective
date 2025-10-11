from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes import SelectiveLoraLoader


class SelectiveLoraLoaderExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SelectiveLoraLoader]


async def comfy_entrypoint() -> SelectiveLoraLoaderExtension:
    return SelectiveLoraLoaderExtension()
