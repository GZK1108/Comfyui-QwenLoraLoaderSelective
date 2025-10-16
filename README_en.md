# LordLoraQwenImage 节点文档（Selective Qwen Image layers from ai-toolkit）

`LordLoraQwenImage` 是专为 Qwen Image Edit 工作流定制的 ComfyUI 节点，可根据关键字筛选仅在指定的 UNet transformer 层注入 Qwen Image 编辑类 LoRA，从而保持其余网络权重不受影响，实现更精准的编辑效果。

---

## 功能概览

- 🎯 **层级精细控制**：通过 `layer_filter` 关键字列表匹配目标层（例如 `transformer_blocks.0`），可以自由控制Lora应用的层级。
- 🧩 **Qwen Image Edit 适配**：针对 ai-toolkit 发布的 Qwen Image 编辑 LoRA 进行了优化，可直接融入官方或社区工作流。

---

## 输入与输出

| 名称 | 类型 | 说明 |
|------|------|------|
| `model` | MODEL（可选） | 需要注入 LoRA 的扩散模型，可直接连接 `Checkpoint Loader` 等上游节点的 `MODEL` 输出。 |
| `lora_name` | Combo | 选择要加载的 LoRA 文件，来源于 `ComfyUI/models/loras` 目录。 |
| `strength_model` | FLOAT | 控制 LoRA 对模型的影响强度，默认 1.0，支持负值。 |
| `layer_filter` | STRING（多行） | 可选。以逗号或换行分隔关键字，用于匹配目标权重路径；留空则对全部匹配层生效。 |

输出：

- `model` — 注入 LoRA 后的模型，可继续连接到采样器或其它节点。

---

## 适用范围

- **推荐**：Qwen Image Edit / Qwen Image-VL 编辑类 LoRA，尤其是 ai-toolkit 发布的风格或指令编辑模型。
- **兼容**：遵循 Stable Diffusion / SDXL 命名规则的 LoRA，只要层命名与关键字匹配即可。

### Qwen Image 模型说明

Qwen Image 与 Qwen Image Edit 的 UNet 主干共有 **60 个 transformer block**，编号范围为 `transformer_blocks.0` 至 `transformer_blocks.59`。在 `layer_filter` 中可以填写任意一个或多个目标编号，例如：

```text
transformer_blocks.0,
transformer_blocks.5,
transformer_blocks.0, transformer_blocks.12
```

若留空，LoRA 将作用到全部可匹配的 transformer 层。
- **兼容**：遵循 Stable Diffusion / SDXL 命名规则的 LoRA，只要层命名与关键字匹配即可。
---

## 使用步骤

1. 将 `Comfyui-QwenLoraLoaderSelective` 文件夹复制到 `ComfyUI/custom_nodes`。
2. 启动 ComfyUI。节点位于 **loaders** 分类，名称为 **LordLoraQwenEdit (Selective Qwen Edit layers from ai-toolkit)**。
3. 在工作流中配置：
   1. 连接上游 `MODEL` 输出（例如 `Checkpoint Loader (Qwen Image)`）。
   2. 在 `lora_name` 中选择需要使用的 LoRA 文件。
   3. 在 `layer_filter` 中填写层关键字，例如：

      ```text
      transformer_blocks.0
      cross_attention
      ```

   4. 调整 `strength_model` 以控制作用强度。
4. 将节点输出连接至 `KSampler` 等采样节点，完成后续编辑流程。

节点示例：

![节点示例](images/nodes_image.png)


### 示例工作流

```text
Checkpoint Loader (Qwen Image) → LordLoraQwenEdit → KSampler → VAE Decode → Save Image
```

若 `layer_filter = transformer_blocks.0`，则 LoRA 仅作用于第一个 transformer block，适合对局部风格或特定指令进行微调。

---

## 故障排查

- **LoRA 未生效**：检查 `layer_filter` 是否匹配正确的权重命名；留空测试是否可全量应用。
- **加载报错**：确认 LoRA 文件位于 `models/loras`，且与 Qwen Image 模型兼容；若日志提示某些键缺失，请调整关键字。

---

## 许可证

MIT开源许可。使用任何 LoRA 资源时，请遵循其各自的授权条款。

> **补充说明**：本节点目前使用 ai-toolkit 训练框架生成的 Qwen Image LoRA 进行测试与验证。其他训练框架生成的 LoRA 可能存在不兼容情况，请自行测试。如遇问题，欢迎提交 issue 反馈，我们会协助排查。
