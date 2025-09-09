import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os # 确保os已导入

# --- 这是最终、修正后的可视化函数 ---
def visualize_unified_attention(image_path, generation_output, generated_token_ids, model, processor):
    """
    从一个统一的自注意力矩阵中提取并可视化文本对图像的注意力。
    适用于像Qwen2-VL这样的Decoder-Only LMM架构。

    Args:
        image_path (str): 原始图片路径。
        generation_output: model.generate函数的完整输出对象。
        generated_token_ids (list): 由模型新生成的token ID列表 (已去除prompt部分)。
        model: 模型对象。
        processor: 模型对应的processor。
    """
    # --- 1. 确定图像和文本token在完整序列中的位置 ---
    
    # 模型的总输入序列长度 (prompt + image patches)
    prompt_len = generation_output.sequences.shape[1] - len(generated_token_ids)
    
    # 从模型配置计算图像patch数量和网格尺寸
    vision_cfg = model.config.vision_config
    print(vision_cfg)
    patch_size = vision_cfg.patch_size
    image_size = 448
    grid_size = image_size // patch_size
    num_patches = grid_size * grid_size

    # 假设图像patches是输入序列的最开始部分
    # 注意：这在大多数模型中是正确的，但如果模型架构特殊，可能需要微调
    
    # --- 2. 智能选择要分析的tokens (处理句号) ---
    is_last_token_period = False
    if len(generated_token_ids) > 0 and processor.decode(generated_token_ids[-1]) == '<|im_end|>':
        is_last_token_period = True

    if is_last_token_period and len(generated_token_ids) > 1:
        tokens_to_analyze_ids = generated_token_ids[:-1]
        token_indices_in_full_sequence = range(prompt_len, generation_output.sequences.shape[1] - 1)
    else:
        tokens_to_analyze_ids = generated_token_ids
        token_indices_in_full_sequence = range(prompt_len, generation_output.sequences.shape[1])

    if tokens_to_analyze_ids is None or len(tokens_to_analyze_ids) == 0:
        print("没有可供分析的有效token注意力。")
        return
        
    tokens_for_viz_text = processor.decode(tokens_to_analyze_ids)
    print(f"\n正在为 tokens '{tokens_for_viz_text}' 生成聚合热力图...")

    # --- 3. 从 'attentions' 中提取、聚合 ---
    # `generation_output.attentions` 的结构: (每个生成token的注意力,)
    # 每个token的注意力又是: (每一层的注意力,)
    # 每一层的注意力张量形状: [batch, heads, seq_len, seq_len]
    
    aggregated_heatmap_vec = None

    for i, token_abs_idx in enumerate(token_indices_in_full_sequence):
        # 获取当前生成token对应的注意力元组
        token_attentions_tuple = generation_output.attentions[i]
        # 选择最后一层的注意力矩阵
        last_layer_attention = token_attentions_tuple[-1] # Shape: [1, heads, full_seq_len, full_seq_len]

        # 提取当前token(作为query)对所有图像patch(作为key/value)的注意力
        # Query index: token_abs_idx
        # Key indices: 0 to num_patches-1
        attention_to_patches = last_layer_attention[0, :, token_abs_idx, :num_patches].mean(dim=0).cpu()

        if aggregated_heatmap_vec is None:
            aggregated_heatmap_vec = attention_to_patches
        else:
            aggregated_heatmap_vec += attention_to_patches

    # 对所有分析的token求平均
    aggregated_heatmap_vec /= len(tokens_to_analyze_ids)
    
    # --- 4. 可视化 ---
    heatmap_2d = aggregated_heatmap_vec.reshape(grid_size, grid_size).numpy()
    image = Image.open(image_path).convert("RGB")
    heatmap_resized = Image.fromarray(heatmap_2d).resize(image.size, resample=Image.BICUBIC)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.imshow(np.array(heatmap_resized), alpha=0.6, cmap='viridis')
    ax.axis('off')
    save_path = f"./heatmaps/heatmap_{os.path.basename(image_path)}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

