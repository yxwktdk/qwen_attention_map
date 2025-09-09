import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from functools import partial
from qwen_vl_utils import process_vision_info
import math
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


# 导入 ViT 所需的旋转位置编码（RoPE）应用函数
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision


# --- 1. 配置区域 ---
# 模型路径、图片路径、提示词和输出设置
MODEL_PATH = "/scratch/stu2/Qwen"
# IMAGE_PATH = "/scratch/stu2/math_data/bird.png"
# IMAGE_PATH = "/scratch/stu2/math_data/autogeo/images/1.png"
IMAGE_PATH = "/scratch/stu2/math_data/mathverse/images/image_1.png"
# PROMPT = "What is in picture?"
PROMPT = "What is the length of side BC?"
OUTPUT_FILENAME_BASE = "attention_visualization/"
# 选择你希望可视化的视觉编码器层（索引从0开始）
LAYERS_TO_VISUALIZE = [7, 15, 23, 31]

# 用于在模型前向传播期间捕获注意力图的全局变量
captured_attention_map = None

# --- 2. 加载模型和处理器 ---
print("正在加载模型和处理器...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("模型加载成功！")

# --- 3. 定义用于替换的自定义前向传播方法 (v6, 终极调试版) ---
def patched_attention_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
    """
    这是一个为解码器层定制的自定义注意力前向传播函数。
    它使用独立的 q_proj, k_proj, v_proj 来计算 Q, K, V。
    修正了 rotary_emb 的调用方式。
    加入了处理分组查询注意力 (GQA) 的逻辑。
    新增了详细的形状打印，用于最终调试。
    """
    global captured_attention_map

    bsz, q_len, _ = hidden_states.size()

    # 使用独立的 q_proj, k_proj, v_proj
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # 重塑以适应多头注意力
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # 应用旋转位置编码 (RoPE)
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        # 在 generate 循环的解码步骤中, kv_seq_len 会加上缓存的长度
        kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # 缓存 K, V (如果需要)
    if past_key_value is not None:
        # update() 会将当前的 key_states/value_states 与缓存拼接
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

    # --- 新增：处理 GQA 的关键逻辑 ---
    num_key_value_groups = self.num_heads // self.num_key_value_heads
    key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
    value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)
    # --- GQA 处理结束 ---

    # --- 终极调试打印 ---
    print("\n" + "="*20 + " DEBUG INFO " + "="*20)
    print(f"Layer Index: {self.layer_idx}")
    print(f"Is using KV cache: {past_key_value is not None}")
    print(f"Query sequence length (q_len): {q_len}")
    print(f"Query states shape (Q): {query_states.shape}")
    print(f"Key states shape (K) after cache & GQA: {key_states.shape}")
    print(f"Value states shape (V) after cache & GQA: {value_states.shape}")
    print("="*52 + "\n")
    # --- 调试结束 ---

    # 计算注意力权重
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    # 应用 attention_mask
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # 使用 Softmax 归一化得到注意力图，并将其捕获到全局变量
    captured_attention_map = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # 应用注意力权重到 V
    attn_output = torch.matmul(captured_attention_map, value_states)

    # 后续处理
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value




# --- 4. 准备输入数据 ---
print("\n正在准备输入数据...")
try:
    image = Image.open(IMAGE_PATH).convert("RGB")
except FileNotFoundError:
    print(f"错误: 找不到图片文件，请检查路径: {IMAGE_PATH}")
    exit()

# messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
messages = [{"role": "user", "content": [{"type": "image", "image":IMAGE_PATH}, {"type": "text", "text": PROMPT}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)

# --- 5. 循环执行替换、推理和恢复操作 ---
all_attention_maps = {}

for layer_idx in LAYERS_TO_VISUALIZE:
    print("\n" + "="*40)
    print(f"         开始处理第 {layer_idx+1} 层 (索引 {layer_idx})         ")
    print("="*40)

    captured_attention_map = None

    try:
        # 定位到要修改的目标层
        # target_layer = model.visual.blocks[layer_idx].attn
        # print(f"定位到目标层: visual.blocks[{layer_idx}].attn")
        target_layer = model.model.layers[layer_idx].self_attn
        print(f"定位到目标层: model.model.layers[{layer_idx}].self_attn")

        # 保存原始的 forward 方法
        original_forward = target_layer.forward
        # 使用我们的自定义方法替换它
        target_layer.forward = partial(patched_attention_forward, target_layer)
        print("成功执行 Monkey Patch。")

    except Exception as e:
        print(f"Monkey Patch 失败: {e}")
        continue

    print("正在执行模型前向传播...")
    with torch.no_grad():
        outputs = model.generate(**inputs, output_attentions=True, return_dict_in_generate=True, max_new_tokens=128)
    print("前向传播完成。")

    # 恢复原始的 forward 方法
    target_layer.forward = original_forward
    print("已恢复原始 forward 方法。")

    if captured_attention_map is not None:
        all_attention_maps[layer_idx] = captured_attention_map
        print(f"第 {layer_idx+1} 层的注意力图已成功捕获并存储。")

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs[0])
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f"生成的回答: {output_text[0]}")

# --- 6. 循环处理并保存所有捕获到的注意力图 (新版) ---
print("\n" + "="*40)
print(f"         所有推理完成，开始生成可视化图片         ")
print("="*40)

# 首先，我们需要知道图像块的数量。这在ViT处理后是固定的。
# 通过模型配置可以获得，或者通过一次前向传播间接获得。
# 一个简单的方法是查看第一次捕获的注意力图的维度。
if not all_attention_maps:
    print("\n最终错误: 未能捕获到任何层的注意力图。")
else:
    # 从输入中获取文本token的数量
    num_text_tokens = inputs.input_ids.shape[1]
    
    for layer_idx, attn_map_raw in all_attention_maps.items():
        print(f"正在处理第 {layer_idx+1} 层的可视化...")

        # [B, H, N, N] -> [H, N, N]
        attn_map = attn_map_raw.squeeze(0).detach()
        
        # N 是总token数 = 图像块数 + 文本token数
        total_tokens = attn_map.shape[-1]
        num_image_patches = total_tokens - num_text_tokens

        # --- 关键的切片操作 ---
        # 我们想要的是 文本token (作为Query) 对 图像块 (作为Key/Value) 的注意力
        # 维度示意: attn_map[注意力头, Query位置, Key位置]
        # Query位置: 我们只关心文本token，它们在序列的末尾，所以是 [num_image_patches:]
        # Key位置: 我们只关心图像块，它们在序列的开头，所以是 [:num_image_patches]
        interest_region = attn_map[:, num_image_patches:, :num_image_patches]
        
        # 现在 interest_region 的形状是 [注意力头数, 文本token数, 图像块数]
        # 我们将所有文本token对图像块的注意力进行平均，得到每个图像块获得的“总关注度”
        attn_map_summary = interest_region.mean(dim=[0, 1]) # 在注意力和文本token维度上取平均
        
        # 检查图像块数量是否有效
        if num_image_patches <= 0:
            print(f"错误: 无法确定第 {layer_idx+1} 层的图像块数量。")
            continue

        # 动态计算网格形状（这部分逻辑和之前一样）
        factors = [i for i in range(1, int(num_image_patches**0.5) + 1) if num_image_patches % i == 0]
        if not factors:
            # 对于非平方数，寻找最接近平方的因子
            h_grid = int(num_image_patches**0.5)
            while num_image_patches % h_grid != 0:
                h_grid -= 1
        else:
            h_grid = factors[-1]
        w_grid = num_image_patches // h_grid

        attention_grid = attn_map_summary.reshape(h_grid, w_grid).cpu().to(torch.float32).numpy()

        # 上采样、着色、叠加（这部分逻辑和之前一样）
        original_image = np.array(image)
        h, w, _ = original_image.shape
        attention_heatmap = cv2.resize(attention_grid, (w, h), interpolation=cv2.INTER_CUBIC)

        attention_heatmap = (attention_heatmap - np.min(attention_heatmap)) / (np.max(attention_heatmap) - np.min(attention_heatmap))
        attention_heatmap_colored = cv2.applyColorMap(np.uint8(255 * attention_heatmap), cv2.COLORMAP_JET)

        superimposed_img = cv2.addWeighted(original_image, 0.6, attention_heatmap_colored, 0.4, 0)
        superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

        # 绘制 & 保存
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        ax1.imshow(original_image); ax1.set_title("Original Image"); ax1.axis('off')
        ax2.imshow(superimposed_img_rgb); ax2.set_title(f"Decoder Cross-Attention Heatmap (Layer {layer_idx+1})"); ax2.axis('off')
        plt.suptitle(f'Qwen2.5-VL Attention Visualization\nPrompt: "{PROMPT}"', fontsize=16)

        output_path = f"{OUTPUT_FILENAME_BASE}PROMPT_layer_{layer_idx+1}_cross_attention.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        print(f"第 {layer_idx+1} 层的可视化结果已成功保存到文件!")
        print(f"   文件路径: {os.path.abspath(output_path)}")