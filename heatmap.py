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

# --- 1. 配置区域 (保持不变) ---
MODEL_PATH = "/scratch/stu2/Qwen"
IMAGE_PATH = "/scratch/stu2/math_data/mathverse/images/image_1.png"
PROMPT = "What is the length of side BC?"
OUTPUT_FILENAME_BASE = "attention_visualization/"
# 选择你希望可视化的解码器层 (0-27)
LAYERS_TO_VISUALIZE = [7, 15, 23, 31]


# --- 2. 加载模型和处理器 (保持不变) ---
print("正在加载模型和处理器...")
# 关键补充: 加载模型时，添加 attn_implementation="eager" 来强制使用能输出注意力的实现
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager" # <-- 添加这一行
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("模型加载成功！")

# --- 准备输入数据 ---
print("\n正在准备输入数据...")
try:
    image = Image.open(IMAGE_PATH).convert("RGB")
except FileNotFoundError:
    print(f"错误: 找不到图片文件，请检查路径: {IMAGE_PATH}")
    exit()

messages = [{"role": "user", "content": [{"type": "image", "image":IMAGE_PATH}, {"type": "text", "text": PROMPT}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)


# --- 新的执行与可视化逻辑 ---
print("\n" + "="*40)
print(f"      开始执行模型生成并捕获注意力      ")
print("="*40)

with torch.no_grad():
    # 一次性执行 generate，并让它返回所有层的注意力
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        return_dict_in_generate=True,
        output_attentions=True # <-- 请求输出注意力
    )

# `outputs.attentions` 是一个元组，每个元素对应生成的一个token
# 每个元素内部又是一个元组，对应模型的所有解码器层
# 我们关注最后一个token的注意力图，因为它综合了所有信息
last_token_attentions = outputs.attentions[-1] # 获取最后一个生成步骤的注意力

# 解码生成的文本
generated_ids = outputs.sequences
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"生成的回答:\n{output_text}")


print("\n" + "="*40)
print(f"         开始生成可视化图片         ")
print("="*40)

# 从输入中获取文本token的数量
num_text_tokens = inputs.input_ids.shape[1]

for layer_idx in LAYERS_TO_VISUALIZE:
    print(f"正在处理第 {layer_idx+1} 层的可视化...")

    # 从返回结果中直接获取指定层的注意力图
    attn_map_raw = last_token_attentions[layer_idx] # [batch, heads, q_len, kv_len]

    # [B, H, N, N] -> [H, N, N]
    attn_map = attn_map_raw.squeeze(0).detach().cpu()

    # 在 generate 的解码阶段，q_len 总是 1 (当前token)
    # kv_len 是总长度 (图像块 + 所有文本token)
    # 我们要提取的是这 1 个 query token 对所有 key/value token 的注意力
    total_tokens = attn_map.shape[-1]
    num_image_patches = total_tokens - num_text_tokens

    # 提取对图像块的注意力部分 [Heads, 1, Num_Image_Patches]
    interest_region = attn_map[:, 0, :num_image_patches]

    # 平均所有注意力头，得到每个图像块的关注度 [Num_Image_Patches]
    attn_map_summary = interest_region.mean(dim=0)

    if num_image_patches <= 0:
        print(f"错误: 第 {layer_idx+1} 层未能解析出图像块。")
        continue

    # 动态计算网格形状
    h_grid = int(num_image_patches**0.5)
    while num_image_patches % h_grid != 0:
        h_grid -= 1
    w_grid = num_image_patches // h_grid

    attention_grid = attn_map_summary.reshape(h_grid, w_grid).to(torch.float32).numpy()

    # 上采样、着色、叠加
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
    ax2.imshow(superimposed_img_rgb); ax2.set_title(f"Decoder Attention Heatmap (Layer {layer_idx+1})"); ax2.axis('off')
    plt.suptitle(f'Qwen2.5-VL Attention Visualization\nPrompt: "{PROMPT}"', fontsize=16)
    
    # 清理prompt中的特殊字符，使其可用作文件名
    safe_prompt = "".join(c for c in PROMPT if c.isalnum() or c in (' ', '_')).rstrip()
    output_path = f"{OUTPUT_FILENAME_BASE}{safe_prompt}_layer_{layer_idx+1}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    print(f"第 {layer_idx+1} 层的可视化结果已成功保存到文件!")
    print(f"   文件路径: {os.path.abspath(output_path)}")

print("\n处理完成！")