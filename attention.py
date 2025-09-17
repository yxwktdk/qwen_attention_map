from modelscope import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
from tqdm import tqdm
import re
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 默认设置：在可用设备上加载模型
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/scratch/stu2/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("/scratch/stu2/Qwen2-VL-7B-Instruct")

question_file = "/scratch/stu2/math_data/test.jsonl"
answer_file_name = './qwen2vl_answers_with_attention_zh.jsonl'
answer_file = os.path.expanduser(answer_file_name)

os.makedirs(os.path.dirname(answer_file), exist_ok=True)
ans_file = open(answer_file, "w")

questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]

for line in tqdm(questions[19:20]):
    idx = line["pid"]
    image_file = os.path.join("/scratch/stu2/math_data", line["Image"])
    qs = line["Question"]
    cur_prompt = qs
    # qs = """Please answer the question directly based on the image, without any reasoning or calculation. """ + qs+"""If the subject in the question does not exist or the prerequisite is wrong, answer 'wrong prerequisite'.If the prerequisite is correct, but the answer is not explicitly provided in the image, answer 'not given'."""+"""The answer should be a word or a letter or a number, do not add any other context,such as 'Answer:'."""
    # qs = "Describe the image in detail. "

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_file,
                    # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                },
                {"type": "text", "text": qs},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出并捕获完整的输出对象
    # 关键在于设置 `return_dict_in_generate=True` 和 `output_attentions=True`
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            # max_new_tokens=128, # 为生成设置一个合理的长度限制
            temperature=0.7,  # 中等随机性

            output_attentions=True,
            return_dict_in_generate=True,
        )
    # 依此输出：inputs.input_ids.shape, image_inputs[0].size, inputs.pixel_values.shape, input.image_grid_thw, outputs.sequences.shape,outputs.attentions[0][0].shape
    print(f"inputs.input_ids.shape: {inputs.input_ids.shape}, \nimage_inputs[0].size: {image_inputs[0].size}, \ninputs.pixel_values.shape: {inputs.pixel_values.shape}, \ninput.image_grid_thw: {inputs.image_grid_thw}, \noutputs.sequences.shape: {outputs.sequences.shape}, \noutputs.attentions[0][0].shape: {outputs.attentions[0][0].shape}")
    # --- 注意力提取与分析 ---
    
    # 1. 分离生成的 token 和注意力权重
    generated_ids = outputs.sequences
    attentions = outputs.attentions #[output_token_len,28,1,28,input_len,input_len]

    vision_start_token_id = 151652
    vision_end_token_id = 151653
    image_start_pos = inputs['input_ids'][0].tolist().index(vision_start_token_id)
    image_end_pos = inputs['input_ids'][0].tolist().index(vision_end_token_id)
    # 图片所在部分是[image_start_pos+1:image_end_pos]

    # 句号token_id 
    period_token_id = 13
    end_of_sentence_token_id = 151645
    # 首先提取文本
    # --- 用于保存预测结果的原始代码 ---
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    # 解码时跳过特殊字符，否则会看到 <|im_start|> 等内容
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # 输出结果包含了原始的 prompt，我们需要清理它
    # 一个简单的方法是找到生成的部分
    cleaned_output = output_text[0].split(qs)[-1].strip()

    ans_file.write(json.dumps({"question_id": idx,
                                "prompt": cur_prompt,
                                "predict": cleaned_output,
                                "answer": line["Answer"],
                                "type":line["Type"],
                                "category":line["Category"],
                                "subcategory":line["SubCategory"],
                                "ID":line["ID"],
                                "image_path": line["Image"]}, ensure_ascii=False) + "\n") # ensure_ascii=False 以正确显示中文
    ans_file.flush()

    # 删除
    

    
    first_token_attentions = attentions[0] # 这是一个长度为解码器层数的元组

    # 3. 选择一个层进行分析（例如，最后一层）
    # 最后一层通常最能反映模型的最终决策。
    # Qwen2-VL-7B 有 28 层 (索引 0-27)。我们选择 -1 层（即最后一层）。
    layer_idx = -1 
    last_layer_attention = first_token_attentions[layer_idx] # 形状: (批次大小, 注意力头数, 序列长度, 序列长度)
    
    # 在所有注意力头上取平均值
    avg_last_layer_attention = last_layer_attention.mean(dim=1).squeeze(0) # 形状: (序列长度, 序列长度)
    
    # 4. 识别哪些是视觉 token，哪些是文本 token
    # 模型首先处理视觉 token，然后是文本 token。
    # 我们可以通过检查输入张量来找到边界。
    
    # 从注意力图中获取总序列长度
    total_seq_len = avg_last_layer_attention.shape[0]

    # 获取文本 token 的数量
    # 注意：这里的 `input_ids` 包括了特殊的提示 (prompt) token
    num_text_tokens = inputs['input_ids'].shape[1]
    
    # 视觉 token 的数量就是总长度与文本 token 数量的差值
    num_visual_tokens = total_seq_len
    
    # print(f"总序列长度: {total_seq_len}")
    # print(f"视觉 token 数量: {num_visual_tokens}")
    # print(f"文本 token 数量: {num_text_tokens}")
    
    # 5. 提取从“即将生成的 token 位置”到“视觉 tokens”的注意力
    # 查询 token 是输入序列中的最后一个 token（在生成开始之前的位置）。
    # 它的索引是 `total_seq_len - 1` 或简写为 `-1`。
    # 我们想看这个 token 如何关注所有在它之前的 token。
    attention_from_last_token = avg_last_layer_attention[-1, :] # 形状: (total_seq_len,)
    
    # 现在，对这个向量进行切片，只保留对应于视觉 token 的分数
    attention_to_image = attention_from_last_token[:num_visual_tokens] # 形状: (num_visual_tokens,)
    
    print("\n从最后一个输入token到每个视觉token的注意力分数:")
    print(attention_to_image.shape)
    
    # (可选) 你可以保存这些注意力分数或将其可视化
    # 为了可视化，你需要将视觉 token 重塑为一个二维网格。
    # 网格大小是 num_visual_tokens 的平方根。这在它是一个完全平方数时有效（例如, 256 -> 16x16）。
    grid_size = int(np.sqrt(num_visual_tokens))
    if grid_size * grid_size == num_visual_tokens:
    # 先用 .to(torch.float32) 将数据类型转为 float32, 然后再调用 .numpy()
        attention_map = attention_to_image.cpu().to(torch.float32).numpy().reshape(grid_size, grid_size)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(attention_map, cmap='hot')
        plt.title(f'注意力图 (第 {layer_idx} 层) - 图像 ID {idx}')
        plt.colorbar()
        # 保存图像
        plot_filename = f'./attention_map_{idx}_layer_{layer_idx}.png'
        plt.savefig(plot_filename)
        print(f"\n已将注意力可视化图像保存至 {plot_filename}")
        plt.close()

    

ans_file.close()