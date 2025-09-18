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
import torch.nn.functional as F

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

for line in tqdm(questions[0:1]):
    idx = line["pid"]
    image_file = os.path.join("/scratch/stu2/math_data", line["Image"])
    qs = line["Question"]
    cur_prompt = qs
    qs = """Please answer the question directly based on the image, without any reasoning or calculation. """ + qs+"""If the subject in the question does not exist or the prerequisite is wrong, answer 'wrong prerequisite'.If the prerequisite is correct, but the answer is not explicitly provided in the image, answer 'not given'."""+"""The answer should be a word or a letter or a number, do not add any other context,such as 'Answer:'."""
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
    print(f"inputs.input_ids.shape: {inputs.input_ids.shape}, \nimage_inputs[0].size: {image_inputs[0].size}, \ninputs.pixel_values.shape: {inputs.pixel_values.shape}, \ninput.image_grid_thw: {inputs.image_grid_thw}, \noutputs.sequences.shape: {outputs.sequences.shape}, \noutputs.attentions.shape: {len(outputs.attentions)} {len(outputs.attentions[0])} {outputs.attentions[0][0].shape}")
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

    # 查看generated_ids_trimmed的句尾部分，去除最后的结束符和结束符前面的句号
    # period_pos = None
    # for i, token_id in enumerate(generated_ids_trimmed[0]):
    #     if token_id == period_token_id:
    #         period_pos = i
    #         break
    # if period_pos is None:
    #     period_pos = len(generated_ids_trimmed[0])
    # attentions = attentions[:period_pos-1]# 截断到句号位置

    attentions = attentions[0]
    # 注意力：(output_len ,num_layers, batch_size, num_heads, seq_len, seq_len)
    # 保留最后一层，将所有注意力头平均
    attentions_tensor = torch.stack(attentions)
    last_layer_attentions = attentions_tensor[-1, :, :, :, :]
    # squeeze 去掉批次维度
    last_layer_attentions = last_layer_attentions[0]
    averaged_attentions = torch.mean(last_layer_attentions, axis=0)
    
    attention_to_image = averaged_attentions[image_start_pos+1:image_end_pos,image_start_pos+1:image_end_pos]
    # attentions 现在是一个长度为解码器

    print("\n从最后一个输入token到每个视觉token的注意力分数:")
    print(attention_to_image.shape)
    
    # attention_to_map resize成原图大小，image_inputs[0].size
    target_size = image_inputs[0].size
    resized_attention = F.interpolate(
        attention_to_image.unsqueeze(0).unsqueeze(0),  # 添加批次和通道维度
        size=(target_size[0], target_size[1]),  # 目标大小为 (height, width)
        mode='bilinear',
        align_corners=False
    ).squeeze()  # 去掉批次和通道维度

    resized_attention = resized_attention.cpu().numpy()
    resized_attention = (resized_attention - np.min(resized_attention)) / (np.max(resized_attention) - np.min(resized_attention) + 1e-8)  # 归一化到0-1之间 
    print(f"resized_attention shape: {resized_attention.shape}, min: {np.min(resized_attention)}, max: {np.max(resized_attention)}")
    # 可视化注意力图
    plt.figure(figsize=(8, 8))
    plt.imshow(resized_attention, cmap='jet', alpha=0.5)  #
    plt.colorbar()
    plt.title('Attention Map Overlay')
    plt.axis('off')
    # 叠加在原图上
    original_image = Image.open(image_file).convert("RGB")
    original_image = original_image.resize(target_size)
    plt.imshow(original_image, alpha=0.5)
    plt.imshow(resized_attention, cmap='jet', alpha=0.5)  #
    plt.axis('off')
    # 保存注意力图
    attention_map_file = f"./attention_maps/attention_map_{idx}.png"
    os.makedirs(os.path.dirname(attention_map_file), exist_ok=True)
    plt.savefig(attention_map_file)
    plt.close()
    

    

ans_file.close()