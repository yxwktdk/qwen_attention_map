from modelscope import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
from tqdm import tqdm
import re
import os
from visual_attention import visualize_unified_attention

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/scratch/stu2/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto", attn_implementation="eager" 
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("/scratch/stu2/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
# 读取ouput_merged.jsonl文件
question_file = "/scratch/stu2/math_data/test.jsonl"

answer_file_name = './qwen2vl_answers_topp095.jsonl'
answer_file = os.path.expanduser(answer_file_name)

os.makedirs(os.path.dirname(answer_file), exist_ok=True)
ans_file = open(answer_file, "w")

questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]


for line in tqdm(questions[19:20]):

    # print(line)
    idx = line["pid"]
    image_file = os.path.join("/scratch/stu2/math_data", line["Image"])
    qs = line["Question"]
    cur_prompt = qs
    qs = """Please answer the question directly based on the image, without any reasoning or calculation. """ + qs+"""If the subject in the question does not exist or the prerequisite is wrong, answer 'wrong prerequisite'.If the prerequisite is correct, but the answer is not explicitly provided in the image, answer 'not given'."""+"""The answer should be a word or a letter or a number, do not add any other context,such as 'Answer:'."""

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

    # generated_ids = model.generate(**inputs, max_new_tokens=128)
    generation_output = model.generate(
        **inputs, 
        max_new_tokens=128,
        temperature=0.7,
        top_k=5,
        top_p=0.95,
        output_attentions=True,
        return_dict_in_generate=True,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generation_output.sequences)
    ][0] # 直接取第一个样本
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 根据判断结果，选择要传入可视化函数的
    # --- 调用新的可视化函数 ---
    try:
        visualize_unified_attention(image_file, generation_output, generated_ids_trimmed, model, processor)
    except Exception as e:
        print(f"为问题ID {idx} 生成热力图失败: {e}")
        import traceback
        traceback.print_exc() # 打印详细错误，方便调试
    
    ans_file.write(json.dumps({"question_id": idx,
                                "prompt": cur_prompt,
                                "predict": output_text[0],
                                "answer": line["Answer"],
                                "type":line["Type"],
                                "category":line["Category"],
                                "subcategory":line["SubCategory"],
                                "ID":line["ID"],
                                "image_path": line["Image"]}) + "\n")
    ans_file.flush()
ans_file.close()

