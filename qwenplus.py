from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
from tqdm import tqdm
import re
import os
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/scratch/stu2/Qwen", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("/scratch/stu2/Qwen")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
question_file = "/scratch/stu2/math_data/test.jsonl"

answer_file_name = './qwen2.5vl_answers.jsonl'
answer_file = os.path.expanduser(answer_file_name)

os.makedirs(os.path.dirname(answer_file), exist_ok=True)
ans_file = open(answer_file, "w")

questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]

for line in tqdm(questions):
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
                },
                {"type": "text", "text": qs},
            ],
        }
    ]

    # Preparation for inference
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

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
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