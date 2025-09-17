# 假设 tokenizer 已经加载好了
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/scratch/stu2/Qwen2-VL-7B-Instruct")

# 定义关键的视觉开始标记
vision_start_token = '<|vision_start|>'
vision_end_token = '<|vision_end|>'

# 从分词器的词汇表中获取该标记对应的 ID
# 注意：一些分词器可能需要设置 add_special_tokens=False
try:
    vision_start_token_id = tokenizer.convert_tokens_to_ids(vision_start_token)
    print(f"'{vision_start_token}' 对应的 token_id 是: {vision_start_token_id}")
    vision_end_token_id = tokenizer.convert_tokens_to_ids(vision_end_token)
    print(f"'{vision_end_token}' 对应的 token_id 是: {vision_end_token_id}")
except Exception as e:
    print(f"无法获取 token_id: {e}")
    # 如果出错，可能需要检查你的分词器是否真的定义了这个 special token