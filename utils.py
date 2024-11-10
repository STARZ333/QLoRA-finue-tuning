# util.py

from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"
    # return f"你是一個文言文翻譯專家，以下為用戶和文言文翻譯專家的對話，請提供準確清晰完整，有邏輯的回答。USER: {instruction} ASSISTANT:"
    # return f"你是一個文言文翻譯專家，以下為用戶和文言文翻譯專家的對話，請提供準確清晰完整，有邏輯的回答。以下是一個例子， instruction: 高祖初，為內秘書侍禦中散。\n翻譯成現代文：,output: 高祖初年，任內秘書侍禦中散。USER: {instruction} ASSISTANT:"
    # return f"你是一個文言文翻譯專家，以下為用戶和文言文翻譯專家的對話，請提供準確清晰完整，有邏輯的回答。以下是兩個例子，1. instruction: 高祖初，為內秘書侍禦中散。\n翻譯成現代文：,  output: 高祖初年，任內秘書侍禦中散。2. instruction:它的旁邊有一顆小星，名叫長沙星，星不宜明，若與軫宿的四顆星一樣明亮，五顆星進入軫宿，錶示將有大的戰爭發生。\n這句話在中國古代怎麼說：output: 其旁有一小星，曰長沙，星星不欲明；明與四星等，若五星入軫中，兵大起。USER: {instruction} ASSISTANT:"

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig for QLoRA.'''
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,   # 启用双重量化
        bnb_4bit_quant_type='nf4',        # 使用NF4量化
        bnb_4bit_compute_dtype=torch.float16  # 计算时使用float16
    )
