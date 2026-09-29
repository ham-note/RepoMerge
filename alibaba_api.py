import os
import sys
from openai import OpenAI
from utils import extract_resolved_code
sys.argv=['']
del sys

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY", "YOUR_DASHSCOPE_API_KEY"),
    base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)

"""Qwen 一次对话"""
def qwen(question, number, truth):

    all_responses = [] 
    print(question)
    print(f"\n<Truth Resolution>:-------->\n{truth}\n</Truth Resolution>-------->")
    for i in range(number):
        print(f"\n---------------Qwen的第{i + 1}个Resolution---------------")

        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {
                "role": "system", "content": "You are a senior Java program developer with extensive experience in resolving code conflicts caused by branch merging." 
                },
                {
                "role": "user", "content": question
                }
            ], 
            max_tokens = 4096, # This model's maximum context length is 4097 tokens.
            temperature = 1.90, # 0-2之间，越大越随机，越小越确定 0.01
        )
        
        current_response = response.choices[0].message.content
        print(current_response)
        only_code = extract_resolved_code(current_response)
        all_responses.append(only_code)
    return all_responses





