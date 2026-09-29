import os
import sys
from openai import OpenAI
from utils import extract_resolved_code
sys.argv=['']
del sys

"""配置API key(通过环境变量 DEEPSEEK_API_KEY 提供；若是中转则需要配置 DEEPSEEK_BASE_URL)"""
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY"),
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

"""deepSeek 一次对话"""
def deepSeek(question, number, truth):

    all_responses = [] 
    print(question)
    print(f"\n<Truth Resolution>:-------->\n{truth}\n</Truth Resolution>-------->")
    for i in range(number):
        print(f"\n---------------ChatGPT的第{i + 1}个Resolution---------------")

        response = client.chat.completions.create(
            model="deepseek-v4-pro", # gpt-3.5-turbo
            messages=[
                {
                "role": "system", "content": "You are a senior Java program developer with extensive experience in resolving code conflicts caused by branch merging." 
                },
                {
                "role": "user", "content": question
                }
            ], 
            max_tokens = 4096, # This model's maximum context length is 4097 tokens.
            temperature = 1.99, # 0-2之间，越大越随机，越小越确定 0.01
        )
        
        current_response = response.choices[0].message.content
        print(current_response)
        only_code = extract_resolved_code(current_response)
        all_responses.append(only_code)
    return all_responses



