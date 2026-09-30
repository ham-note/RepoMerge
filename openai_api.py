import os
import sys
from openai import OpenAI
from utils import extract_resolved_code
import time
sys.argv=['']
del sys

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai-proxy.org/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
)

print(f"""OpenAI API_KEY Configuration completed in {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}""")

"""ChatGPT(GPT-3.5) 一次对话"""
def gpt_35(question, number, truth):

    all_responses = [] 
    print(question)
    print(f"\n<Truth Resolution>:-------->\n{truth}\n</Truth Resolution>-------->")
    for i in range(number):
        print(f"\n---------------ChatGPT的第{i + 1}个Resolution---------------")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # gpt-3.5-turbo
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


"""ChatGPT(GPT-4o) 一次对话"""
def gpt_4o(question, number, truth):

    all_responses = [] 
    print(question)
    print(f"\n<Truth Resolution>:-------->\n{truth}\n</Truth Resolution>-------->")
    for i in range(number):
        print(f"\n---------------ChatGPT的第{i + 1}个Resolution---------------")

        response = client.chat.completions.create(
            model="gpt-4o", # gpt-3.5-turbo gpt-4o
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