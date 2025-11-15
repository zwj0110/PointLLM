import os, openai

# 如果环境变量没生效，这里打印应该为空
print("Shell 里读到的 OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

# 下面尝试直接调用一次 ChatGPT
openai.api_key = os.getenv("OPENAI_API_KEY")
resp = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "一句话介绍下 OpenAI ChatGPT。"}
    ]
)
print("ChatGPT 的回答：", resp.choices[0].message.content)
