import os, openai, time

# 确保已经 export 过环境变量
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY 为空，请先 export key")

# 下面从 evaluator.py 里拷贝出来 Close‐set 的系统 prompt＋分类提示
close_set_template = """Given the following free-form description of a 3D object, please determine the most probable class index from the following 40 available categories, even if the description doesn't clearly refer to any one of them. Make your best-educated guess based on the information provided. If the description already contains a valid index, then the index should be selected. If it contains more than one valid index, then randomly select one index (specify your reason). If there is no valid index and it cannot be inferred from the information, return '-1#NA#Cannot infer'.
Categories:
0: airplane
1: bathtub
2: bed
3: bench
4: bookshelf
5: bottle
6: bowl
7: car
8: chair
9: cone
10: cup
11: curtain
12: desk
13: door
14: dresser
15: flower pot
16: glass box
17: guitar
18: keyboard
19: lamp
20: laptop
21: mantel
22: monitor
23: night stand
24: person
25: piano
26: plant
27: radio
28: range hood
29: sink
30: sofa
31: stairs
32: stool
33: table
34: tent
35: toilet
36: tv stand
37: vase
38: wardrobe
39: xbox
Reply with the format of 'index#class#short reason (no more than 10 words)'.

Now analyze the following:
Input: {model_output}
Output: """

# 1. 在这里，把 {model_output} 换成你 JSON 里第一条的 model_output
first_model_output = "This is a detailed 3D model of an airplane, painted primarily in black. The model showcases typical airplane features including a fuselage, wings, tail, and engines. It appears to be a passenger plane, given its size and design. This model could be used for various purposes such as for an animated simulation, educational purposes, or for rendering in a digital art project."

prompt = close_set_template.format(model_output=first_model_output)

print("[manual_test] “第一条” close‐set prompt：")
print(prompt)
print("\nNow send to OpenAI…\n")

start = time.time()
try:
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo-1106",  # 或者你想用的具体版本
        messages=[{"role":"user", "content": prompt}],
        timeout=15
    )
    elapsed = time.time() - start
    answer = resp.choices[0].message.content.strip()
    print(f"[manual_test] OpenAI 返回（耗时 {elapsed:.2f}s）:\n{answer}")
except Exception as e:
    print(f"[manual_test] 调用失败，异常：{e}")
