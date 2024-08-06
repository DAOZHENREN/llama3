# app.py

from flask import Flask, request, render_template, jsonify
import torch
import torch.distributed
from llama import Dialog, Llama
import os
from typing import List, Optional 
app = Flask(__name__)

# 配置模型参数
CKPT_DIR = "/root/llama-models/Meta-Llama-3.1-8B-Instruct"
TOKENIZER_PATH ="/root/llama-models/Meta-Llama-3.1-8B-Instruct/tokenizer.model"
TEMPERATURE = 0.6
TOP_P = 0.9
MAX_SEQ_LEN = 512
MAX_BATCH_SIZE = 4
MAX_GEN_LEN = 256

# 构建模型生成器
generator = Llama.build(
    ckpt_dir=CKPT_DIR,
    tokenizer_path=TOKENIZER_PATH,
    max_seq_len=MAX_SEQ_LEN,
    max_batch_size=MAX_BATCH_SIZE,
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        user_input = data.get('user_input', '')
        system_prompt = data.get('system_prompt', '')
        if system_prompt is None:
            dialogs: List[Dialog] = [
            [{"role":"system", "content":system_prompt},{"role": "user", "content": user_input}],
            [{"role":"system", "content":system_prompt},{"role": "user", "content": user_input}],
            [{"role":"system", "content":system_prompt},{"role": "user", "content": user_input}],
            [{"role":"system", "content":system_prompt},{"role": "user", "content": user_input}]
        ]
        else:
            dialogs: List[Dialog] = [
                [{"role": "user", "content": user_input}],
                [{"role": "user", "content": user_input}],
                [{"role": "user", "content": user_input}],
                [{"role": "user", "content": user_input}]
            ]
        results = generator.chat_completion(
            dialogs,
            max_gen_len=MAX_GEN_LEN,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        responses = [result['generation']['content']+"\n==================================\n" for result in results]
        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
        
        print("\n==================================\n")
        return jsonify({"response": responses})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=False)
