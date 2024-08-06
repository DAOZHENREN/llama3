from flask import Flask, request, render_template_string
import torch
from llama import Llama
from typing import List

app = Flask(__name__)

# Configuration for the generator
CKPT_DIR = '/root/llama-models/Meta-Llama-3.1-8B'
TOKENIZER_PATH = '/root/llama-models/Meta-Llama-3.1-8B-Instruct/tokenizer.model'
TEMPERATURE = 0.6
TOP_P = 0.9
MAX_SEQ_LEN = 7* 1024
MAX_GEN_LEN = 7* 1024
MAX_BATCH_SIZE = 4

# Initialize the generator
generator = Llama.build(
    ckpt_dir=CKPT_DIR,
    tokenizer_path=TOKENIZER_PATH,
    max_seq_len=MAX_SEQ_LEN,
    max_batch_size=MAX_BATCH_SIZE,
)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        prompt = request.form['prompt']
        results = generator.text_completion(
            [prompt],
            max_gen_len=MAX_GEN_LEN,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        result = results[0]['generation']
    
    return render_template_string('''
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
            <title>Text Generator</title>
          </head>
          <body>
            <div style="max-width: 600px; margin: auto; padding: 20px;">
              <h1>Text Generator</h1>
              <form method="post">
                <div>
                  <label for="prompt">Enter your prompt:</label>
                  <textarea id="prompt" name="prompt" rows="4" cols="50" style="width: 100%;">{{ request.form.get('prompt', '') }}</textarea>
                </div>
                <br>
                <button type="submit">Generate</button>
              </form>
              {% if result %}
              <h2>Generated Text:</h2>
              <p>{{ result }}</p>
              {% endif %}
            </div>
          </body>
        </html>
    ''', result=result)

if __name__ == '__main__':
    app.run(debug=False,port=4000)
