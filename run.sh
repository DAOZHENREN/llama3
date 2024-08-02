torchrun /root/llama3/example_text_completion.py \
--ckpt_dir "/root/Meta-Llama-3-8B" --tokenizer_path "/root/Meta-Llama-3-8B/original/tokenizer.model" \
--temperature 0.6 --top_p 0.9 --max_seq_len 128 --max_gen_len 64 --max_batch_size 4
