torchrun --nproc_per_node=1 /root/llama3/example_text_completion.py \
--ckpt_dir "/root/Meta-Llama-3-8B" --tokenizer_path "/root/Meta-Llama-3-8B/original/tokenizer.model" \
--temperature 0.7 --top_p 0.9 --max_seq_len 256 --max_gen_len 128 --max_batch_size 4
