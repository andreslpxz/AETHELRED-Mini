import os
import argparse
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

def process_data(input_file, tokenizer_path, output_path, seq_len=1024):
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    all_tokens = []
    print(f"Tokenizing {input_file}...")
    for line in tqdm(lines):
        tokens = sp.encode_as_ids(line.strip())
        if tokens:
            all_tokens.extend(tokens + [sp.eos_id()])

    # Chunking: we need seq_len + 1 tokens per chunk to get
    # input and target of length seq_len
    chunk_size = seq_len + 1
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    num_chunks = len(all_tokens) // chunk_size

    if num_chunks == 0:
        print("Data too small for the given sequence length.")
        return

    all_tokens = all_tokens[:num_chunks * chunk_size]
    chunks = all_tokens.reshape(-1, chunk_size)

    np.save(output_path, chunks)
    print(f"Processed data saved to {output_path}. Shape: {chunks.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=1024)
    args = parser.parse_args()

    process_data(args.input, args.tokenizer, args.output, args.seq_len)
