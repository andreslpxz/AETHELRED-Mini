import sentencepiece as spm
import argparse
import os

def train_tokenizer(input_file, model_prefix, vocab_size=32000):
    cmd = f"--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} " \
          f"--model_type=bpe --character_coverage=1.0 --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3"
    spm.SentencePieceTrainer.train(cmd)
    print(f"Tokenizer trained and saved with prefix: {model_prefix}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw text file(s)")
    parser.add_argument("--model_prefix", type=str, default="spm", help="Prefix for model and vocab files")
    parser.add_argument("--vocab_size", type=int, default=32000)
    args = parser.parse_args()

    train_tokenizer(args.input, args.model_prefix, args.vocab_size)
