import argparse
import os
import struct
import time

import constriction
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DEFAULT_MODEL = "distilgpt2"
DEFAULT_CONTEXT_SIZE = 1024
DEFAULT_STRIDE = 512


def load_model(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} on {device}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, tokenizer, device


def _get_next_probs(model, input_ids: torch.Tensor, past_kv=None):
    with torch.no_grad():
        out = model(input_ids, past_key_values=past_kv, use_cache=True)
        logits = out.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)[0]
        probs = np.clip(probs, 1e-9, 1.0)
        probs /= probs.sum()
    return probs, out.past_key_values


def _reset_context(model, context_tokens, device: torch.device):
    input_ids = torch.tensor([context_tokens], device=device)
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        logits = out.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)[0]
        probs = np.clip(probs, 1e-9, 1.0)
        probs /= probs.sum()
    return probs, out.past_key_values, len(context_tokens)


def compress_file(
    model,
    tokenizer,
    device: torch.device,
    input_path: str,
    output_path: str,
    context_size: int,
    stride: int,
):
    started = time.perf_counter()

    with open(input_path, "rb") as f:
        raw_bytes = f.read()

    text = raw_bytes.decode("utf-8", errors="replace")
    tokens = tokenizer.encode(text)

    num_tokens = len(tokens)
    original_size = len(raw_bytes)

    if tokenizer.decode(tokens).encode("utf-8") != raw_bytes:
        print("Warning: tokenization roundtrip is not byte-perfect for this input.")

    print(f"Compressing {input_path} -> {output_path}")
    print(f"Original: {original_size:,} bytes, tokens: {num_tokens:,}")

    encoder = constriction.stream.queue.RangeEncoder()
    bos_id = tokenizer.eos_token_id
    if bos_id is None:
        raise ValueError("Tokenizer has no eos_token_id.")

    past_kv = None
    kv_len = 0
    all_seen = [bos_id]

    for i, token in enumerate(tokens):
        if kv_len >= context_size:
            context = all_seen[-stride:]
            probs, past_kv, kv_len = _reset_context(model, context, device)
        else:
            input_ids = torch.tensor([[all_seen[-1]]], device=device)
            probs, past_kv = _get_next_probs(model, input_ids, past_kv)
            kv_len += 1

        dist = constriction.stream.model.Categorical(probs, perfect=False)
        encoder.encode(int(token), dist)
        all_seen.append(token)

        if (i + 1) % 500 == 0:
            print(f"  encoded {i + 1}/{num_tokens}")

    compressed_words = encoder.get_compressed()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", original_size))
        f.write(struct.pack("<I", num_tokens))
        f.write(compressed_words.tobytes())

    took = time.perf_counter() - started
    compressed_size = os.path.getsize(output_path)
    ratio = (original_size / compressed_size) if compressed_size else 0.0
    print(f"Done in {took:.2f}s")
    print(f"Compressed: {compressed_size:,} bytes, ratio: {ratio:.3f}x")


def decompress_file(
    model,
    tokenizer,
    device: torch.device,
    input_path: str,
    output_path: str,
    context_size: int,
    stride: int,
):
    started = time.perf_counter()

    with open(input_path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError("Invalid compressed file: missing header.")
        original_size, num_tokens = struct.unpack("<II", header)
        bits = np.frombuffer(f.read(), dtype=np.uint32)

    print(f"Decompressing {input_path} -> {output_path}")
    print(f"Tokens: {num_tokens:,}, expected bytes: {original_size:,}")

    decoder = constriction.stream.queue.RangeDecoder(bits)
    bos_id = tokenizer.eos_token_id
    if bos_id is None:
        raise ValueError("Tokenizer has no eos_token_id.")

    past_kv = None
    kv_len = 0
    all_seen = [bos_id]
    decoded_tokens = []

    for i in range(num_tokens):
        if kv_len >= context_size:
            context = all_seen[-stride:]
            probs, past_kv, kv_len = _reset_context(model, context, device)
        else:
            input_ids = torch.tensor([[all_seen[-1]]], device=device)
            probs, past_kv = _get_next_probs(model, input_ids, past_kv)
            kv_len += 1

        dist = constriction.stream.model.Categorical(probs, perfect=False)
        token = int(decoder.decode(dist))

        decoded_tokens.append(token)
        all_seen.append(token)

        if (i + 1) % 500 == 0:
            print(f"  decoded {i + 1}/{num_tokens}")

    text = tokenizer.decode(decoded_tokens)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(text.encode("utf-8"))

    took = time.perf_counter() - started
    out_size = os.path.getsize(output_path)
    print(f"Done in {took:.2f}s")
    print(f"Decompressed size: {out_size:,} bytes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmc",
        description="LLM Compressor CLI (DistilGPT-2 + arithmetic coding)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    compress_parser = subparsers.add_parser("compress", help="Compress file")
    compress_parser.add_argument("input", help="Path to input file")
    compress_parser.add_argument("output", help="Path to output .bin file")
    compress_parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model name")
    compress_parser.add_argument(
        "--context-size", type=int, default=DEFAULT_CONTEXT_SIZE, help="KV context size"
    )
    compress_parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Context stride")

    decompress_parser = subparsers.add_parser("decompress", help="Decompress file")
    decompress_parser.add_argument("input", help="Path to input .bin file")
    decompress_parser.add_argument("output", help="Path to output file")
    decompress_parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model name")
    decompress_parser.add_argument(
        "--context-size", type=int, default=DEFAULT_CONTEXT_SIZE, help="KV context size"
    )
    decompress_parser.add_argument(
        "--stride", type=int, default=DEFAULT_STRIDE, help="Context stride"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.context_size <= 0 or args.stride <= 0:
        raise ValueError("--context-size and --stride must be > 0")

    model, tokenizer, device = load_model(args.model)

    if args.command == "compress":
        compress_file(
            model,
            tokenizer,
            device,
            args.input,
            args.output,
            args.context_size,
            args.stride,
        )
    elif args.command == "decompress":
        decompress_file(
            model,
            tokenizer,
            device,
            args.input,
            args.output,
            args.context_size,
            args.stride,
        )


if __name__ == "__main__":
    main()
