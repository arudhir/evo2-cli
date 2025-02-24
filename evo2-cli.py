#!/usr/bin/env python3
import os
import requests
import base64
import io
import numpy as np
import argparse
import tempfile
from math import ceil
from tqdm import tqdm
from Bio import SeqIO  # Requires: pip install biopython
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define chunk size (50 kb)
CHUNK_SIZE = 50000

def get_sequence(args):
    """Return a sequence from either a raw input string or a FASTA file."""
    if args.sequence:
        return args.sequence.strip()
    elif args.fasta:
        with open(args.fasta, "r") as handle:
            records = list(SeqIO.parse(handle, "fasta"))
            if not records:
                raise ValueError("No sequences found in the FASTA file.")
            if len(records) > 1:
                print("Multiple sequences found in FASTA file; using the first one.")
            return str(records[0].seq).strip()
    else:
        raise ValueError("Either a sequence or a FASTA file must be provided.")

def query_forward(sequence_chunk, url, key):
    """Submit a forward (embedding) request for a sequence chunk."""
    payload = {
        "sequence": sequence_chunk,
        "output_layers": ["blocks.25.mlp.l3"],
    }
    response = requests.post(
        url=url,
        headers={"Authorization": f"Bearer {key}"},
        json=payload,
    )
    content_type = response.headers.get("Content-Type", "")
    if content_type.startswith("application/problem+json"):
        error_info = response.json()
        error_detail = error_info.get("detail", error_info)
        raise ValueError(f"Error from API: {error_detail}")
    elif "application/json" in content_type:
        response_json = response.json()
        npz_bytes = base64.b64decode(response_json["data"])
        npz_file = np.load(io.BytesIO(npz_bytes))
        embedding_output = npz_file[npz_file.files[0]]
        return embedding_output
    else:
        raise ValueError(f"Unexpected Content-Type: {content_type}")

def run_forward(args, key, url):
    """Process the forward pass: chunk sequence if needed and combine embeddings."""
    input_sequence = get_sequence(args)
    seq_length = len(input_sequence)
    print(f"Input sequence length: {seq_length} bases")

    if seq_length > CHUNK_SIZE:
        n_chunks = ceil(seq_length / CHUNK_SIZE)
        print(f"Sequence longer than {CHUNK_SIZE} bp, processing in {n_chunks} chunks...")
        embeddings = [None] * n_chunks

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {}
            for i in range(n_chunks):
                start = i * CHUNK_SIZE
                end = min((i + 1) * CHUNK_SIZE, seq_length)
                chunk_seq = input_sequence[start:end]
                future = executor.submit(query_forward, chunk_seq, url, key)
                future_to_index[future] = i

            for future in tqdm(as_completed(future_to_index), total=n_chunks, desc="Processing chunks"):
                i = future_to_index[future]
                try:
                    emb = future.result()
                    embeddings[i] = emb
                    print(f"Chunk {i+1}/{n_chunks} shape: {emb.shape}")
                except Exception as e:
                    print(f"Error processing chunk {i+1}: {e}")
                    return

        # Combine embeddings: if each chunk has shape (1, L, D), combine along axis=1.
        if embeddings[0].ndim == 3:
            final_embedding = np.concatenate(embeddings, axis=1)
        else:
            final_embedding = np.concatenate(embeddings, axis=0)
    else:
        print("Sequence length is within chunk size. Processing as a single request...")
        try:
            final_embedding = query_forward(input_sequence, url, key)
        except Exception as e:
            print(f"Error processing sequence: {e}")
            return

    print("Final combined embedding shape:", final_embedding.shape)
    print("Basic statistics:")
    print(f"  Mean: {np.mean(final_embedding):.4f}")
    print(f"  Std:  {np.std(final_embedding):.4f}")
    print(f"  Min:  {np.min(final_embedding):.4f}")
    print(f"  Max:  {np.max(final_embedding):.4f}")

    # Determine output file name.
    if args.output is None:
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            output_filename = tmp.name
    else:
        output_filename = args.output

    np.save(output_filename, final_embedding)
    print(f"Embedding saved to {output_filename}")

def run_generate(args, key, url):
    """Process a generation request and save the generated sequence as FASTA."""
    input_sequence = get_sequence(args)
    print(f"Input prompt sequence length: {len(input_sequence)} bases")

    payload = {
        "sequence": input_sequence,
        "num_tokens": args.num_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "random_seed": args.random_seed,
        "enable_logits": args.enable_logits,
        "enable_sampled_probs": args.enable_sampled_probs,
        "enable_elapsed_ms_per_token": args.enable_elapsed_ms_per_token,
    }

    response = requests.post(
        url=url,
        headers={"Authorization": f"Bearer {key}"},
        json=payload,
    )
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        response_json = response.json()
        generated_seq = response_json.get("sequence")
        if not generated_seq:
            print("No sequence generated in the response.")
            return
    elif "text/plain" in content_type:
        generated_seq = response.text.strip()
    else:
        print("Unexpected Content-Type:", content_type)
        return

    print("Generated sequence length:", len(generated_seq))

    # Determine output filename.
    if args.output is None:
        with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as tmp:
            output_filename = tmp.name
    else:
        output_filename = args.output

    # Save as FASTA format (wrap lines at 80 characters).
    with open(output_filename, "w") as f:
        f.write(">generated_sequence\n")
        for i in range(0, len(generated_seq), 80):
            f.write(generated_seq[i:i+80] + "\n")
    print(f"Generated sequence saved to {output_filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Evo2 client script for forward (embedding extraction) and generate modes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")

    # Forward subcommand
    forward_parser = subparsers.add_parser("forward", help="Run forward pass to extract embeddings from Evo2.")
    group_forward = forward_parser.add_mutually_exclusive_group(required=True)
    group_forward.add_argument("-s", "--sequence", type=str, help="Input DNA sequence.")
    group_forward.add_argument("-f", "--fasta", type=str, help="Path to a FASTA file containing the DNA sequence.")
    forward_parser.add_argument("-o", "--output", type=str, default=None, help="Output file for embedding (default: temporary file).")

    # Generate subcommand
    generate_parser = subparsers.add_parser("generate", help="Generate DNA sequence using Evo2.")
    group_generate = generate_parser.add_mutually_exclusive_group(required=True)
    group_generate.add_argument("-s", "--sequence", type=str, help="Input DNA sequence prompt.")
    group_generate.add_argument("-f", "--fasta", type=str, help="Path to a FASTA file containing the prompt sequence.")
    generate_parser.add_argument("--num_tokens", type=int, default=100, help="Number of tokens to generate (default: 100).")
    generate_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature (default: 0.7).")
    generate_parser.add_argument("--top_k", type=int, default=3, help="Top-k (default: 3).")
    generate_parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (default: 1.0).")
    generate_parser.add_argument("--random_seed", type=int, default=None, help="Random seed (optional).")
    generate_parser.add_argument("--enable_logits", action="store_true", help="Enable logits output.")
    generate_parser.add_argument("--enable_sampled_probs", action="store_true", help="Enable sampled probabilities.")
    generate_parser.add_argument("--enable_elapsed_ms_per_token", action="store_true", help="Enable per-token timing.")
    generate_parser.add_argument("-o", "--output", type=str, default=None, help="Output file for generated sequence FASTA (default: temporary file).")

    args = parser.parse_args()

    # Retrieve API key from environment or prompt the user.
    key = os.getenv("NGC_API_KEY") or input("Paste the Run Key: ")

    if args.command == "forward":
        # Forward endpoint for embeddings.
        url = os.getenv("URL", "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward").strip('"')
        run_forward(args, key, url)
    elif args.command == "generate":
        # Use the working generate endpoint.
        url = os.getenv("URL", "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate").strip('"')
        run_generate(args, key, url)

if __name__ == "__main__":
    main()

