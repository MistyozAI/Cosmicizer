"""
Enhanced dataset preparation script for OpenWebText and Wikipedia.
Handles both individual dataset processing and mixed dataset creation.
"""

import os
import argparse
from tqdm.auto import tqdm
import numpy as np
import tiktoken
import multiprocessing
import psutil
import time
import pickle
import json
import logging
import sys
import random
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class PreparationConfig:
    dataset_name: str = "mixed"  # "openwebtext", "wikipedia", or "mixed"
    test_size: float = 0.0005  # Fraction to use for validation
    seed: int = 2357
    num_proc: int = multiprocessing.cpu_count()  # Use all available processors
    batch_size: int = 1024
    output_dir: str = "data"
    max_seq_length: int = 512  # Reduced from 1024 per our plan
    wikipedia_weight: float = 0.6  # Weight given to Wikipedia in the mixed dataset


def log_memory_usage():
    """Get current memory usage"""
    process = psutil.Process()
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    return f"{memory_gb:.1f} GB"


def prepare_openwebtext(config):
    """Prepare the OpenWebText dataset"""
    from datasets import load_dataset

    logger.info("=== Preparing OpenWebText dataset ===")
    enc = tiktoken.get_encoding("gpt2")
    dataset_dir = os.path.join(config.output_dir, "openwebtext")
    os.makedirs(dataset_dir, exist_ok=True)

    logger.info(f"Using {config.num_proc} processors")
    logger.info(f"Current memory usage: {log_memory_usage()}")

    # Download OpenWebText dataset
    logger.info("Downloading OpenWebText dataset...")
    dataset = load_dataset("openwebtext", num_proc=config.num_proc, trust_remote_code=True)

    # Split dataset
    logger.info("Splitting dataset into train and validation sets...")
    split_dataset = dataset["train"].train_test_split(
        test_size=config.test_size,
        seed=config.seed,
        shuffle=True
    )
    split_dataset['val'] = split_dataset.pop('test')

    logger.info("Dataset statistics:")
    for split, data in split_dataset.items():
        logger.info(f"{split}: {len(data):,} examples")

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    logger.info(f"Current memory usage: {log_memory_usage()}")

    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}

    tokenized = {}
    for split, dset in split_dataset.items():
        logger.info(f"Tokenizing {split} split...")
        tokenized[split] = dset.map(
            process,
            remove_columns=['text'],
            num_proc=config.num_proc,
            desc=f"Tokenizing {split}"
        )

    # Write binary files
    logger.info("Writing binary files...")
    logger.info(f"Current memory usage: {log_memory_usage()}")

    total_tokens = {split: 0 for split in tokenized.keys()}

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(dataset_dir, f'{split}.bin')
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))

        idx = 0
        # Create smaller batch size if needed
        effective_batch_size = min(config.batch_size, len(dset))

        for batch_idx in tqdm(range(effective_batch_size), desc=f"Writing {split}.bin"):
            batch = dset.shard(
                num_shards=effective_batch_size,
                index=batch_idx,
                contiguous=True
            ).with_format('numpy')

            arr_batch = np.concatenate(batch['ids'])
            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
            total_tokens[split] += len(arr_batch)

        arr.flush()

    # Save metadata
    meta = {
        'vocab_size': enc.n_vocab,
        'total_tokens': total_tokens,
        'dataset_name': 'openwebtext',
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(os.path.join(dataset_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    logger.info("OpenWebText preparation completed!")
    logger.info("Token statistics:")
    for split, count in total_tokens.items():
        logger.info(f"{split}: {count:,} tokens")

    return total_tokens


def prepare_wikipedia(config):
    """Prepare the Wikipedia dataset"""
    from datasets import load_dataset

    logger.info("=== Preparing Wikipedia dataset ===")
    enc = tiktoken.get_encoding("gpt2")
    dataset_dir = os.path.join(config.output_dir, "wikipedia")
    os.makedirs(dataset_dir, exist_ok=True)

    logger.info(f"Using {config.num_proc} processors")
    logger.info(f"Current memory usage: {log_memory_usage()}")

    # Download Wikipedia dataset
    logger.info("Downloading Wikipedia dataset...")
    dataset = load_dataset("wikipedia", "20220301.en", num_proc=config.num_proc, trust_remote_code=True)

    # Clean Wikipedia text
    logger.info("Cleaning Wikipedia text...")

    def clean_wiki_text(example):
        # Remove references and other Wiki markup
        text = example['text']
        # Basic cleaning - can be expanded with more sophisticated regex
        text = text.replace("===", "").replace("==", "")
        text = text.replace("'''", "").replace("''", "")
        text = text.replace("[[", "").replace("]]", "")
        # Keep only the text, not the references or other metadata
        # Split on "References" or "See also" sections
        for split_text in ["References", "See also", "External links"]:
            if split_text in text:
                text = text.split(split_text)[0]
        return {"text": text}

    logger.info("Processing Wikipedia articles...")
    dataset = dataset.map(
        clean_wiki_text,
        num_proc=config.num_proc,
        desc="Cleaning Wikipedia text"
    )

    # Split dataset
    logger.info("Splitting dataset into train and validation sets...")
    split_dataset = dataset["train"].train_test_split(
        test_size=config.test_size,
        seed=config.seed,
        shuffle=True
    )
    split_dataset['val'] = split_dataset.pop('test')

    logger.info("Dataset statistics:")
    for split, data in split_dataset.items():
        logger.info(f"{split}: {len(data):,} examples")

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    logger.info(f"Current memory usage: {log_memory_usage()}")

    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}

    tokenized = {}
    for split, dset in split_dataset.items():
        logger.info(f"Tokenizing {split} split...")
        tokenized[split] = dset.map(
            process,
            remove_columns=['text', 'title', 'url', 'id'],
            num_proc=config.num_proc,
            desc=f"Tokenizing {split}"
        )

    # Write binary files
    logger.info("Writing binary files...")
    logger.info(f"Current memory usage: {log_memory_usage()}")

    total_tokens = {split: 0 for split in tokenized.keys()}

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(dataset_dir, f'{split}.bin')
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))

        idx = 0
        # Create smaller batch size if needed
        effective_batch_size = min(config.batch_size, len(dset))

        for batch_idx in tqdm(range(effective_batch_size), desc=f"Writing {split}.bin"):
            batch = dset.shard(
                num_shards=effective_batch_size,
                index=batch_idx,
                contiguous=True
            ).with_format('numpy')

            arr_batch = np.concatenate(batch['ids'])
            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
            total_tokens[split] += len(arr_batch)

        arr.flush()

    # Save metadata
    meta = {
        'vocab_size': enc.n_vocab,
        'total_tokens': total_tokens,
        'dataset_name': 'wikipedia',
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(os.path.join(dataset_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    logger.info("Wikipedia preparation completed!")
    logger.info("Token statistics:")
    for split, count in total_tokens.items():
        logger.info(f"{split}: {count:,} tokens")

    return total_tokens


def prepare_mixed_dataset(config):
    """Prepare both OpenWebText and Wikipedia, then create a weighted dataset info file"""

    # First, prepare individual datasets
    logger.info("Preparing mixed dataset (OpenWebText + Wikipedia)...")

    # Check if datasets already exist
    openwebtext_exists = (
            os.path.exists(os.path.join(config.output_dir, "openwebtext", "train.bin")) and
            os.path.exists(os.path.join(config.output_dir, "openwebtext", "val.bin"))
    )

    wikipedia_exists = (
            os.path.exists(os.path.join(config.output_dir, "wikipedia", "train.bin")) and
            os.path.exists(os.path.join(config.output_dir, "wikipedia", "val.bin"))
    )

    # Prepare OpenWebText if needed
    if not openwebtext_exists:
        logger.info("OpenWebText dataset not found, preparing it now...")
        config.dataset_name = "openwebtext"
        openwebtext_tokens = prepare_openwebtext(config)
    else:
        logger.info("OpenWebText dataset already exists, skipping preparation")
        with open(os.path.join(config.output_dir, "openwebtext", "meta.pkl"), 'rb') as f:
            meta = pickle.load(f)
            openwebtext_tokens = meta['total_tokens']

    # Prepare Wikipedia if needed
    if not wikipedia_exists:
        logger.info("Wikipedia dataset not found, preparing it now...")
        config.dataset_name = "wikipedia"
        wikipedia_tokens = prepare_wikipedia(config)
    else:
        logger.info("Wikipedia dataset already exists, skipping preparation")
        with open(os.path.join(config.output_dir, "wikipedia", "meta.pkl"), 'rb') as f:
            meta = pickle.load(f)
            wikipedia_tokens = meta['total_tokens']

    # Create mixed dataset metadata
    mixed_dir = os.path.join(config.output_dir, "mixed")
    os.makedirs(mixed_dir, exist_ok=True)

    # Record the weighting and token counts
    mixed_meta = {
        'openwebtext_tokens': openwebtext_tokens,
        'wikipedia_tokens': wikipedia_tokens,
        'wikipedia_weight': config.wikipedia_weight,
        'total_effective_tokens': {
            'train': int(openwebtext_tokens['train'] * (1 - config.wikipedia_weight) +
                         wikipedia_tokens['train'] * config.wikipedia_weight),
            'val': int(openwebtext_tokens['val'] * (1 - config.wikipedia_weight) +
                       wikipedia_tokens['val'] * config.wikipedia_weight)
        },
        'dataset_name': 'mixed',
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Save the mixed metadata
    with open(os.path.join(mixed_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(mixed_meta, f)

    # Also save as JSON for easier inspection
    with open(os.path.join(mixed_dir, 'meta.json'), 'w') as f:
        # Convert numpy values to Python types for JSON
        serializable_meta = {k: v if not isinstance(v, dict) else
        {k2: int(v2) if isinstance(v2, np.integer) else v2
         for k2, v2 in v.items()}
                             for k, v in mixed_meta.items()}
        json.dump(serializable_meta, f, indent=2)

    logger.info("Mixed dataset preparation completed!")
    logger.info(f"Effective token distribution:")
    logger.info(f"  Wikipedia: {config.wikipedia_weight * 100:.1f}%")
    logger.info(f"  OpenWebText: {(1 - config.wikipedia_weight) * 100:.1f}%")
    logger.info(f"Total effective tokens: {mixed_meta['total_effective_tokens']['train']:,}")

    return mixed_meta


def prepare_dataset(config):
    """Main dataset preparation function that handles all dataset types"""
    os.makedirs(config.output_dir, exist_ok=True)

    if config.dataset_name == "openwebtext":
        prepare_openwebtext(config)
    elif config.dataset_name == "wikipedia":
        prepare_wikipedia(config)
    elif config.dataset_name == "mixed":
        prepare_mixed_dataset(config)
    else:
        logger.error(f"Unknown dataset: {config.dataset_name}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for training")
    parser.add_argument("--dataset_name", type=str, default="mixed",
                        help="Dataset to prepare: 'openwebtext', 'wikipedia', or 'mixed' (default: mixed)")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for processed data (default: data)")
    parser.add_argument("--test_size", type=float, default=0.0005,
                        help="Validation split size (default: 0.0005)")
    parser.add_argument("--seed", type=int, default=2357,
                        help="Random seed for dataset splitting (default: 2357)")
    parser.add_argument("--num_proc", type=int,
                        default=multiprocessing.cpu_count(),
                        help="Number of processing cores to use (default: all available)")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for processing (default: 1024)")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    parser.add_argument("--wikipedia_weight", type=float, default=0.6,
                        help="Weight for Wikipedia in mixed dataset (default: 0.6)")

    args = parser.parse_args()

    config = PreparationConfig(
        dataset_name=args.dataset_name,
        test_size=args.test_size,
        seed=args.seed,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        wikipedia_weight=args.wikipedia_weight
    )

    # Print configuration
    logger.info("=== Dataset Preparation Configuration ===")
    for key, value in vars(config).items():
        logger.info(f"{key}: {value}")

    # Run dataset preparation
    prepare_dataset(config)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Error during dataset preparation: {str(e)}", exc_info=True)
        sys.exit(1)