"""
Train the Cosmicizer tokenizer using SentencePiece on prepared datasets.
This script samples from existing binary datasets, trains SentencePiece, and creates cosmicizer.pkl.
"""

import os
import sys
import argparse
import tempfile
import time
import random
import pickle
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import tiktoken
import sentencepiece as spm
from tqdm.auto import tqdm

from cosmicizer import Cosmicizer, save_cosmicizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def sample_pretrain_data(data_dir: str, max_samples: int = 5000000) -> List[str]:
    """
    Sample text from all pretraining datasets (OpenWebText + Wikipedia + Conversational).

    Args:
        data_dir: Directory containing mixed dataset metadata or direct dataset folder
        max_samples: Maximum number of samples to extract

    Returns:
        List of text samples
    """
    logger.info(f"Sampling pretraining data from {data_dir}")

    # Check if this is a mixed dataset (only has metadata)
    train_path = os.path.join(data_dir, 'train.bin')
    val_path = os.path.join(data_dir, 'val.bin')

    if not os.path.exists(train_path) and data_dir.endswith('mixed'):
        # This is a mixed dataset, sample from all three: openwebtext, wikipedia, and conversational
        logger.info("Detected mixed dataset, sampling from OpenWebText, Wikipedia, and Conversational data")

        base_dir = os.path.dirname(data_dir)  # Get parent directory (data/)
        openwebtext_dir = os.path.join(base_dir, 'openwebtext')
        wikipedia_dir = os.path.join(base_dir, 'wikipedia')
        conversational_dir = os.path.join(base_dir, 'conversational')

        # Sample from all three datasets with weights
        samples = []

        # Distribution: 35% OpenWebText, 45% Wikipedia, 20% Conversational
        owt_samples = int(max_samples * 0.35)
        wiki_samples = int(max_samples * 0.45)
        conv_samples = int(max_samples * 0.20)

        if os.path.exists(os.path.join(openwebtext_dir, 'train.bin')):
            logger.info(f"Sampling {owt_samples:,} samples from OpenWebText")
            owt_data = sample_single_dataset(openwebtext_dir, owt_samples)
            samples.extend(owt_data)

        if os.path.exists(os.path.join(wikipedia_dir, 'train.bin')):
            logger.info(f"Sampling {wiki_samples:,} samples from Wikipedia")
            wiki_data = sample_single_dataset(wikipedia_dir, wiki_samples)
            samples.extend(wiki_data)

        if os.path.exists(os.path.join(conversational_dir, 'train.bin')):
            logger.info(f"Sampling {conv_samples:,} samples from Conversational")
            conv_data = sample_single_dataset(conversational_dir, conv_samples, is_conversation=True)
            samples.extend(conv_data)

        if not samples:
            raise FileNotFoundError(f"No dataset files found in {openwebtext_dir}, {wikipedia_dir}, or {conversational_dir}")

        # Shuffle the combined samples
        import random
        random.shuffle(samples)
        logger.info(f"Combined total: {len(samples):,} samples from all datasets")
        return samples

    elif not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    elif not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    else:
        # Single dataset
        return sample_single_dataset(data_dir, max_samples)


def sample_single_dataset(data_dir: str, max_samples: int, is_conversation: bool = False) -> List[str]:
    """Sample from a single dataset directory."""
    train_path = os.path.join(data_dir, 'train.bin')
    val_path = os.path.join(data_dir, 'val.bin')

    # Load tiktoken for decoding existing tokens
    enc = tiktoken.get_encoding("gpt2")

    # Load binary data
    train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_path, dtype=np.uint16, mode='r')

    dataset_name = os.path.basename(data_dir)
    logger.info(f"Found {len(train_data):,} training tokens and {len(val_data):,} validation tokens in {dataset_name}")

    # Sample from both train and val
    all_data = [train_data, val_data]
    data_weights = [0.95, 0.05]  # Mostly train, some val

    samples = []
    block_size = 1024 if is_conversation else 512  # Longer blocks for conversations

    with tqdm(total=max_samples, desc=f"Sampling from {dataset_name}") as pbar:
        while len(samples) < max_samples:
            # Choose dataset
            dataset = random.choices(all_data, weights=data_weights)[0]

            # Random starting position
            start_idx = random.randint(0, len(dataset) - block_size - 1)

            # Extract tokens and decode
            tokens = dataset[start_idx:start_idx + block_size].astype(np.int64)

            try:
                # Decode using tiktoken
                text = enc.decode(tokens.tolist())

                # Clean and validate text
                min_length = 30 if is_conversation else 50
                if len(text.strip()) > min_length:
                    if is_conversation:
                        # Format conversation text with special tokens
                        formatted_text = format_conversation_text(text)
                        if formatted_text:
                            samples.append(formatted_text)
                            pbar.update(1)
                    else:
                        samples.append(text.strip())
                        pbar.update(1)

            except Exception as e:
                # Skip invalid token sequences
                continue

    logger.info(f"Sampled {len(samples):,} text samples from {dataset_name}")
    return samples


def sample_conversation_data(data_dir: str, max_samples: int = 1000000) -> List[str]:
    """
    DEPRECATED: Conversation data is now sampled in sample_pretrain_data().
    This function is kept for compatibility but will return empty list.
    """
    logger.info("Conversation data is now included in pretraining data sampling")
    return []


def format_conversation_text(text: str) -> str:
    """
    Format conversation text with special tokens.

    Args:
        text: Raw conversation text

    Returns:
        Formatted text with <HUMAN> and <ASSISTANT> tokens
    """
    # Simple formatting - replace common patterns
    text = text.replace("Human:", "<HUMAN>")
    text = text.replace("human:", "<HUMAN>")
    text = text.replace("Assistant:", "<ASSISTANT>")
    text = text.replace("assistant:", "<ASSISTANT>")

    # Add EOS tokens after turns
    text = text.replace("<HUMAN>", "<HUMAN>")
    text = text.replace("<ASSISTANT>", "<EOS><ASSISTANT>")

    # Clean up extra spaces and newlines
    import re
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('<EOS> ', '<EOS>')

    # Ensure it ends with EOS
    if not text.endswith('<EOS>'):
        text += '<EOS>'

    return text.strip()


def prepare_training_corpus(pretrain_samples: List[str], conversation_samples: List[str], output_file: str):
    """
    Combine all samples into a training corpus for SentencePiece.
    Memory-efficient version that writes in batches.

    Args:
        pretrain_samples: Pretraining text samples
        conversation_samples: Conversation samples
        output_file: Path to save the combined corpus
    """
    logger.info("Preparing training corpus...")

    total_samples = len(pretrain_samples) + len(conversation_samples)
    logger.info(f"Combining {len(pretrain_samples):,} pretraining + {len(conversation_samples):,} conversation samples")

    # Write in batches to avoid memory issues
    batch_size = 50000  # Process 50k samples at a time

    with open(output_file, 'w', encoding='utf-8') as f:
        # Process pretrain samples in batches
        logger.info("Writing pretraining samples...")
        for i in tqdm(range(0, len(pretrain_samples), batch_size), desc="Writing pretrain batches"):
            batch = pretrain_samples[i:i + batch_size]
            for sample in batch:
                # Ensure each sample is on its own line
                f.write(sample.replace('\n', ' ').strip() + '\n')
            # Clear batch from memory
            del batch

        # Process conversation samples in batches
        if conversation_samples:
            logger.info("Writing conversation samples...")
            for i in tqdm(range(0, len(conversation_samples), batch_size), desc="Writing conversation batches"):
                batch = conversation_samples[i:i + batch_size]
                for sample in batch:
                    f.write(sample.replace('\n', ' ').strip() + '\n')
                del batch

    logger.info(f"Training corpus saved to {output_file} with {total_samples:,} samples")

    # Clear the large lists from memory
    pretrain_samples.clear()
    conversation_samples.clear()

    # Force garbage collection
    import gc
    gc.collect()


def train_sentencepiece_model(corpus_file: str, model_prefix: str, vocab_size: int = 32000) -> str:
    """
    Train SentencePiece model with BPE algorithm.

    Args:
        corpus_file: Path to training corpus file
        model_prefix: Prefix for output model files
        vocab_size: Target vocabulary size

    Returns:
        Path to trained .model file
    """
    logger.info(f"Training SentencePiece model with vocab_size={vocab_size}")

    # Define special tokens
    special_tokens = [
        '<UNK>', '<BOS>', '<EOS>', '<PAD>', '<SEP>',
        '<HUMAN>', '<ASSISTANT>', '<s>', '<THINK>',
        '<CODE>', '<ENDCODE>', '<MATH>', '<ENDMATH>'
    ]

    # SentencePiece training arguments
    train_args = [
        f'--input={corpus_file}',
        f'--model_prefix={model_prefix}',
        f'--vocab_size={vocab_size}',
        '--model_type=bpe',
        '--character_coverage=0.9995',
        '--num_threads=16',
        '--split_digits=false',
        '--allow_whitespace_only_pieces=true',
        '--byte_fallback=true',
        '--normalization_rule_name=identity',
        f'--user_defined_symbols={",".join(special_tokens)}',
        '--train_extremely_large_corpus=true',
        '--input_sentence_size=10000000',
        '--shuffle_input_sentence=true',
    ]

    # Train the model
    logger.info("Starting SentencePiece training...")
    start_time = time.time()

    spm.SentencePieceTrainer.train(' '.join(train_args))

    training_time = time.time() - start_time
    logger.info(f"SentencePiece training completed in {training_time:.2f} seconds")

    model_file = f"{model_prefix}.model"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Training failed - model file not created: {model_file}")

    logger.info(f"Model saved to {model_file}")
    return model_file


def create_cosmicizer(model_file: str) -> Cosmicizer:
    """
    Create a Cosmicizer instance from trained SentencePiece model.

    Args:
        model_file: Path to .model file

    Returns:
        Cosmicizer instance
    """
    logger.info(f"Creating Cosmicizer from {model_file}")

    # Load SentencePiece model
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(model_file)

    # Create Cosmicizer wrapper
    cosmicizer = Cosmicizer(sp_model=sp_model)

    logger.info(f"Cosmicizer created with vocab_size={cosmicizer.vocab_size}")
    return cosmicizer


def validate_tokenizer(cosmicizer: Cosmicizer) -> bool:
    """
    Run basic validation tests on the tokenizer.

    Args:
        cosmicizer: Cosmicizer instance to test

    Returns:
        True if validation passes
    """
    logger.info("Running tokenizer validation...")

    # Test 1: Basic encoding/decoding
    test_text = "Hello world! This is a test."
    encoded = cosmicizer.encode(test_text)
    decoded = cosmicizer.decode(encoded)

    if decoded.strip() != test_text.strip():
        logger.error(f"Round-trip test failed: '{test_text}' != '{decoded}'")
        return False

    # Test 2: Special tokens
    special_text = "<HUMAN>Hello<EOS><ASSISTANT>Hi there!<EOS>"
    encoded_special = cosmicizer.encode(special_text)
    decoded_special = cosmicizer.decode(encoded_special)

    if '<HUMAN>' not in decoded_special or '<ASSISTANT>' not in decoded_special:
        logger.error("Special token test failed")
        return False

    # Test 3: Conversation encoding
    human_msg = "What is AI?"
    assistant_msg = "AI stands for Artificial Intelligence."
    conv_encoded = cosmicizer.encode_conversation(human_msg, assistant_msg)
    conv_decoded = cosmicizer.decode(conv_encoded)

    if '<HUMAN>' not in conv_decoded or '<ASSISTANT>' not in conv_decoded:
        logger.error("Conversation encoding test failed")
        return False

    # Test 4: Vocabulary size
    if cosmicizer.vocab_size < 30000 or cosmicizer.vocab_size > 35000:
        logger.warning(f"Vocabulary size {cosmicizer.vocab_size} outside expected range [30000-35000]")

    # Test 5: Special token IDs
    for token, expected_id in cosmicizer.special_tokens.items():
        actual_id = cosmicizer.get_special_token_id(token)
        if actual_id != expected_id:
            logger.error(f"Special token ID mismatch for {token}: expected {expected_id}, got {actual_id}")
            return False

    logger.info("All validation tests passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Train Cosmicizer tokenizer")
    parser.add_argument("--pretrain_data", type=str, default="data/mixed",
                        help="Directory containing pretraining data (default: data/mixed)")
    parser.add_argument("--conversation_data", type=str, default="data/conversational",
                        help="Directory containing conversation data (default: data/conversational)")
    parser.add_argument("--output", type=str, default="cosmicizer.pkl",
                        help="Output pickle file (default: cosmicizer.pkl)")
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="Target vocabulary size (default: 32000)")
    parser.add_argument("--total_samples", type=int, default=2000000,
                        help="Total number of samples (default: 2M - reduced for memory efficiency)")
    parser.add_argument("--pretrain_samples", type=int, default=5000000,
                        help="Number of pretraining samples (DEPRECATED - use total_samples)")
    parser.add_argument("--conversation_samples", type=int, default=1000000,
                        help="Number of conversation samples (DEPRECATED - included in total)")
    parser.add_argument("--conversation_ratio", type=float, default=0.20,
                        help="Ratio of conversation data (default: 0.20 = 20%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--temp_dir", type=str, default=None,
                        help="Temporary directory for training files")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 80)
    logger.info("COSMICIZER TOKENIZER TRAINING")
    logger.info("=" * 80)
    logger.info(f"Target vocabulary size: {args.vocab_size:,}")
    logger.info(f"Total samples: {args.total_samples:,}")
    logger.info(f"Dataset distribution:")
    logger.info(f"  - OpenWebText: 35%")
    logger.info(f"  - Wikipedia: 45%")
    logger.info(f"  - Conversational: 20%")
    logger.info(f"Output file: {args.output}")

    try:
        # Create temporary directory for training files
        temp_dir = args.temp_dir or tempfile.mkdtemp(prefix="cosmicizer_")
        logger.info(f"Using temporary directory: {temp_dir}")

        # Step 1: Sample all data (pretraining + conversational combined)
        logger.info("\n" + "=" * 40)
        logger.info("STEP 1: SAMPLING ALL TRAINING DATA")
        logger.info("=" * 40)
        all_samples = sample_pretrain_data(args.pretrain_data, args.total_samples)

        # Step 2: Skip separate conversation sampling (already included)
        logger.info("\n" + "=" * 40)
        logger.info("STEP 2: CONVERSATION DATA (INCLUDED ABOVE)")
        logger.info("=" * 40)
        conversation_samples = []  # Empty since already included

        # Step 3: Prepare training corpus
        logger.info("\n" + "=" * 40)
        logger.info("STEP 3: PREPARING TRAINING CORPUS")
        logger.info("=" * 40)
        corpus_file = os.path.join(temp_dir, "training_corpus.txt")
        prepare_training_corpus(all_samples, conversation_samples, corpus_file)

        # Step 4: Train SentencePiece model
        logger.info("\n" + "=" * 40)
        logger.info("STEP 4: TRAINING SENTENCEPIECE MODEL")
        logger.info("=" * 40)
        model_prefix = os.path.join(temp_dir, "cosmicizer")
        model_file = train_sentencepiece_model(corpus_file, model_prefix, args.vocab_size)

        # Step 5: Create Cosmicizer wrapper
        logger.info("\n" + "=" * 40)
        logger.info("STEP 5: CREATING COSMICIZER")
        logger.info("=" * 40)
        cosmicizer = create_cosmicizer(model_file)

        # Step 6: Validate tokenizer
        logger.info("\n" + "=" * 40)
        logger.info("STEP 6: VALIDATION")
        logger.info("=" * 40)
        if not validate_tokenizer(cosmicizer):
            raise RuntimeError("Tokenizer validation failed")

        # Step 7: Save final tokenizer
        logger.info("\n" + "=" * 40)
        logger.info("STEP 7: SAVING COSMICIZER")
        logger.info("=" * 40)
        save_cosmicizer(cosmicizer, args.output)

        # Success!
        logger.info("\n" + "=" * 80)
        logger.info("COSMICIZER TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Tokenizer saved to: {args.output}")
        logger.info(f"Vocabulary size: {cosmicizer.vocab_size:,}")
        logger.info(f"Special tokens: {len(cosmicizer.special_tokens)}")

        # Print sample tokenizations
        logger.info("\nSample tokenizations:")
        test_samples = [
            "Hello, how are you today?",
            "<HUMAN>What is 2+2?<EOS><ASSISTANT>2+2 equals 4.<EOS>",
            "The quick brown fox jumps over the lazy dog.",
            "<CODE>def hello(): print('world')<ENDCODE>"
        ]

        for sample in test_samples:
            encoded = cosmicizer.encode(sample)
            logger.info(f"  '{sample}' -> {len(encoded)} tokens")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

    finally:
        # Cleanup temporary files
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception:
                logger.warning(f"Could not clean up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()
