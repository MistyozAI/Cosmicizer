"""
Prepare single-turn conversational dataset for fine-tuning CosmicFish.
This script processes instruction-response datasets like Alpaca, Dolly, or LIMA.
"""

import os
import sys
import argparse
import json
import numpy as np
import tiktoken
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
import logging
import time
from dataclasses import dataclass
import pickle
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    dataset_name: str = "tatsu-lab/alpaca"  # Default to Alpaca dataset
    output_dir: str = "data/singleturn"
    test_size: float = 0.05  # 5% for validation
    seed: int = 42
    max_seq_length: int = 1024  # Maximum sequence length (must match model's block_size)
    human_prefix: str = "Human: "
    assistant_prefix: str = "Assistant: "
    end_of_turn: str = "\n\n"  # Delimiter between conversation turns
    instruction_prefix: str = "Below is a conversation between a helpful AI assistant and a human. The assistant is knowledgeable, friendly, and provides detailed, helpful responses.\n\n"
    encoding_name: str = "gpt2"  # GPT-2 tokenizer (same as used in original model)
    overwrite: bool = False
    language: str = "en"  # Filter for English data if dataset has language markers


def format_conversation(question, answer, config):
    """Format a question and answer into a standardized format."""
    formatted_text = config.instruction_prefix
    formatted_text += f"{config.human_prefix}{question}{config.end_of_turn}"
    formatted_text += f"{config.assistant_prefix}{answer}{config.end_of_turn}"
    return formatted_text


def process_alpaca_dataset(config):
    """Process the Stanford Alpaca dataset."""
    logger.info(f"Loading Alpaca dataset...")

    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # Format into single-turn conversations
    conversations = []

    for item in tqdm(dataset, desc="Processing Alpaca examples"):
        instruction = item["instruction"]
        response = item["output"]

        # If there's input, add it to the instruction
        if item["input"] and item["input"].strip():
            instruction += f"\n{item['input']}"

        conversations.append({
            "question": instruction,
            "answer": response
        })

    logger.info(f"Processed {len(conversations)} Alpaca conversations")
    return conversations


def process_dolly_dataset(config):
    """Process the Dolly dataset."""
    logger.info(f"Loading Dolly dataset...")

    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    # Format into single-turn conversations
    conversations = []

    for item in tqdm(dataset, desc="Processing Dolly examples"):
        instruction = item["instruction"]
        response = item["response"]

        # If there's context, add it to the instruction
        if "context" in item and item["context"] and item["context"].strip():
            instruction += f"\nContext: {item['context']}"

        conversations.append({
            "question": instruction,
            "answer": response
        })

    logger.info(f"Processed {len(conversations)} Dolly conversations")
    return conversations


def process_lima_dataset(config):
    """Process the LIMA dataset."""
    logger.info(f"Loading LIMA dataset...")

    try:
        dataset = load_dataset("GAIR/lima", split="train")

        # Format into single-turn conversations
        conversations = []

        for item in tqdm(dataset, desc="Processing LIMA examples"):
            # Extract just the first exchange
            if len(item["conversations"]) >= 2:  # Need at least one human and one assistant message
                human_msg = item["conversations"][0]["value"]
                assistant_msg = item["conversations"][1]["value"]

                conversations.append({
                    "question": human_msg,
                    "answer": assistant_msg
                })

        logger.info(f"Processed {len(conversations)} LIMA conversations")
        return conversations
    except Exception as e:
        logger.error(f"Error loading LIMA dataset: {e}")
        return []


def process_oasst1_single_turns(config):
    """Process Open Assistant dataset, but extract only single turns."""
    logger.info(f"Loading OpenAssistant/oasst1 dataset...")

    # Specifically load the English subset if possible
    try:
        dataset = load_dataset("OpenAssistant/oasst1", "en", split="train")
        logger.info(f"Successfully loaded English OASST1 dataset")
    except Exception:
        # Fall back to loading the whole dataset and filtering
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
        if config.language:
            dataset = dataset.filter(lambda example: example.get("lang") == config.language)
            logger.info(f"Filtered for {config.language} language: {len(dataset)} examples")

    # Group messages to find question-answer pairs
    conversations = []
    message_dict = {}
    parent_to_children = {}

    # First pass: build the message dictionary and parent-child relationships
    for item in tqdm(dataset, desc="Processing messages"):
        message_id = item["message_id"]
        parent_id = item["parent_id"]
        role = "human" if item["role"] == "prompter" else "assistant"

        # Store this message
        message_dict[message_id] = {
            "role": role,
            "content": item["text"],
            "parent_id": parent_id
        }

        # Add to parent-child mapping
        if parent_id not in parent_to_children:
            parent_to_children[parent_id] = []
        parent_to_children[parent_id].append(message_id)

    # Second pass: find human questions with exactly one assistant response
    for message_id, message in message_dict.items():
        # Only consider human messages
        if message["role"] != "human":
            continue

        # Check if this message has children
        if message_id in parent_to_children and len(parent_to_children[message_id]) == 1:
            child_id = parent_to_children[message_id][0]
            child = message_dict.get(child_id)

            # Make sure the child is an assistant message
            if child and child["role"] == "assistant":
                conversations.append({
                    "question": message["content"],
                    "answer": child["content"]
                })

    logger.info(f"Extracted {len(conversations)} single-turn conversations from OASST1")
    return conversations


def prepare_dataset(config):
    """Prepare the specified single-turn dataset."""
    os.makedirs(config.output_dir, exist_ok=True)

    # Check if processed data already exists
    train_path = os.path.join(config.output_dir, 'train.bin')
    val_path = os.path.join(config.output_dir, 'val.bin')

    if os.path.exists(train_path) and os.path.exists(val_path) and not config.overwrite:
        logger.info(f"Processed data already exists at {config.output_dir}. Use --overwrite to reprocess.")
        return

    # Load the tokenizer
    enc = tiktoken.get_encoding(config.encoding_name)

    # Process the dataset based on which one was specified
    if "alpaca" in config.dataset_name.lower():
        conversations = process_alpaca_dataset(config)
    elif "dolly" in config.dataset_name.lower():
        conversations = process_dolly_dataset(config)
    elif "lima" in config.dataset_name.lower():
        conversations = process_lima_dataset(config)
    elif "oasst" in config.dataset_name.lower():
        conversations = process_oasst1_single_turns(config)
    else:
        logger.error(f"Unknown dataset: {config.dataset_name}")
        sys.exit(1)

    if not conversations:
        logger.error(f"No conversations extracted from dataset. Exiting.")
        sys.exit(1)

    # Shuffle conversations
    random.seed(config.seed)
    random.shuffle(conversations)

    # Split into train and validation sets
    val_size = int(len(conversations) * config.test_size)
    train_conversations = conversations[val_size:]
    val_conversations = conversations[:val_size]

    logger.info(f"Train: {len(train_conversations)} conversations")
    logger.info(f"Validation: {len(val_conversations)} conversations")

    # Format and tokenize the conversations
    def process_conversations(conversation_list):
        all_tokens = []
        for conv in tqdm(conversation_list, desc="Formatting and tokenizing"):
            # Format the conversation
            formatted_text = format_conversation(conv["question"], conv["answer"], config)

            # Tokenize
            tokens = enc.encode(formatted_text)
            if len(tokens) > config.max_seq_length:
                tokens = tokens[:config.max_seq_length]

            all_tokens.extend(tokens)
            # Add an extra token to separate conversations
            all_tokens.append(enc.eot_token)

        return all_tokens

    logger.info("Processing training conversations...")
    train_tokens = process_conversations(train_conversations)

    logger.info("Processing validation conversations...")
    val_tokens = process_conversations(val_conversations)

    logger.info(f"Train tokens: {len(train_tokens)}")
    logger.info(f"Validation tokens: {len(val_tokens)}")

    # Save as binary files
    def save_to_binary(tokens, filename):
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(filename)
        logger.info(f"Saved {len(tokens)} tokens to {filename}")

    save_to_binary(train_tokens, train_path)
    save_to_binary(val_tokens, val_path)

    # Save metadata
    meta = {
        'vocab_size': enc.n_vocab,
        'total_tokens': {
            'train': len(train_tokens),
            'val': len(val_tokens)
        },
        'dataset_name': config.dataset_name,
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {k: v for k, v in vars(config).items()},
        'num_conversations': len(conversations)
    }

    with open(os.path.join(config.output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    # Also save a few examples as text for inspection
    with open(os.path.join(config.output_dir, 'examples.txt'), 'w', encoding='utf-8') as f:
        for i, conv in enumerate(val_conversations[:5]):
            f.write(f"Example {i + 1}:\n")
            f.write("-" * 50 + "\n")
            f.write(format_conversation(conv["question"], conv["answer"], config))
            f.write("\n\n" + "=" * 50 + "\n\n")

    logger.info("Dataset preparation completed!")


def main():
    parser = argparse.ArgumentParser(description="Prepare single-turn dataset for fine-tuning")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca",
                        help="Dataset to use (default: tatsu-lab/alpaca, other options: databricks/databricks-dolly-15k, OpenAssistant/oasst1, GAIR/lima)")
    parser.add_argument("--output_dir", type=str, default="data/singleturn",
                        help="Output directory for processed data (default: data/singleturn)")
    parser.add_argument("--test_size", type=float, default=0.05,
                        help="Validation split size (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length (default: 1024)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files")
    parser.add_argument("--language", type=str, default="en",
                        help="Filter for language (default: en for English)")
    parser.add_argument("--encoding", type=str, default="gpt2",
                        help="Tokenizer encoding (default: gpt2)")

    args = parser.parse_args()

    config = DatasetConfig(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        max_seq_length=args.max_length,
        overwrite=args.overwrite,
        language=args.language,
        encoding_name=args.encoding
    )

    prepare_dataset(config)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error during dataset preparation: {str(e)}", exc_info=True)
        sys.exit(1)