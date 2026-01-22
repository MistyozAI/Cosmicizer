# Cosmicizer Tokenizer

A custom SentencePiece-based tokenizer built by **Mistyoz AI**, designed for the **CosmicFish-pico** language model.

## Features

- **tiktoken-compatible** - Drop-in replacement for tiktoken with methods like `encode()`, `decode()`, `encode_ordinary()`
- **Conversation-aware special tokens** - Built-in tokens for `<HUMAN>`, `<ASSISTANT>`, `<SYSTEM>`, `<CODE>`, `<MATH>`, etc.
- **SentencePiece BPE backend** - Efficient subword tokenization
- **Optimized for CosmicFish** - Specifically designed for CosmicFish pico language model training
- **Serialization support** - Easy save/load with pickle

## Installation
```bash
pip install -r requirements_txt.txt
```

## Quick Start
```python
from cosmicizer import load_cosmicizer

# Load the tokenizer
tokenizer = load_cosmicizer("cosmicizer.pkl")

# Encode text
tokens = tokenizer.encode("Hello, world!")

# Decode tokens
text = tokenizer.decode(tokens)

# Encode conversations
conv_tokens = tokenizer.encode_conversation(
    human_text="What is AI?",
    assistant_text="AI stands for Artificial Intelligence..."
)
```

## Dataset Preparation

Prepare datasets for CosmicFish training:
```bash
# Mixed dataset (OpenWebText + Wikipedia)
python prepare.py --dataset_name mixed --wikipedia_weight 0.6

# Single-turn conversations (Alpaca, Dolly, OASST1, LIMA)
python convd.py --dataset tatsu-lab/alpaca --output_dir data/singleturn
```

## Special Tokens

- `<BOS>`, `<EOS>`, `<PAD>`, `<UNK>` - Standard tokens
- `<HUMAN>`, `<ASSISTANT>`, `<SYSTEM>` - Conversation roles
- `<CODE>`, `<ENDCODE>` - Code blocks
- `<MATH>`, `<ENDMATH>` - Mathematical expressions
- `<THINK>` - Reasoning/chain-of-thought

## License

 MIT License
 
---

Mistyoz AI, Hyderabad
