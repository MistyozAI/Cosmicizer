"""
Cosmicizer: Custom tokenizer for CosmicFish using SentencePiece with BPE.
Provides tiktoken-compatible interface for seamless integration.
"""

import pickle
import os
import sentencepiece as spm
from typing import List, Union, Optional
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Cosmicizer:
    """
    Custom tokenizer wrapper around SentencePiece for CosmicFish.
    Provides tiktoken-compatible interface with conversation-aware special tokens.
    """
    
    def __init__(self, sentencepiece_model_path: Optional[str] = None, sp_model: Optional[spm.SentencePieceProcessor] = None):
        """Initialize Cosmicizer with either a model path or SentencePiece model."""
        
        # Special tokens definition - unique and clean format
        self.special_tokens = {
            '<UNK>': 0,        # Unknown token
            '<BOS>': 1,        # Begin of sequence
            '<EOS>': 2,        # End of sequence  
            '<PAD>': 3,        # Padding token
            '<SEP>': 4,        # Document/text separator
            '<HUMAN>': 5,      # Human speaker
            '<ASSISTANT>': 6,  # Assistant speaker
            '<SYSTEM>': 7,     # System messages
            '<THINK>': 8,      # Reasoning/thinking
            '<CODE>': 9,       # Start code block
            '<ENDCODE>': 10,   # End code block
            '<MATH>': 11,      # Math expressions
            '<ENDMATH>': 12,   # End math
        }
        
        # Reverse mapping for decoding
        self.id_to_special = {v: k for k, v in self.special_tokens.items()}
        
        # Initialize SentencePiece model
        if sp_model is not None:
            self.sp_model = sp_model
        elif sentencepiece_model_path is not None:
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(sentencepiece_model_path)
        else:
            raise ValueError("Either sentencepiece_model_path or sp_model must be provided")
        
        # Cache important properties
        self._vocab_size = self.sp_model.vocab_size()
        self._unk_token_id = self.special_tokens['<UNK>']
        self._bos_token_id = self.special_tokens['<BOS>']
        self._eos_token_id = self.special_tokens['<EOS>']
        self._pad_token_id = self.special_tokens['<PAD>']
        
        logger.info(f"Cosmicizer initialized with vocab_size={self._vocab_size}")
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size (tiktoken-compatible)."""
        return self._vocab_size
    
    @property
    def n_vocab(self) -> int:
        """Return vocabulary size (alternative name)."""
        return self._vocab_size
    
    @property
    def unk_token(self) -> int:
        """Return UNK token ID."""
        return self._unk_token_id
    
    @property
    def bos_token(self) -> int:
        """Return BOS token ID."""
        return self._bos_token_id
    
    @property
    def eos_token(self) -> int:
        """Return EOS token ID."""
        return self._eos_token_id
    
    @property
    def eot_token(self) -> int:
        """Return end-of-text token ID (alias for EOS, tiktoken-compatible)."""
        return self._eos_token_id
    
    @property
    def pad_token(self) -> int:
        """Return PAD token ID."""
        return self._pad_token_id
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs (tiktoken-compatible).
        
        Args:
            text: Input text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")
        
        # Use SentencePiece to encode
        token_ids = self.sp_model.encode(text, out_type=int)
        
        # Add special tokens if requested
        if add_special_tokens:
            token_ids = [self._bos_token_id] + token_ids + [self._eos_token_id]
        
        return token_ids
    
    def encode_ordinary(self, text: str) -> List[int]:
        """Encode text without special tokens (tiktoken-compatible)."""
        return self.encode(text, add_special_tokens=False)
    
    def decode(self, token_ids: Union[List[int], int]) -> str:
        """
        Decode token IDs to text (tiktoken-compatible).
        
        Args:
            token_ids: Token ID or list of token IDs
            
        Returns:
            Decoded text string
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        
        if not isinstance(token_ids, list):
            raise TypeError("token_ids must be an int or list of ints")
        
        # Use SentencePiece to decode
        text = self.sp_model.decode(token_ids)
        return text
    
    def encode_conversation(self, human_text: str, assistant_text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a conversation turn with proper formatting.
        
        Args:
            human_text: Human message
            assistant_text: Assistant message  
            add_special_tokens: Whether to add conversation tokens
            
        Returns:
            List of token IDs for the formatted conversation
        """
        if add_special_tokens:
            # Format: <HUMAN>human_text<EOS><ASSISTANT>assistant_text<EOS>
            formatted_text = f"<HUMAN>{human_text}<EOS><ASSISTANT>{assistant_text}<EOS>"
        else:
            formatted_text = f"{human_text} {assistant_text}"
        
        return self.encode(formatted_text)
    
    def get_special_token_id(self, token: str) -> int:
        """Get the ID for a specific special token."""
        if token not in self.special_tokens:
            raise ValueError(f"Unknown special token: {token}")
        return self.special_tokens[token]
    
    def get_special_token_text(self, token_id: int) -> Optional[str]:
        """Get the text for a special token ID."""
        return self.id_to_special.get(token_id)
    
    def is_special_token(self, token_id: int) -> bool:
        """Check if a token ID is a special token."""
        return token_id in self.id_to_special
    
    def tokenize_with_special(self, text: str) -> List[int]:
        """
        Advanced tokenization that properly handles embedded special tokens.
        
        Args:
            text: Input text that may contain special token strings
            
        Returns:
            List of token IDs with special tokens properly converted
        """
        # Replace special token strings with their IDs during encoding
        # This is handled automatically by SentencePiece if tokens were in training vocab
        return self.encode(text)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self._vocab_size
    
    def __getstate__(self):
        """Prepare object for pickling."""
        state = self.__dict__.copy()
        # SentencePiece model needs special handling for pickling
        if hasattr(self.sp_model, 'serialized_model_proto'):
            # Store the serialized model
            state['_sp_model_data'] = self.sp_model.serialized_model_proto()
            state['sp_model'] = None
        return state
    
    def __setstate__(self, state):
        """Restore object from pickle."""
        self.__dict__.update(state)
        
        # Restore SentencePiece model
        if '_sp_model_data' in state and state['_sp_model_data'] is not None:
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load_from_serialized_proto(state['_sp_model_data'])
        elif self.sp_model is None:
            raise ValueError("Could not restore SentencePiece model from pickle")


def load_cosmicizer(pkl_path: str) -> Cosmicizer:
    """
    Load a Cosmicizer from a pickle file.
    
    Args:
        pkl_path: Path to the cosmicizer.pkl file
        
    Returns:
        Loaded Cosmicizer instance
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Cosmicizer file not found: {pkl_path}")
    
    try:
        with open(pkl_path, 'rb') as f:
            cosmicizer = pickle.load(f)
        
        if not isinstance(cosmicizer, Cosmicizer):
            raise ValueError("Loaded object is not a Cosmicizer instance")
        
        logger.info(f"Cosmicizer loaded successfully from {pkl_path}")
        logger.info(f"Vocabulary size: {cosmicizer.vocab_size}")
        
        return cosmicizer
    
    except Exception as e:
        raise RuntimeError(f"Failed to load Cosmicizer from {pkl_path}: {str(e)}")


def save_cosmicizer(cosmicizer: Cosmicizer, pkl_path: str):
    """
    Save a Cosmicizer to a pickle file.
    
    Args:
        cosmicizer: Cosmicizer instance to save
        pkl_path: Path where to save the pickle file
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(pkl_path) if os.path.dirname(pkl_path) else '.', exist_ok=True)
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(cosmicizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Cosmicizer saved successfully to {pkl_path}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save Cosmicizer to {pkl_path}: {str(e)}")


# Convenience functions for easy import
def get_encoding(name: str = "cosmicizer") -> Cosmicizer:
    """
    Get encoding function (tiktoken-compatible).
    For now, just loads from default location.
    """
    if name == "cosmicizer":
        return load_cosmicizer("cosmicizer.pkl")
    else:
        raise ValueError(f"Unknown encoding: {name}")


# Example usage and testing
if __name__ == "__main__":
    # Simple test if SentencePiece model exists
    print("Cosmicizer class definition loaded successfully!")
    print("Special tokens defined:")
    
    # Create a dummy instance to test methods
    # This won't work without a trained model, just for interface testing
    special_tokens = {
        '<UNK>': 0, '<BOS>': 1, '<EOS>': 2, '<PAD>': 3, '<SEP>': 4,
        '<HUMAN>': 5, '<ASSISTANT>': 6, '<SYSTEM>': 7, '<THINK>': 8,
        '<CODE>': 9, '<ENDCODE>': 10, '<MATH>': 11, '<ENDMATH>': 12,
    }
    
    for token, token_id in special_tokens.items():
        print(f"  {token}: {token_id}")
    
    print("\nCosmicizer ready for training!")