"""
Comprehensive testing and validation script for Cosmicizer tokenizer.
Tests functionality, performance, and provides detailed analysis.
"""

import os
import sys
import argparse
import time
import statistics
import random
from typing import List, Dict, Tuple
import logging

from cosmicizer import load_cosmicizer, Cosmicizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class CosmicizerTester:
    """Comprehensive tester for Cosmicizer tokenizer."""
    
    def __init__(self, cosmicizer: Cosmicizer):
        self.cosmicizer = cosmicizer
        self.test_results = {}
        
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all test suites and return results."""
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE COSMICIZER TESTING")
        logger.info("=" * 80)
        
        test_suites = [
            ("Basic Functionality", self.test_basic_functionality),
            ("Round-trip Encoding", self.test_roundtrip_encoding),
            ("Special Tokens", self.test_special_tokens),
            ("Conversation Formatting", self.test_conversation_formatting),
            ("Edge Cases", self.test_edge_cases),
            ("Performance", self.test_performance),
            ("Compression Analysis", self.test_compression),
            ("Vocabulary Analysis", self.test_vocabulary),
        ]
        
        results = {}
        for suite_name, test_func in test_suites:
            logger.info(f"\n{'-' * 40}")
            logger.info(f"TESTING: {suite_name}")
            logger.info(f"{'-' * 40}")
            
            try:
                result = test_func()
                results[suite_name] = result
                status = "PASS" if result else "FAIL"
                logger.info(f"Result: {status}")
            except Exception as e:
                logger.error(f"Test suite failed with exception: {str(e)}")
                results[suite_name] = False
        
        return results
    
    def test_basic_functionality(self) -> bool:
        """Test basic encode/decode functionality."""
        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "This is a test of the tokenizer functionality.",
            "Testing 123 with numbers and symbols @#$%",
            "",  # Empty string
            " ",  # Single space
            "A",  # Single character
        ]
        
        for text in test_texts:
            try:
                # Test encoding
                tokens = self.cosmicizer.encode(text)
                
                # Validate token list
                if not isinstance(tokens, list):
                    logger.error(f"encode() returned {type(tokens)}, expected list")
                    return False
                
                if not all(isinstance(token, int) for token in tokens):
                    logger.error("encode() returned non-integer tokens")
                    return False
                
                # Test decoding
                decoded = self.cosmicizer.decode(tokens)
                
                if not isinstance(decoded, str):
                    logger.error(f"decode() returned {type(decoded)}, expected str")
                    return False
                
                # Test properties
                vocab_size = self.cosmicizer.vocab_size
                if not isinstance(vocab_size, int) or vocab_size <= 0:
                    logger.error(f"Invalid vocab_size: {vocab_size}")
                    return False
                
                logger.debug(f"✓ '{text}' -> {len(tokens)} tokens -> '{decoded}'")
                
            except Exception as e:
                logger.error(f"Basic functionality test failed for '{text}': {str(e)}")
                return False
        
        logger.info(f"✓ Basic functionality tests passed for {len(test_texts)} samples")
        return True
    
    def test_roundtrip_encoding(self) -> bool:
        """Test that encode -> decode returns original text."""
        test_texts = [
            "Simple text without special characters.",
            "Text with punctuation: Hello, world! How are you? Fine, thanks.",
            "Numbers and symbols: 123 + 456 = 579. Cost: $12.34 (tax: 8.5%)",
            "Unicode test: café, naïve, résumé, 你好, مرحبا, привет",
            "Mixed case: CamelCase, snake_case, UPPERCASE, lowercase",
            "Whitespace handling:   multiple   spaces   and\ttabs\nand newlines",
        ]
        
        failures = 0
        for text in test_texts:
            try:
                tokens = self.cosmicizer.encode(text)
                decoded = self.cosmicizer.decode(tokens)
                
                # For most cases, we expect exact match
                # Some tokenizers may normalize whitespace
                if decoded != text:
                    # Check if it's just whitespace normalization
                    if decoded.strip() == text.strip() and len(decoded.split()) == len(text.split()):
                        logger.warning(f"Minor whitespace difference: '{text}' != '{decoded}'")
                    else:
                        logger.error(f"Round-trip failed: '{text}' != '{decoded}'")
                        failures += 1
                else:
                    logger.debug(f"✓ Round-trip passed: '{text}'")
                    
            except Exception as e:
                logger.error(f"Round-trip test failed for '{text}': {str(e)}")
                failures += 1
        
        success_rate = (len(test_texts) - failures) / len(test_texts)
        logger.info(f"✓ Round-trip success rate: {success_rate:.1%} ({len(test_texts) - failures}/{len(test_texts)})")
        
        return failures == 0
    
    def test_special_tokens(self) -> bool:
        """Test special token handling."""
        
        # Test 1: Special token IDs
        expected_special_tokens = {
            '<UNK>': 0, '<BOS>': 1, '<EOS>': 2, '<PAD>': 3, '<SEP>': 4,
            '<HUMAN>': 5, '<ASSISTANT>': 6, '<s>': 7, '<THINK>': 8,
            '<CODE>': 9, '<ENDCODE>': 10, '<MATH>': 11, '<ENDMATH>': 12,
        }
        
        for token, expected_id in expected_special_tokens.items():
            try:
                actual_id = self.cosmicizer.get_special_token_id(token)
                if actual_id != expected_id:
                    logger.error(f"Special token ID mismatch: {token} expected {expected_id}, got {actual_id}")
                    return False
                
                # Test reverse lookup
                token_text = self.cosmicizer.get_special_token_text(actual_id)
                if token_text != token:
                    logger.error(f"Reverse lookup failed: ID {actual_id} expected '{token}', got '{token_text}'")
                    return False
                    
            except Exception as e:
                logger.error(f"Special token test failed for {token}: {str(e)}")
                return False
        
        # Test 2: Special token properties
        properties_to_test = [
            ('eos_token', 2),
            ('bos_token', 1), 
            ('pad_token', 3),
            ('unk_token', 0),
        ]
        
        for prop_name, expected_value in properties_to_test:
            try:
                actual_value = getattr(self.cosmicizer, prop_name)
                if actual_value != expected_value:
                    logger.error(f"Property {prop_name} expected {expected_value}, got {actual_value}")
                    return False
            except Exception as e:
                logger.error(f"Property test failed for {prop_name}: {str(e)}")
                return False
        
        # Test 3: Special token encoding/decoding
        test_cases = [
            "<BOS>Hello<EOS>",
            "<HUMAN>How are you?<EOS>",
            "<ASSISTANT>I'm fine, thanks!<EOS>",
            "<CODE>print('hello')<ENDCODE>",
            "<MATH>2 + 2 = 4<ENDMATH>",
        ]
        
        for test_case in test_cases:
            try:
                tokens = self.cosmicizer.encode(test_case)
                decoded = self.cosmicizer.decode(tokens)
                
                # Check that special tokens are preserved
                for special_token in expected_special_tokens.keys():
                    if special_token in test_case:
                        if special_token not in decoded:
                            logger.error(f"Special token {special_token} not preserved in: {test_case}")
                            return False
                            
            except Exception as e:
                logger.error(f"Special token encoding test failed for '{test_case}': {str(e)}")
                return False
        
        logger.info(f"✓ Special token tests passed for {len(expected_special_tokens)} tokens")
        return True
    
    def test_conversation_formatting(self) -> bool:
        """Test conversation-specific functionality."""
        
        test_conversations = [
            ("Hello!", "Hi there!"),
            ("What is 2+2?", "2+2 equals 4."),
            ("Can you help me with Python?", "Of course! What do you need help with?"),
            ("", "Hello!"),  # Empty human message
            ("Thanks!", ""),  # Empty assistant message
        ]
        
        for human_msg, assistant_msg in test_conversations:
            try:
                # Test conversation encoding
                conv_tokens = self.cosmicizer.encode_conversation(human_msg, assistant_msg)
                
                if not isinstance(conv_tokens, list) or not conv_tokens:
                    logger.error(f"Conversation encoding failed for: '{human_msg}' / '{assistant_msg}'")
                    return False
                
                # Decode and check format
                decoded_conv = self.cosmicizer.decode(conv_tokens)
                
                # Should contain conversation markers
                if '<HUMAN>' not in decoded_conv and human_msg:
                    logger.error(f"Missing <HUMAN> marker in: {decoded_conv}")
                    return False
                
                if '<ASSISTANT>' not in decoded_conv and assistant_msg:
                    logger.error(f"Missing <ASSISTANT> marker in: {decoded_conv}")
                    return False
                
                logger.debug(f"✓ Conversation: '{human_msg}' / '{assistant_msg}' -> {len(conv_tokens)} tokens")
                
            except Exception as e:
                logger.error(f"Conversation test failed for '{human_msg}' / '{assistant_msg}': {str(e)}")
                return False
        
        logger.info(f"✓ Conversation formatting tests passed for {len(test_conversations)} examples")
        return True
    
    def test_edge_cases(self) -> bool:
        """Test edge cases and error handling."""
        
        # Test 1: Empty and None inputs
        edge_cases = [
            "",           # Empty string
            " ",          # Space only
            "\n",         # Newline only
            "\t",         # Tab only
            "   \n\t  ",  # Mixed whitespace
        ]
        
        for case in edge_cases:
            try:
                tokens = self.cosmicizer.encode(case)
                decoded = self.cosmicizer.decode(tokens)
                logger.debug(f"✓ Edge case '{repr(case)}' -> {len(tokens)} tokens")
                
            except Exception as e:
                logger.error(f"Edge case failed for {repr(case)}: {str(e)}")
                return False
        
        # Test 2: Invalid inputs
        invalid_inputs = [
            (None, "encode with None"),
            (123, "encode with integer"),
            ([], "encode with list"),
        ]
        
        for invalid_input, description in invalid_inputs:
            try:
                tokens = self.cosmicizer.encode(invalid_input)
                logger.error(f"Should have failed: {description}")
                return False
            except (TypeError, ValueError, AttributeError):
                logger.debug(f"✓ Correctly rejected: {description}")
            except Exception as e:
                logger.error(f"Unexpected exception for {description}: {str(e)}")
                return False
        
        # Test 3: Large inputs
        large_text = "This is a test sentence. " * 1000  # ~25k characters
        try:
            tokens = self.cosmicizer.encode(large_text)
            decoded = self.cosmicizer.decode(tokens)
            logger.debug(f"✓ Large text: {len(large_text)} chars -> {len(tokens)} tokens")
        except Exception as e:
            logger.error(f"Large text test failed: {str(e)}")
            return False
        
        # Test 4: Unicode edge cases
        unicode_cases = [
            "emoji test: 😀😃😄😁😆",
            "symbols: ©®™°±²³µ¶§",
            "accents: àáâãäåæçèéêë",
            "asian: 你好世界 こんにちは 안녕하세요",
            "arabic: مرحبا بالعالم",
            "russian: Привет мир",
        ]
        
        for unicode_case in unicode_cases:
            try:
                tokens = self.cosmicizer.encode(unicode_case)
                decoded = self.cosmicizer.decode(tokens)
                logger.debug(f"✓ Unicode: '{unicode_case}' -> {len(tokens)} tokens")
            except Exception as e:
                logger.error(f"Unicode test failed for '{unicode_case}': {str(e)}")
                return False
        
        logger.info(f"✓ Edge case tests passed for {len(edge_cases) + len(unicode_cases)} cases")
        return True
    
    def test_performance(self) -> bool:
        """Test encoding/decoding performance."""
        
        # Generate test texts of different sizes
        test_texts = [
            "Short text.",
            "Medium length text with several words and punctuation marks.",
            "Long text. " * 100,  # ~1k characters
            "Very long text. " * 1000,  # ~10k characters
        ]
        
        encoding_times = []
        decoding_times = []
        
        for text in test_texts:
            # Test encoding performance
            start_time = time.time()
            for _ in range(10):  # Multiple runs for averaging
                tokens = self.cosmicizer.encode(text)
            encoding_time = (time.time() - start_time) / 10
            encoding_times.append(encoding_time)
            
            # Test decoding performance
            start_time = time.time()
            for _ in range(10):  # Multiple runs for averaging
                decoded = self.cosmicizer.decode(tokens)
            decoding_time = (time.time() - start_time) / 10
            decoding_times.append(decoding_time)
            
            chars_per_sec_encode = len(text) / encoding_time if encoding_time > 0 else 0
            chars_per_sec_decode = len(text) / decoding_time if decoding_time > 0 else 0
            
            logger.debug(f"Text len {len(text):>5}: "
                        f"encode {encoding_time*1000:.2f}ms ({chars_per_sec_encode:.0f} chars/s), "
                        f"decode {decoding_time*1000:.2f}ms ({chars_per_sec_decode:.0f} chars/s)")
        
        avg_encode_time = statistics.mean(encoding_times)
        avg_decode_time = statistics.mean(decoding_times)
        
        logger.info(f"✓ Performance - Avg encode: {avg_encode_time*1000:.2f}ms, "
                   f"Avg decode: {avg_decode_time*1000:.2f}ms")
        
        # Performance should be reasonable (< 10ms for typical texts)
        if avg_encode_time > 0.01 or avg_decode_time > 0.01:
            logger.warning("Performance may be slower than expected")
        
        return True
    
    def test_compression(self) -> bool:
        """Analyze compression ratio and efficiency."""
        
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a high-level programming language.",
            "Machine learning models require large datasets for training.",
            "Natural language processing involves understanding human language.",
            "Artificial intelligence will transform many industries.",
        ]
        
        total_chars = 0
        total_tokens = 0
        
        logger.info("Compression analysis:")
        
        for text in test_texts:
            tokens = self.cosmicizer.encode(text)
            char_count = len(text)
            token_count = len(tokens)
            compression_ratio = char_count / token_count if token_count > 0 else 0
            
            total_chars += char_count
            total_tokens += token_count
            
            logger.info(f"  '{text[:50]}...' -> {char_count} chars, {token_count} tokens, "
                       f"ratio: {compression_ratio:.2f}")
        
        overall_ratio = total_chars / total_tokens if total_tokens > 0 else 0
        logger.info(f"✓ Overall compression ratio: {overall_ratio:.2f} chars/token")
        
        # Good tokenizers typically achieve 3-4 chars per token for English
        if overall_ratio < 2.0:
            logger.warning("Compression ratio seems low (under-segmentation)")
        elif overall_ratio > 6.0:
            logger.warning("Compression ratio seems high (over-segmentation)")
        
        return True
    
    def test_vocabulary(self) -> bool:
        """Analyze vocabulary composition and coverage."""
        
        vocab_size = self.cosmicizer.vocab_size
        special_token_count = len(self.cosmicizer.special_tokens)
        regular_token_count = vocab_size - special_token_count
        
        logger.info(f"Vocabulary analysis:")
        logger.info(f"  Total vocabulary size: {vocab_size:,}")
        logger.info(f"  Special tokens: {special_token_count}")
        logger.info(f"  Regular tokens: {regular_token_count:,}")
        logger.info(f"  Special token ratio: {special_token_count/vocab_size:.1%}")
        
        # Test token distribution with common words
        common_words = [
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
            "a", "an", "is", "are", "was", "were", "be", "been", "have", "has",
            "do", "does", "did", "will", "would", "could", "should", "can", "may"
        ]
        
        single_token_words = 0
        for word in common_words:
            tokens = self.cosmicizer.encode(word)
            if len(tokens) == 1:
                single_token_words += 1
            logger.debug(f"  '{word}' -> {len(tokens)} tokens")
        
        single_token_ratio = single_token_words / len(common_words)
        logger.info(f"  Common words as single tokens: {single_token_ratio:.1%} ({single_token_words}/{len(common_words)})")
        
        # Good tokenizers should encode most common words as single tokens
        if single_token_ratio < 0.7:
            logger.warning("Many common words are split into multiple tokens")
        
        return True
    
    def print_sample_tokenizations(self):
        """Print sample tokenizations for manual inspection."""
        logger.info("\n" + "=" * 40)
        logger.info("SAMPLE TOKENIZATIONS")
        logger.info("=" * 40)
        
        samples = [
            "Hello, how are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "<HUMAN>What is artificial intelligence?<EOS>",
            "<ASSISTANT>AI is the simulation of human intelligence in machines.<EOS>",
            "Programming in Python is fun and efficient.",
            "<CODE>def factorial(n): return 1 if n <= 1 else n * factorial(n-1)<ENDCODE>",
            "<MATH>E = mc²<ENDMATH>",
            "I can't believe it's not butter! Cost: $3.99 (was $4.99)",
            "Unicode test: café naïve résumé 你好 мир",
        ]
        
        for sample in samples:
            tokens = self.cosmicizer.encode(sample)
            decoded = self.cosmicizer.decode(tokens)
            
            logger.info(f"\nText: '{sample}'")
            logger.info(f"Tokens ({len(tokens)}): {tokens}")
            logger.info(f"Decoded: '{decoded}'")
            logger.info(f"Round-trip: {'✓' if decoded == sample else '✗'}")


def main():
    parser = argparse.ArgumentParser(description="Test Cosmicizer tokenizer")
    parser.add_argument("--tokenizer", type=str, default="cosmicizer.pkl",
                        help="Path to cosmicizer.pkl file (default: cosmicizer.pkl)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--samples", action="store_true", 
                        help="Show sample tokenizations")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load the tokenizer
    logger.info(f"Loading Cosmicizer from {args.tokenizer}")
    
    try:
        cosmicizer = load_cosmicizer(args.tokenizer)
    except FileNotFoundError:
        logger.error(f"Tokenizer file not found: {args.tokenizer}")
        logger.error("Make sure to run train_tokenizer.py first to create the tokenizer.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        sys.exit(1)
    
    # Create tester and run tests
    tester = CosmicizerTester(cosmicizer)
    results = tester.run_all_tests()
    
    # Show sample tokenizations if requested
    if args.samples:
        tester.print_sample_tokenizations()
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name:.<40} {status}")
    
    logger.info(f"\nOverall result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests PASSED! Cosmicizer is ready for use.")
        sys.exit(0)
    else:
        logger.error(f"❌ {total - passed} tests FAILED. Please check the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()