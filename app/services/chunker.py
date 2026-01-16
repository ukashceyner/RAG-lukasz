"""Text chunking service with token-based splitting and overlap."""

import logging
import re
from dataclasses import dataclass

import tiktoken

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text with its index."""
    content: str
    index: int
    token_count: int


class TextChunker:
    """Handles text chunking with configurable size and overlap."""

    def __init__(self):
        self.settings = get_settings()
        # Use cl100k_base encoding (GPT-4/Claude compatible)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_text(self, text: str) -> list[TextChunk]:
        """
        Split text into overlapping chunks based on token count.

        Args:
            text: The full document text to chunk

        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []

        # Clean and normalize text
        text = self._clean_text(text)

        # Get token count settings
        chunk_size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap

        # Tokenize the entire text
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)

        if total_tokens <= chunk_size:
            # Text fits in single chunk
            return [TextChunk(content=text, index=0, token_count=total_tokens)]

        chunks = []
        start = 0
        chunk_index = 0

        while start < total_tokens:
            # Calculate end position
            end = min(start + chunk_size, total_tokens)

            # Extract tokens for this chunk
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Try to end at a sentence boundary if not at the end
            if end < total_tokens:
                chunk_text = self._adjust_to_sentence_boundary(chunk_text)

            chunks.append(TextChunk(
                content=chunk_text.strip(),
                index=chunk_index,
                token_count=len(self.tokenizer.encode(chunk_text))
            ))

            # Move start position with overlap
            start = end - overlap if end < total_tokens else total_tokens
            chunk_index += 1

            logger.debug(f"Created chunk {chunk_index} with {len(chunk_tokens)} tokens")

        logger.info(f"Split text into {len(chunks)} chunks (total tokens: {total_tokens})")
        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        # Remove null bytes and other control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text.strip()

    def _adjust_to_sentence_boundary(self, text: str) -> str:
        """Try to end text at a sentence boundary."""
        # Look for sentence endings in the last 20% of the chunk
        cutoff = int(len(text) * 0.8)
        end_portion = text[cutoff:]

        # Find the last sentence ending
        sentence_endings = ['.', '!', '?', '.\n', '!\n', '?\n']
        last_ending = -1

        for ending in sentence_endings:
            pos = end_portion.rfind(ending)
            if pos > last_ending:
                last_ending = pos
                ending_len = len(ending)

        if last_ending != -1:
            # Found a sentence ending - include it
            return text[:cutoff + last_ending + ending_len]

        return text


# Singleton instance
chunker = TextChunker()
