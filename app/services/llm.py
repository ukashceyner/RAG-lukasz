"""Google Gemini LLM service for Q&A generation."""

import logging
from typing import Optional

import google.generativeai as genai

from app.config import get_settings
from app.models.schemas import SourceChunk

logger = logging.getLogger(__name__)


class LLMService:
    """Handles Q&A generation using Google Gemini."""

    def __init__(self):
        self.settings = get_settings()
        self._model: Optional[genai.GenerativeModel] = None

    @property
    def model(self) -> genai.GenerativeModel:
        """Lazy initialization of Gemini model."""
        if self._model is None:
            if not self.settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY is not configured")
            genai.configure(api_key=self.settings.google_api_key)
            self._model = genai.GenerativeModel(self.settings.gemini_model)
        return self._model

    async def generate_answer(
        self,
        question: str,
        sources: list[SourceChunk]
    ) -> str:
        """
        Generate an answer based on the question and source documents.

        Args:
            question: User's question
            sources: List of relevant source chunks

        Returns:
            Generated answer with source citations
        """
        if not sources:
            return "I couldn't find any relevant information to answer your question."

        # Build context from sources
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(
                f"[Source {i}: {source.filename}, Chunk {source.chunk_index + 1}]\n"
                f"{source.content}\n"
            )
        context = "\n---\n".join(context_parts)

        # Create the prompt
        prompt = f"""You are a helpful assistant that answers questions based on provided source documents.
Use ONLY the information from the sources below to answer the question.
If the sources don't contain enough information to fully answer the question, say so.
Always cite your sources using [Source N] notation when referencing specific information.

SOURCES:
{context}

QUESTION: {question}

ANSWER:"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
                )
            )

            answer = response.text
            logger.info(f"Generated answer of {len(answer)} characters")
            return answer

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise

    def is_configured(self) -> bool:
        """Check if the LLM service is properly configured."""
        return bool(self.settings.google_api_key)


# Singleton instance
llm_service = LLMService()
