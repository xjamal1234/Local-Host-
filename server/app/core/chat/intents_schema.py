"""
Intent schema definitions for chat intent extraction (OpenAI Structured Outputs).

This schema defines all allowed intents and their arguments for the chat system.
Intent extraction uses OpenAI Responses API with Structured Outputs to guarantee
exact JSON Schema conformance.
"""

# JSON Schema for intent extraction (OpenAI Structured Outputs format)
INTENT_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "count_paragraphs",
                "has_bullets",
                "has_headings",
                "get_main_title",
                "get_page_number",
                "paragraph_word_count",
                "read_paragraph",
                "summarize_paragraph",
                "read_line",
                "where_is_term",
                "list_subtitles",
                "list_bullets_in_paragraph",
                "unsupported"
            ],
            "description": "The extracted intent from user query"
        },
        "args": {
            "type": "object",
            "description": "Arguments for the intent",
            "properties": {
                # For paragraph_word_count
                "paragraph_index": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "1-based paragraph index"
                },
                # For read_line
                "line_index": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "1-based line index within a paragraph"
                },
                # For summarize_paragraph
                "style": {
                    "type": "string",
                    "enum": ["short", "detailed"],
                    "description": "Summarization style (default: short)"
                },
                # For where_is_term
                "term": {
                    "type": "string",
                    "description": "Search term to locate in document"
                },
                # For unsupported
                "original_query": {
                    "type": "string",
                    "description": "Original user query that is unsupported"
                }
            },
            "required": [],
            "additionalProperties": False
        }
    },
    "required": ["intent", "args"],
    "additionalProperties": False
}


# Pydantic models for type safety (optional, for internal validation)
from typing import Literal, Optional
from pydantic import BaseModel, Field


class IntentArgs(BaseModel):
    """Arguments for various intents."""
    paragraph_index: Optional[int] = Field(None, ge=1, description="1-based paragraph index")
    line_index: Optional[int] = Field(None, ge=1, description="1-based line index")
    style: Optional[Literal["short", "detailed"]] = Field(None, description="Summarization style")
    term: Optional[str] = Field(None, description="Search term")
    original_query: Optional[str] = Field(None, description="Original unsupported query")


class IntentExtraction(BaseModel):
    """Structured intent extraction result."""
    intent: Literal[
        "count_paragraphs",
        "has_bullets",
        "has_headings",
        "get_main_title",
        "get_page_number",
        "paragraph_word_count",
        "read_paragraph",
        "summarize_paragraph",
        "read_line",
        "where_is_term",
        "list_subtitles",
        "list_bullets_in_paragraph",
        "unsupported"
    ]
    args: IntentArgs


# Intent descriptions for documentation
INTENT_DESCRIPTIONS = {
    "count_paragraphs": "Count the number of paragraphs in the document",
    "has_bullets": "Check if document contains bulleted lists",
    "has_headings": "Check if document contains headings/subtitles",
    "get_main_title": "Get the main title of the document",
    "get_page_number": "Get the page number",
    "paragraph_word_count": "Count words in a specific paragraph (requires paragraph_index)",
    "read_paragraph": "Read a specific paragraph (requires paragraph_index)",
    "summarize_paragraph": "Summarize a specific paragraph (requires paragraph_index, optional style)",
    "read_line": "Read a specific line from a paragraph (requires paragraph_index and line_index)",
    "where_is_term": "Find where a term appears in the document (requires term)",
    "list_subtitles": "List all subtitles/headings in the document",
    "list_bullets_in_paragraph": "List bullets in a specific paragraph (requires paragraph_index)",
    "unsupported": "Query is outside the scope of supported commands"
}

