# GPT-Chatting (Server) Implementation

## Overview

This document describes the implementation of GPT-Chatting feature with structured intent extraction (OpenAI Responses API + Structured Outputs) and deterministic local execution against `gpt_####.json`.

## Architecture

The chat system follows a clean architecture pattern with clear separation of concerns:

1. **Intent Extraction** - OpenAI Responses API with Structured Outputs
2. **Deterministic Execution** - Local JSON operations (no LLM calls)
3. **Dual Entry Points** - WebSocket and REST API
4. **Arabic-First** - All responses in Arabic regardless of input language

## Components

### 1. Intent Schema (`app/core/chat/intents_schema.py`)

Defines strict JSON Schema for 13 supported intents:

- `count_paragraphs` - Count paragraphs in document
- `has_bullets` - Check for bulleted lists
- `has_headings` - Check for headings/subtitles
- `get_main_title` - Get main document title
- `get_page_number` - Get page number
- `paragraph_word_count` - Count words in specific paragraph
- `read_paragraph` - Read specific paragraph
- `summarize_paragraph` - Summarize paragraph (short/detailed)
- `read_line` - Read specific line from paragraph
- `where_is_term` - Find term in document (up to 5 hits)
- `list_subtitles` - List all subtitles (level > 1)
- `list_bullets_in_paragraph` - List bullets in specific paragraph
- `unsupported` - Fallback for out-of-scope queries

Schema enforces:
- 1-based indexing for paragraphs and lines
- Type safety with enums and constraints
- Strict argument validation

### 2. Prompt Templates (`app/core/chat/prompt_templates.py`)

System prompt and few-shot examples for intent extraction:

- **System Prompt**: Clear instruction listing all allowed commands
- **Few-Shot Examples**: 40+ examples covering:
  - Modern Standard Arabic (MSA)
  - Gulf Arabic colloquial
  - English queries
  - Edge cases and unsupported queries

### 3. OpenAI Intents Client (`app/services/openai_intents.py`)

**Key Features:**
- Uses OpenAI Responses API with Structured Outputs
- Model: `gpt-4o-2024-08-06` (Structured Outputs capable)
- Temperature: 0.0 (deterministic)
- Retry logic with exponential backoff
- Schema validation using Pydantic
- Fallback to `unsupported` on any error

**Configuration:**
```python
OPENAI_API_KEY = env("NOOR_OPENAI_API_KEY")
CHAT_INTENT_MODEL = env("NOOR_CHAT_INTENT_MODEL", default="gpt-4o-2024-08-06")
CHAT_INTENT_TIMEOUT_SEC = env("NOOR_CHAT_INTENT_TIMEOUT_SEC", default=30)
```

### 4. JSON Executor (`app/core/chat/json_executor.py`)

Deterministic, local-only execution against `gpt_####.json`:

**Extractive Summarization Rules:**
- `short` style: First sentence, max 200 chars
- `detailed` style: First two lines, max 350 chars

**Search Rules:**
- Case-insensitive
- Returns up to 5 hits with locations

**Error Handling:**
- Bounds checking for all indices
- Polite Arabic error messages
- "not available" for missing fields

**No LLM Calls:** All operations are deterministic string/array operations.

### 5. WebSocket Integration (`app/interfaces/ws/guidance.py`)

**New Message Type:**

Inbound:
```json
{
  "type": "chat_user",
  "docId": "0007",
  "text": "اقرأ الفقرة الأولى"
}
```

Outbound:
```json
{
  "type": "chat_assistant",
  "text": "الفقرة 1:\nالعنود بحالة...",
  "meta": {
    "intent": "read_paragraph",
    "args": {"paragraph_index": 1}
  }
}
```

**Session State:**
- Chat only accepted after `mode == "processing_pending"`
- Before processing complete: "جارٍ المعالجة... الرجاء الانتظار"
- WebSocket stays open after final capture

**Logging:**
- `chat_intent_extracted` - Intent and args
- `chat_execution_done` - Execution complete
- `chat_reply_sent` - Response sent

### 6. REST Endpoint (`app/interfaces/api/chat.py`)

**Endpoint:** `POST /api/v1/chat`

**Request:**
```json
{
  "docId": "0007",
  "text": "كم عدد الفقرات؟",
  "locale": "ar"  // optional
}
```

**Response:**
```json
{
  "text": "عدد الفقرات: 3",
  "meta": {
    "intent": "count_paragraphs",
    "args": {}
  }
}
```

**Error Responses:**
- 404: Document not found
- 500: Processing error

### 7. Dependency Injection (`app/core/di.py`)

Wired services:
- `intents_client` - OpenAIIntentsClient
- `json_executor` - JsonExecutor

Getters:
- `container.get_intents_client()`
- `container.get_json_executor()`

## Configuration

### Environment Variables

```bash
# OpenAI Configuration (reused from Phase 6)
NOOR_OPENAI_API_KEY="sk-..."
NOOR_OPENAI_MODEL="gpt-4o-2024-08-06"
NOOR_OPENAI_API_BASE="https://api.openai.com/v1"

# Chat-Specific Configuration
NOOR_CHAT_INTENT_TIMEOUT_SEC=30
NOOR_CHAT_INTENT_MODEL="gpt-4o-2024-08-06"  # Defaults to OPENAI_MODEL
```

## Data Flow

### WebSocket Flow

1. User sends `chat_user` message with `docId` and `text`
2. Check if session mode is `processing_pending` (GPT JSON ready)
3. Extract intent using OpenAI Responses API
4. Execute intent deterministically against `gpt_####.json`
5. Send `chat_assistant` response with Arabic text + metadata

### REST Flow

1. Client sends POST to `/api/v1/chat` with `docId` and `text`
2. Extract intent using OpenAI Responses API
3. Execute intent deterministically against `gpt_####.json`
4. Return JSON response with Arabic text + metadata

## GPT JSON Structure

Expected structure in `final_captures/####/gpt_####.json`:

```json
{
  "docId": "...",
  "schema_version": "1.0",
  "page": {
    "number": null
  },
  "title": null,
  "subtitles": [
    {"text": "...", "level": 2}
  ],
  "paragraphs": [
    {
      "id": "1",
      "role": "paragraph",
      "text": "...",
      "lines": [
        {
          "id": "1.1",
          "text": "...",
          "words": ["...", "..."],
          "list_marker": null
        }
      ]
    }
  ],
  "metadata": {
    "has_lists": false,
    "has_tables": false,
    "has_figures": false
  },
  "metrics": {
    "paragraph_count": 1,
    "line_count": 4,
    "word_count": 15
  }
}
```

## Error Handling

### Unsupported Intent

When intent extraction returns `unsupported` or query is out of scope:

```
عذرًا، هذا الطلب غير مدعوم حاليًا.

الأوامر المدعومة:
• كم عدد الفقرات؟
• هل يوجد نقاط أو عناوين؟
• اقرأ الفقرة [رقم]
• لخص الفقرة [رقم]
...

أمثلة:
- "اقرأ الفقرة الأولى"
- "وين كلمة العنود؟"
```

### Out of Range

When indices are invalid:

```
الفقرة 5 غير موجودة (عدد الفقرات: 3).
```

### Missing Document

When `gpt_####.json` not found:

```
عذرًا، لم يتم العثور على المستند 0007.
```

## Testing

### Manual QA Checklist

**Basic Commands:**
- ✓ "كم عدد الفقرات؟"
- ✓ "فيه عناوين؟"
- ✓ "ما هو العنوان الرئيسي؟"

**Paragraph Operations:**
- ✓ "اقرأ الفقرة الأولى"
- ✓ "لخص الفقرة الثانية"
- ✓ "كم كلمة في الفقرة الثالثة؟"

**Line Operations:**
- ✓ "اقرأ السطر 2 من الفقرة 3"

**Search:**
- ✓ "وين كلمة العنود؟"
- ✓ "ابحث عن كلمة نظام"

**Colloquial/English:**
- ✓ "شو العنوان؟" (Gulf Arabic)
- ✓ "how many paragraphs?" (English)

**Unsupported:**
- ✓ "ترجم للإنجليزية" → Unsupported help message
- ✓ "solve this equation" → Unsupported help message

**Edge Cases:**
- ✓ Out of range indices
- ✓ Missing document ID
- ✓ Chat before processing complete

### Test Documents

Use `final_captures/0003/gpt_0003.json` for testing with docId `0003`.

## Logging Events

- `chat_intent_extracted` - Intent and arguments extracted
- `chat_execution_done` - Local execution completed
- `chat_reply_sent` - Response sent to client
- `chat_handler_error` - Error during chat handling
- `chat_api_request` - REST API request received
- `intent_extraction_start` - Starting intent extraction
- `intent_extraction_success` - Intent extracted successfully
- `intent_extraction_failed` - All extraction attempts failed

## Security & Constraints

1. **No Image Processing:** Chat only reads from GPT JSON
2. **No OCR:** Chat does not trigger OCR
3. **Read-Only:** No mutations to stored data
4. **Strict Schema:** OpenAI Structured Outputs guarantee conformance
5. **Rate Limiting:** Inherits WebSocket rate limits
6. **Timeout:** 30-second default for intent extraction
7. **Locale:** Always responds in Arabic

## Future Enhancements (Not Implemented)

- Streaming responses (YAGNI for now)
- Voice input/output integration
- Context-aware follow-up questions
- Multi-document chat
- Paragraph highlighting/navigation
- Custom summarization parameters

## Dependencies

No new dependencies added. Uses existing:
- `httpx` - Async HTTP client
- `pydantic` - Schema validation
- `fastapi` - REST endpoints
- `openai` schema format - Structured Outputs

## File Structure

```
server/app/
├── core/
│   ├── chat/
│   │   ├── intents_schema.py      # Intent definitions
│   │   ├── prompt_templates.py    # System prompts + few-shot examples
│   │   └── json_executor.py       # Deterministic execution
│   ├── config.py                   # Added chat config vars
│   └── di.py                       # Wired chat services
├── services/
│   └── openai_intents.py          # Intent extraction client
├── interfaces/
│   ├── api/
│   │   └── chat.py                # REST endpoint
│   └── ws/
│       └── guidance.py            # Added chat message handler
└── main.py                        # Registered chat router
```

## Acceptance Criteria

✅ WebSocket accepts `chat_user` messages after processing complete
✅ Intent extraction via OpenAI Responses API with Structured Outputs
✅ Deterministic local execution from `gpt_####.json`
✅ Arabic-only text responses
✅ Metadata includes intent and args
✅ REST endpoint `POST /api/v1/chat` implemented
✅ Unsupported queries handled with helpful message
✅ Out-of-range indices handled gracefully
✅ Structured logging for all chat events
✅ No new external dependencies

## API Documentation

Full API documentation available at:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

The chat endpoint will be visible under the "chat" tag.

