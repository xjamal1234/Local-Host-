# WebSocket Contract & Streaming Documentation

This document describes the final WebSocket contract for the NOOR server, including lifecycle events, chat functionality, authentication, and future streaming options.

---

## Overview

The NOOR server uses a single WebSocket endpoint (`/ws/guidance`) for:
- Real-time YOLO guidance during document capture
- Lifecycle events for processing pipeline
- Interactive chat about processed documents

All new events and chat messages use a **uniform JSON envelope**:

```json
{
  "type": "<event_type>",
  "data": { ... }
}
```

---

## Lifecycle Events

### 1. `processing_started` (Server → Client)

Emitted immediately after the best frame is frozen and **before** Phase 4 processing begins (image processing, OCR, GPT).

```json
{
  "type": "processing_started",
  "data": {
    "docId": "0007"
  }
}
```

**Client Action**: Display loading UI and stop camera/frame uploads.

### 2. `processing_completed` (Server → Client) ✅ **CHAT-READY SIGNAL**

Emitted **only after** OCR and GPT Layout processing finish successfully **and** `final_captures/{docId}/gpt_{docId}.json` is written to disk.

```json
{
  "type": "processing_completed",
  "data": {
    "docId": "0007",
    "scanned_color": "server/app/static/final_captures/0007/scanned_color_0007.jpg",
    "gpt_ready": true
  }
}
```

**Client Action**: Exit loading UI and enable chat interface.

---

## Chat Messages

### Inbound: `chat_user` (Client → Server)

User sends a chat query about a processed document.

```json
{
  "type": "chat_user",
  "data": {
    "docId": "0007",
    "text": "اقرأ الفقرة الأولى"
  }
}
```

### Outbound: `chat_assistant` (Server → Client)

Server responds with Arabic text and metadata (intent, args).

```json
{
  "type": "chat_assistant",
  "data": {
    "text": "الفقرة 1: العنود بحالة )...",
    "meta": {
      "intent": "read_paragraph",
      "args": { "paragraph_index": 1 }
    }
  }
}
```

---

## File-Based Gating

Chat is **gated by file existence**, not by in-memory mode.

**Rule**: When receiving `chat_user`:
1. Check if `final_captures/{docId}/gpt_{docId}.json` exists
2. If **not exists** → respond with:
   ```json
   {
     "type": "chat_assistant",
     "data": {
       "text": "جارٍ المعالجة... الرجاء الانتظار حتى اكتمال المعالجة.",
       "meta": {}
     }
   }
   ```
3. If **exists** → extract intent via OpenAI Responses API + execute locally

This allows users to:
- Chat about older documents even if the current session isn't in "processing_pending" mode
- Reconnect and chat without maintaining complex session state

---

## Authentication (Optional)

If `NOOR_CHAT_TOKEN` is set in environment variables, the WebSocket requires authentication.

### Methods:

1. **Query Parameter** (preferred):
   ```
   ws://localhost:8080/ws/guidance?token=<your_token>
   ```

2. **First Message** (fallback):
   ```json
   {
     "type": "auth",
     "data": {
       "token": "<your_token>"
     }
   }
   ```

**Precedence**: Query parameter takes precedence over first message.

### Authentication Failure

If authentication fails, the server closes the WebSocket with an error message:

```json
{
  "type": "error",
  "data": {
    "code": "AUTH_FAILED",
    "message": "تم رفض الاتصال: مفقود أو رمز وصول غير صالح."
  }
}
```

### REST API Authentication

For REST endpoint `POST /api/v1/chat`, use Bearer token:

```
Authorization: Bearer <your_token>
```

Returns `401 Unauthorized` if authentication fails.

---

## Rate Limiting

Chat messages are rate-limited per session: **5 messages per 10 seconds** (configurable).

### On Exceed:

1. **First excess**: Server sends throttle message:
   ```json
   {
     "type": "chat_assistant",
     "data": {
       "text": "تم الوصول إلى الحد المسموح للرسائل، الرجاء المحاولة بعد قليل.",
       "meta": { "intent": "rate_limited", "args": {} }
     }
   }
   ```

2. **Additional excess**: Silently dropped (logged server-side)

Rate limit resets after the 10-second window expires.

---

## Supported Chat Intents

The following intents are extracted via OpenAI Responses API with Structured Outputs and executed deterministically on `gpt_{docId}.json`:

| Intent | Example (Arabic) | Example (English) |
|--------|------------------|-------------------|
| `count_paragraphs` | "كم عدد الفقرات؟" | "how many paragraphs?" |
| `read_paragraph` | "اقرأ الفقرة الأولى" | "read paragraph 1" |
| `summarize_paragraph` | "لخص الفقرة 2" | "summarize paragraph 2" |
| `where_is_term` | "وين كلمة العنود؟" | "where is the word 'Anoud'?" |
| `list_subtitles` | "اعرض العناوين الفرعية" | "list subtitles" |
| `has_bullets` | "هل يوجد نقاط؟" | "are there bullets?" |
| `get_page_number` | "كم رقم الصفحة؟" | "what's the page number?" |
| `unsupported` | (any unrecognized query) | (fallback) |

### Unsupported Intent Response

Uses **central Arabic help text** from `app/core/chat/help_text.py`:

```
عذرًا، هذا الطلب غير مدعوم حاليًا.

الأوامر المدعومة:
• كم عدد الفقرات؟
• هل يوجد نقاط أو عناوين؟
• اقرأ الفقرة [رقم]
• لخص الفقرة [رقم]
• كم كلمة في ...

أمثلة:
1. "اقرأ الفقرة الأولى" → يقرأ محتوى الفقرة 1
2. "كم عدد الفقرات؟" → يعطي عدد الفقرات
3. "وين كلمة العنود؟" → يبحث عن الكلمة ويعطي موقعها
```

---

## Future: Streaming Long Responses

For very long chat responses (e.g., reading multi-page documents), we can implement streaming in one of two ways:

### Option 1: WebSocket Chunked Streaming

Send multiple `chat_assistant` chunks with a continuation flag:

```json
{
  "type": "chat_assistant",
  "data": {
    "text": "الفقرة 1: العنود...",
    "meta": { "intent": "read_paragraph", "args": {...}, "chunk": 1, "done": false }
  }
}
```

```json
{
  "type": "chat_assistant",
  "data": {
    "text": "...بحالة ",
    "meta": { "intent": "read_paragraph", "args": {...}, "chunk": 2, "done": true }
  }
}
```

**Advantages**:
- No new connection required
- Real-time user feedback (progressive display)
- Can cancel mid-stream

**Implementation**:
- Modify `JsonExecutor.execute()` to yield chunks
- Update WS handler to send chunks as they arrive
- Client concatenates chunks until `done: true`

### Option 2: Server-Sent Events (SSE) for REST

For REST clients that prefer HTTP:

```
GET /api/v1/chat/stream?docId=0007&text=اقرأ+الفقرة+الأولى
Authorization: Bearer <token>

event: chat_chunk
data: {"text": "الفقرة 1: العنود...", "chunk": 1, "done": false}

event: chat_chunk
data: {"text": "...بحالة ", "chunk": 2, "done": true}
```

**Advantages**:
- Standard HTTP/SSE (widely supported)
- Browser-native EventSource API
- Automatic reconnection on disconnect

**Implementation**:
- Create new endpoint: `GET /api/v1/chat/stream`
- Return `StreamingResponse` with `text/event-stream`
- Yield chunks as SSE events

### When to Implement Streaming?

**Consider streaming when**:
- Average response length > 500 characters
- User feedback indicates "too slow" for long reads
- Mobile network latency is high

**For MVP (current state)**:
- Return full response in one message
- Keep implementation simple
- Add streaming only if user testing shows need

---

## Configuration (Environment Variables)

```bash
# Chat Security
NOOR_CHAT_TOKEN=            # If set, requires authentication for chat (WS + REST)

# Rate Limiting
NOOR_CHAT_RATE_LIMIT_MESSAGES=5        # Max messages per window
NOOR_CHAT_RATE_LIMIT_WINDOW_SEC=10    # Window duration in seconds

# OpenAI (Intent Extraction + Layout)
NOOR_OPENAI_API_KEY=                   # OpenAI API key (required for chat)
NOOR_OPENAI_MODEL=gpt-4o-2024-08-06   # Responses-capable model with Structured Outputs
NOOR_OPENAI_API_BASE=https://api.openai.com/v1
NOOR_OPENAI_RESPONSES_ENDPOINT=/responses
NOOR_OPENAI_TIMEOUT_SEC=45
NOOR_OPENAI_MAX_RETRIES=2

# Chat Intent Extraction
NOOR_CHAT_INTENT_TIMEOUT_SEC=30
NOOR_CHAT_INTENT_MODEL=               # Defaults to OPENAI_MODEL

# Storage
NOOR_FINAL_FRAME_DIR=server/app/static/final_captures
```

---

## Logging Events

The following structured log events are emitted for chat and lifecycle:

| Event | When | Fields |
|-------|------|--------|
| `processing_started_emitted` | After best frame frozen, before Phase 4 | `session_id`, `docId` |
| `processing_completed_emitted` | After GPT JSON written | `session_id`, `docId`, `gpt_ready` |
| `chat_not_ready` | Chat received before GPT ready | `session_id`, `docId` |
| `chat_intent_extracted` | Intent extraction succeeded | `session_id`, `docId`, `intent`, `args` |
| `chat_execution_done` | Intent execution completed | `session_id`, `docId`, `intent` |
| `chat_reply_sent` | Response sent to client | `session_id`, `docId` |
| `chat_auth_failed` | Authentication failed | `session_id` |
| `chat_rate_limited` | Rate limit exceeded | `session_id` |

---

## Testing

### WebSocket Lifecycle + Chat Test

Run the comprehensive test:

```bash
cd server
python test_chat_lifecycle.py
```

This test verifies:
- Connection with optional authentication
- Enveloped `chat_user` message format
- `processing_started` and `processing_completed` lifecycle events
- `chat_assistant` response with `data.text` and `data.meta`
- "Processing not ready" message for non-existent documents

### Manual Testing

1. **Start server** (with or without `NOOR_CHAT_TOKEN`):
   ```bash
   cd server
   conda activate noor311
   export NOOR_CHAT_TOKEN=test_token_123  # Optional
   uvicorn app.main:app --reload --port 8080
   ```

2. **Capture a document** to completion (class 5 dominance)
   - Observe `processing_started` → `processing_completed` events
   - Note the `docId` (e.g., `0007`)

3. **Send chat messages** (WebSocket or REST):
   ```json
   { "type": "chat_user", "data": { "docId": "0007", "text": "كم عدد الفقرات؟" } }
   ```

4. **Test rate limiting**: Send 10 messages rapidly
   - First 5 succeed
   - 6th gets throttle message
   - 7th-10th silently dropped

5. **Test authentication** (if token set):
   - Without token → connection closed
   - With valid token → success

---

## Architecture Notes

### Why File-Based Gating?

**Problem**: In-memory modes don't survive:
- WebSocket reconnections
- Server restarts
- Multiple concurrent clients

**Solution**: Check disk for `gpt_{docId}.json` existence
- Persistent across restarts
- Robust for multi-client scenarios
- Simple truth source

### Why Enveloped Format for New Messages?

**Consistency**: All new events (lifecycle + chat) use `{ "type": "...", "data": {...} }`
- Makes client parsing uniform
- Extensible for future fields
- Industry-standard WebSocket pattern

**Backward Compatibility**: Legacy guidance messages remain unchanged
- No breaking changes for existing clients
- Gradual migration path

### Why Central Help Text?

**DRY Principle**: Single source of truth for help message
- Used by both WS and REST
- Easy to update supported commands
- Consistent user experience

**Maintainability**: Update once, apply everywhere

---

## Summary

✅ **Lifecycle events** signal processing pipeline progress  
✅ **File-based gating** ensures robust chat readiness check  
✅ **Enveloped format** standardizes new WS messages  
✅ **Authentication** protects chat from unauthorized access  
✅ **Rate limiting** prevents abuse  
✅ **Central help text** maintains consistency  
✅ **Streaming option** documented for future scaling

The WebSocket contract is now **production-ready** with clear contracts, robust error handling, and extensibility for future features.

