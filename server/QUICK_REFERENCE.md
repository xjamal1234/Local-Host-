# NOOR Server - Quick Reference Guide

## 🚀 Starting the Server

```bash
cd D:\Noor\server
conda activate noor311
$env:PYTHONPATH = (Get-Location).Path
uvicorn app.main:app --reload --port 8080
```

**With Authentication**:
```bash
$env:NOOR_CHAT_TOKEN = "your_secure_token_here"
uvicorn app.main:app --reload --port 8080
```

---

## 📡 WebSocket Connection

**Without Auth**:
```javascript
ws://localhost:8080/ws/guidance
```

**With Auth**:
```javascript
ws://localhost:8080/ws/guidance?token=your_secure_token_here
```

---

## 💬 Chat Message Format (Enveloped)

**Send** (Client → Server):
```json
{
  "type": "chat_user",
  "data": {
    "docId": "0007",
    "text": "اقرأ الفقرة الأولى"
  }
}
```

**Receive** (Server → Client):
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

## 🔄 Lifecycle Events

**Processing Started**:
```json
{
  "type": "processing_started",
  "data": { "docId": "0007" }
}
```

**Processing Completed** (Chat Ready):
```json
{
  "type": "processing_completed",
  "data": {
    "docId": "0007",
    "scanned_color": "path/to/image.jpg",
    "gpt_ready": true
  }
}
```

---

## 🔐 REST API with Auth

**Endpoint**: `POST /api/v1/chat`

**Headers**:
```
Authorization: Bearer your_secure_token_here
Content-Type: application/json
```

**Body**:
```json
{
  "docId": "0007",
  "text": "كم عدد الفقرات؟"
}
```

**Response**:
```json
{
  "text": "عدد الفقرات: 3",
  "meta": {
    "intent": "count_paragraphs",
    "args": {}
  }
}
```

---

## 🎯 Supported Chat Intents

| Intent | Arabic Example | English Example |
|--------|---------------|-----------------|
| `count_paragraphs` | كم عدد الفقرات؟ | how many paragraphs? |
| `read_paragraph` | اقرأ الفقرة الأولى | read paragraph 1 |
| `summarize_paragraph` | لخص الفقرة 2 | summarize paragraph 2 |
| `where_is_term` | وين كلمة العنود؟ | where is 'Anoud'? |
| `list_subtitles` | اعرض العناوين | list subtitles |
| `has_bullets` | هل يوجد نقاط؟ | are there bullets? |
| `get_page_number` | كم رقم الصفحة؟ | what's the page number? |

---

## ⚙️ Environment Variables

```bash
# Required
NOOR_OPENAI_API_KEY=sk-proj-...

# Security (Production)
NOOR_CHAT_TOKEN=your_secure_token

# Rate Limiting
NOOR_CHAT_RATE_LIMIT_MESSAGES=5
NOOR_CHAT_RATE_LIMIT_WINDOW_SEC=10

# OpenAI
NOOR_OPENAI_MODEL=gpt-4o-2024-08-06
NOOR_OPENAI_API_BASE=https://api.openai.com/v1
NOOR_OPENAI_RESPONSES_ENDPOINT=/responses
```

---

## 🧪 Testing

**Run WebSocket + Lifecycle Test**:
```bash
python test_chat_lifecycle.py
```

**Run WebSocket Chat Test**:
```bash
python test_chat_websocket.py
```

---

## 📊 Rate Limiting

- **Limit**: 5 messages per 10 seconds
- **First excess**: Throttle message
- **Additional**: Silently dropped
- **Reset**: After 10 seconds

**Throttle Message**:
> تم الوصول إلى الحد المسموح للرسائل، الرجاء المحاولة بعد قليل.

---

## 🔍 Common Log Events

```bash
# Processing lifecycle
grep "processing_started_emitted" server.log
grep "processing_completed_emitted" server.log

# Chat events
grep "chat_not_ready" server.log
grep "chat_intent_extracted" server.log
grep "chat_reply_sent" server.log

# Security
grep "chat_auth_failed" server.log
grep "chat_rate_limited" server.log
```

---

## 🐛 Troubleshooting

**Problem**: All intents return "unsupported"
- **Check**: `NOOR_OPENAI_API_KEY` is set and valid
- **Fix**: Set API key in `.env` or environment

**Problem**: WebSocket closes immediately
- **Check**: Token authentication if `NOOR_CHAT_TOKEN` is set
- **Fix**: Add `?token=...` to connection URL

**Problem**: "Processing not ready" after completion
- **Check**: `gpt_{docId}.json` exists in `final_captures/{docId}/`
- **Fix**: Check `processing_completed_emitted` log for errors

**Problem**: Rate limit too restrictive
- **Check**: Current limits (5 msg / 10s)
- **Fix**: Increase `NOOR_CHAT_RATE_LIMIT_MESSAGES`

---

## 📁 File Structure

```
final_captures/
├── 0001/
│   ├── scanned_color_0001.jpg
│   ├── ocr_0001.json
│   └── gpt_0001.json  ← Chat reads from here
├── 0002/
│   ├── scanned_color_0002.jpg
│   ├── ocr_0002.json
│   └── gpt_0002.json
...
```

---

## 📚 Documentation

- **`WS_CONTRACT_AND_STREAMING.md`**: Complete WebSocket spec
- **`PHASE7_WS_CONTRACT_FINALIZATION.md`**: Implementation details
- **`CHAT_IMPLEMENTATION.md`**: Chat system design
- **`QUICK_REFERENCE.md`**: This file

---

## 🎯 Quick Flow

1. **Connect** to WebSocket
2. **Capture** document (YOLO guidance)
3. **Receive** `processing_started` event
4. **Wait** for `processing_completed` (gpt_ready: true)
5. **Chat** about the document
6. **Rate limit**: Max 5 messages per 10 seconds

---

## 🔒 Security Checklist

- [ ] Set `NOOR_CHAT_TOKEN` for production
- [ ] Use HTTPS/WSS (TLS encryption)
- [ ] Configure CORS for allowed origins
- [ ] Monitor rate-limit logs
- [ ] Set up log aggregation
- [ ] Enable firewall rules

---

## 📞 Quick Help

**API Base URL**: `http://localhost:8080`
**WebSocket URL**: `ws://localhost:8080/ws/guidance`
**API Docs**: `http://localhost:8080/docs`
**Health Check**: `http://localhost:8080/api/v1/health`

