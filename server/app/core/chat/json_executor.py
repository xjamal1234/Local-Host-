"""
Deterministic JSON executor for chat intents.

This module executes intents against gpt_####.json files using only local,
deterministic operations. No LLM calls are made during execution.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from app.core.logger import log_info, log_error, log_debug


# Standard unsupported intent help message (Arabic)
UNSUPPORTED_HELP_MESSAGE = """عذرًا، هذا الطلب غير مدعوم حاليًا.

الأوامر المدعومة:
• كم عدد الفقرات؟
• هل يوجد نقاط أو عناوين؟
• اقرأ الفقرة [رقم]
• لخص الفقرة [رقم]
• كم كلمة في الفقرة [رقم]؟
• اقرأ السطر [رقم] من الفقرة [رقم]
• أين كلمة [المصطلح]؟
• ما هو العنوان الرئيسي؟
• اعرض العناوين الفرعية
• ما رقم الصفحة؟

أمثلة:
- "اقرأ الفقرة الأولى"
- "لخص الفقرة الثانية بالتفصيل"
- "وين كلمة العنود؟"
"""


class JsonExecutor:
    """Executor for deterministic intent operations on GPT JSON files."""
    
    def __init__(self, base_dir: str = "server/app/static/final_captures"):
        """
        Initialize JSON executor.
        
        Args:
            base_dir: Base directory for final captures
        """
        self.base_dir = Path(base_dir)
    
    def _load_gpt_json(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Load gpt_####.json file for a given document ID.
        
        Args:
            doc_id: 4-digit document ID (e.g., "0007")
            
        Returns:
            Parsed JSON dictionary or None if file not found
        """
        try:
            # Construct path: final_captures/####/gpt_####.json
            json_path = self.base_dir / doc_id / f"gpt_{doc_id}.json"
            
            if not json_path.exists():
                log_error("gpt_json_not_found", f"GPT JSON not found: {json_path}")
                return None
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            log_debug("gpt_json_loaded", f"Loaded GPT JSON for docId: {doc_id}")
            return data
            
        except Exception as e:
            log_error("gpt_json_load_error", f"Failed to load GPT JSON for {doc_id}: {str(e)}")
            return None
    
    def execute(
        self,
        intent: str,
        args: Dict[str, Any],
        doc_id: str,
        session_id: Optional[str] = None
    ) -> str:
        """
        Execute an intent against a GPT JSON file.
        
        Args:
            intent: Intent name
            args: Intent arguments
            doc_id: 4-digit document ID
            session_id: Optional session ID for logging
            
        Returns:
            Arabic text response
        """
        log_info("chat_execution_start", f"Executing intent: {intent}", session_id=session_id)
        
        # Handle unsupported intent
        if intent == "unsupported":
            return UNSUPPORTED_HELP_MESSAGE
        
        # Load GPT JSON
        data = self._load_gpt_json(doc_id)
        if data is None:
            return f"عذرًا، لم يتم العثور على المستند {doc_id}. الرجاء التأكد من رقم المستند والمحاولة مرة أخرى."
        
        # Execute intent
        try:
            if intent == "count_paragraphs":
                return self._count_paragraphs(data)
            elif intent == "has_bullets":
                return self._has_bullets(data)
            elif intent == "has_headings":
                return self._has_headings(data)
            elif intent == "get_main_title":
                return self._get_main_title(data)
            elif intent == "get_page_number":
                return self._get_page_number(data)
            elif intent == "paragraph_word_count":
                return self._paragraph_word_count(data, args.get("paragraph_index"))
            elif intent == "read_paragraph":
                return self._read_paragraph(data, args.get("paragraph_index"))
            elif intent == "summarize_paragraph":
                return self._summarize_paragraph(
                    data,
                    args.get("paragraph_index"),
                    args.get("style", "short")
                )
            elif intent == "read_line":
                return self._read_line(
                    data,
                    args.get("paragraph_index"),
                    args.get("line_index")
                )
            elif intent == "where_is_term":
                return self._where_is_term(data, args.get("term"))
            elif intent == "list_subtitles":
                return self._list_subtitles(data)
            elif intent == "list_bullets_in_paragraph":
                return self._list_bullets_in_paragraph(data, args.get("paragraph_index"))
            else:
                return UNSUPPORTED_HELP_MESSAGE
                
        except Exception as e:
            log_error("chat_execution_error", f"Execution error for {intent}: {str(e)}", session_id=session_id)
            return f"حدث خطأ أثناء تنفيذ الأمر. الرجاء المحاولة مرة أخرى."
    
    # ========== Intent Implementations ==========
    
    def _count_paragraphs(self, data: Dict[str, Any]) -> str:
        """Count paragraphs in the document."""
        count = len(data.get("paragraphs", []))
        return f"عدد الفقرات: {count}"
    
    def _has_bullets(self, data: Dict[str, Any]) -> str:
        """Check if document has bulleted lists."""
        has_lists = data.get("metadata", {}).get("has_lists", False)
        
        # Also check for list_marker in lines
        for para in data.get("paragraphs", []):
            for line in para.get("lines", []):
                if line.get("list_marker"):
                    return "نعم، يوجد نقاط في المستند."
        
        if has_lists:
            return "نعم، يوجد نقاط في المستند."
        else:
            return "لا، لا يوجد نقاط في المستند."
    
    def _has_headings(self, data: Dict[str, Any]) -> str:
        """Check if document has headings."""
        subtitles = data.get("subtitles", [])
        if subtitles and len(subtitles) > 0:
            return "نعم، يوجد عناوين في المستند."
        else:
            return "لا، لا يوجد عناوين في المستند."
    
    def _get_main_title(self, data: Dict[str, Any]) -> str:
        """Get the main title of the document."""
        title = data.get("title")
        if title:
            return f"العنوان الرئيسي: {title}"
        else:
            return "لا يوجد عنوان رئيسي في المستند."
    
    def _get_page_number(self, data: Dict[str, Any]) -> str:
        """Get the page number."""
        page_num = data.get("page", {}).get("number")
        if page_num:
            return f"رقم الصفحة: {page_num}"
        else:
            return "رقم الصفحة غير متوفر."
    
    def _paragraph_word_count(self, data: Dict[str, Any], paragraph_index: Optional[int]) -> str:
        """Count words in a specific paragraph."""
        if paragraph_index is None:
            return "الرجاء تحديد رقم الفقرة."
        
        paragraphs = data.get("paragraphs", [])
        
        # Validate index (1-based)
        if paragraph_index < 1 or paragraph_index > len(paragraphs):
            return f"الفقرة {paragraph_index} غير موجودة (عدد الفقرات: {len(paragraphs)})."
        
        # Get paragraph (convert to 0-based)
        para = paragraphs[paragraph_index - 1]
        
        # Count words from lines
        word_count = 0
        for line in para.get("lines", []):
            words = line.get("words", [])
            word_count += len(words)
        
        return f"عدد الكلمات في الفقرة {paragraph_index}: {word_count}"
    
    def _read_paragraph(self, data: Dict[str, Any], paragraph_index: Optional[int]) -> str:
        """Read a specific paragraph."""
        if paragraph_index is None:
            return "الرجاء تحديد رقم الفقرة."
        
        paragraphs = data.get("paragraphs", [])
        
        # Validate index
        if paragraph_index < 1 or paragraph_index > len(paragraphs):
            return f"الفقرة {paragraph_index} غير موجودة (عدد الفقرات: {len(paragraphs)})."
        
        # Get paragraph text
        para = paragraphs[paragraph_index - 1]
        text = para.get("text", "")
        
        if not text:
            # Fallback: concatenate lines
            lines = para.get("lines", [])
            text = " ".join([line.get("text", "") for line in lines])
        
        if text:
            return f"الفقرة {paragraph_index}:\n{text}"
        else:
            return f"الفقرة {paragraph_index} فارغة."
    
    def _summarize_paragraph(
        self,
        data: Dict[str, Any],
        paragraph_index: Optional[int],
        style: str = "short"
    ) -> str:
        """
        Summarize a paragraph using extractive summarization.
        
        - short: first sentence, max 200 chars
        - detailed: first two lines, max 350 chars
        """
        if paragraph_index is None:
            return "الرجاء تحديد رقم الفقرة."
        
        paragraphs = data.get("paragraphs", [])
        
        # Validate index
        if paragraph_index < 1 or paragraph_index > len(paragraphs):
            return f"الفقرة {paragraph_index} غير موجودة (عدد الفقرات: {len(paragraphs)})."
        
        # Get paragraph
        para = paragraphs[paragraph_index - 1]
        text = para.get("text", "")
        
        if not text:
            # Fallback: concatenate lines
            lines = para.get("lines", [])
            text = " ".join([line.get("text", "") for line in lines])
        
        if not text:
            return f"الفقرة {paragraph_index} فارغة."
        
        # Extractive summarization
        if style == "short":
            # First sentence, max 200 chars
            # Find first sentence terminator
            sentence_match = re.search(r'^[^.!?؟]+[.!?؟]', text)
            if sentence_match:
                summary = sentence_match.group(0).strip()
            else:
                summary = text[:200]
            
            summary = summary[:200]
            return f"ملخص الفقرة {paragraph_index} (مختصر):\n{summary}"
        
        else:  # detailed
            # First two lines, max 350 chars
            lines = para.get("lines", [])
            if len(lines) >= 2:
                summary = f"{lines[0].get('text', '')} {lines[1].get('text', '')}"
            elif len(lines) == 1:
                summary = lines[0].get('text', '')
            else:
                summary = text[:350]
            
            summary = summary[:350]
            return f"ملخص الفقرة {paragraph_index} (مفصل):\n{summary}"
    
    def _read_line(
        self,
        data: Dict[str, Any],
        paragraph_index: Optional[int],
        line_index: Optional[int]
    ) -> str:
        """Read a specific line from a paragraph."""
        if paragraph_index is None or line_index is None:
            return "الرجاء تحديد رقم الفقرة ورقم السطر."
        
        paragraphs = data.get("paragraphs", [])
        
        # Validate paragraph index
        if paragraph_index < 1 or paragraph_index > len(paragraphs):
            return f"الفقرة {paragraph_index} غير موجودة (عدد الفقرات: {len(paragraphs)})."
        
        # Get paragraph
        para = paragraphs[paragraph_index - 1]
        lines = para.get("lines", [])
        
        # Validate line index
        if line_index < 1 or line_index > len(lines):
            return f"السطر {line_index} غير موجود في الفقرة {paragraph_index} (عدد الأسطر: {len(lines)})."
        
        # Get line text
        line = lines[line_index - 1]
        text = line.get("text", "")
        
        if text:
            return f"السطر {line_index} من الفقرة {paragraph_index}:\n{text}"
        else:
            return f"السطر {line_index} من الفقرة {paragraph_index} فارغ."
    
    def _where_is_term(self, data: Dict[str, Any], term: Optional[str]) -> str:
        """Find where a term appears in the document (up to 5 hits)."""
        if not term:
            return "الرجاء تحديد الكلمة أو العبارة المراد البحث عنها."
        
        term_lower = term.lower()
        hits: List[str] = []
        
        paragraphs = data.get("paragraphs", [])
        
        for para_idx, para in enumerate(paragraphs, start=1):
            lines = para.get("lines", [])
            for line_idx, line in enumerate(lines, start=1):
                line_text = line.get("text", "")
                if term_lower in line_text.lower():
                    hits.append(f"الفقرة {para_idx}، السطر {line_idx}")
                    
                    if len(hits) >= 5:  # Limit to 5 hits
                        break
            
            if len(hits) >= 5:
                break
        
        if hits:
            locations = "، ".join(hits)
            return f"العبارة '{term}' موجودة في:\n{locations}"
        else:
            return f"لم يتم العثور على العبارة '{term}' في المستند."
    
    def _list_subtitles(self, data: Dict[str, Any]) -> str:
        """List all subtitles (headings with level > 1)."""
        subtitles = data.get("subtitles", [])
        
        if not subtitles or len(subtitles) == 0:
            return "لا يوجد عناوين فرعية في المستند."
        
        # Filter subtitles with level > 1 (exclude main title)
        filtered = [s for s in subtitles if s.get("level", 1) > 1]
        
        if not filtered:
            return "لا يوجد عناوين فرعية في المستند."
        
        # Format list
        result = "العناوين الفرعية:\n"
        for idx, subtitle in enumerate(filtered, start=1):
            text = subtitle.get("text", "")
            result += f"{idx}. {text}\n"
        
        return result.strip()
    
    def _list_bullets_in_paragraph(
        self,
        data: Dict[str, Any],
        paragraph_index: Optional[int]
    ) -> str:
        """List bullets in a specific paragraph."""
        if paragraph_index is None:
            return "الرجاء تحديد رقم الفقرة."
        
        paragraphs = data.get("paragraphs", [])
        
        # Validate index
        if paragraph_index < 1 or paragraph_index > len(paragraphs):
            return f"الفقرة {paragraph_index} غير موجودة (عدد الفقرات: {len(paragraphs)})."
        
        # Get paragraph
        para = paragraphs[paragraph_index - 1]
        lines = para.get("lines", [])
        
        # Find lines with list markers
        bullets = []
        for line in lines:
            marker = line.get("list_marker")
            text = line.get("text", "")
            if marker and text:
                bullets.append(text)
        
        if bullets:
            result = f"النقاط في الفقرة {paragraph_index}:\n"
            for idx, bullet in enumerate(bullets, start=1):
                result += f"• {bullet}\n"
            return result.strip()
        else:
            return f"لا يوجد نقاط في الفقرة {paragraph_index}."

