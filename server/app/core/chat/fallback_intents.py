"""
Fallback intent detection using regex patterns.

This module provides a lightweight rule-based intent parser as a fallback
when OpenAI GPT fails or returns unsupported intents.
"""

import re
from typing import Dict, Any, Optional, Tuple
from .text_utils import normalize_text, extract_numbers
from ...core.logger import log_info, log_debug


class FallbackIntentDetector:
    """Regex-based intent detector for common queries."""
    
    def __init__(self):
        """Initialize the fallback intent detector with compiled patterns."""
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Compile regex patterns for intent detection."""
        return {
            'count_paragraphs': {
                'patterns': [
                    r'(كم|قديش)\s*(عدد\s*)?(ال)?فقر(?:ة|ات)',
                    r'how\s*many\s*paragraphs?',
                    r'number\s*of\s*paragraphs?'
                ],
                'intent': 'count_paragraphs',
                'args': {}
            },
            
            'read_paragraph': {
                'patterns': [
                    r'(اقرا|اقري)\s*الفقرة\s*(الاولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة|\d+)',
                    r'read\s*paragraph\s*(\d+)'
                ],
                'intent': 'read_paragraph',
                'args': {}
            },
            
            'summarize_paragraph': {
                'patterns': [
                    r'(لخص|خص)\s*الفقرة\s*(الاولى|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة|\d+)',
                    r'summarize\s*paragraph\s*(\d+)'
                ],
                'intent': 'summarize_paragraph',
                'args': {}
            },
            
            'where_is_term': {
                'patterns': [
                    r'(?:أين)\s+(?:كلمة\s+)?["""\'\']?(.+?)["""\'\']?$',
                    r'where\s+is\s+["""\'\']?(.+?)["""\'\']?$'
                ],
                'intent': 'where_is_term',
                'args': {}
            },
            
            'has_bullets': {
                'patterns': [
                    r'(في|هل\s*في|هل\s*يوجد)\s*(نقاط|بولت|بوليت|تعداد\s*نقطي)',
                    r'has\s*bullets?'
                ],
                'intent': 'has_bullets',
                'args': {}
            },
            
            'list_subtitles': {
                'patterns': [
                    r'(اعرض|عرض|شو|ما)\s*(العناوين\s*الفرعية|العناوين|الهيدنجز|العناوين\s*الصغيرة)',
                    r'list\s*(sub\s*titles|headings)'
                ],
                'intent': 'list_subtitles',
                'args': {}
            },
            
            'paragraph_word_count': {
                'patterns': [
                    r'(كم|قديش)\s*(عدد\s*)?كلمات\s*الفقرة\s*(\d+)',
                    r'word\s*count\s*paragraph\s*(\d+)'
                ],
                'intent': 'paragraph_word_count',
                'args': {}
            },
            
            'get_page_number': {
                'patterns': [
                    r'(كم|قديش)\s*(رقم|نمبر)\s*الصفحة',
                    r'what\s*page\s*number'
                ],
                'intent': 'get_page_number',
                'args': {}
            }
        }
    
    def detect_intent_fallback(self, text: str, reason: str = "gpt_error") -> Dict[str, Any]:
        """
        Detect intent using fallback regex patterns.
        
        Args:
            text: User input text
            reason: Reason for using fallback ("gpt_error" or "gpt_unsupported")
            
        Returns:
            Dictionary with 'intent' and 'args' keys, or unsupported if no match
        """
        if not text or not text.strip():
            return {
                "intent": "unsupported",
                "args": {"original_query": text, "fallback_reason": reason}
            }
        
        # Normalize the input text
        normalized_text = normalize_text(text)
        
        # Try each intent pattern
        for intent_name, intent_config in self.patterns.items():
            for pattern in intent_config['patterns']:
                match = re.search(pattern, normalized_text, re.IGNORECASE)
                if match:
                    # Extract arguments based on the intent
                    args = self._extract_args(intent_name, match, normalized_text)
                    
                    # Handle invalid terms for where_is_term
                    if intent_name == 'where_is_term' and args.get('_invalid_term'):
                        return {
                            "intent": "unsupported",
                            "args": {"original_query": text, "fallback_reason": "bad_term"}
                        }
                    
                    # Special logging for where_is_term
                    if intent_name == 'where_is_term' and 'term' in args:
                        log_info("intent_fallback_used", f"Intent: {intent_name}, Term: {args['term']}", session_id="fallback")
                    
                    # Clean up internal flags
                    if '_invalid_term' in args:
                        del args['_invalid_term']
                    
                    return {
                        "intent": intent_name,
                        "args": args
                    }
        
        # No pattern matched
        return {
            "intent": "unsupported",
            "args": {"original_query": text, "fallback_reason": reason}
        }
    
    def _extract_args(self, intent_name: str, match: re.Match, text: str) -> Dict[str, Any]:
        """Extract arguments from regex match based on intent type."""
        args = {}
        
        if intent_name == 'read_paragraph':
            # For Arabic patterns, group 2 is the paragraph number/ordinal
            # For English patterns, group 1 is the paragraph number
            if len(match.groups()) >= 2 and match.group(2):
                paragraph_text = match.group(2)
            elif len(match.groups()) >= 1 and match.group(1):
                paragraph_text = match.group(1)
            else:
                paragraph_text = match.group(0)
            paragraph_num = self._parse_paragraph_number(paragraph_text)
            if paragraph_num:
                args['paragraph_index'] = paragraph_num
        
        elif intent_name == 'summarize_paragraph':
            # For Arabic patterns, group 2 is the paragraph number/ordinal
            # For English patterns, group 1 is the paragraph number
            if len(match.groups()) >= 2 and match.group(2):
                paragraph_text = match.group(2)
            elif len(match.groups()) >= 1 and match.group(1):
                paragraph_text = match.group(1)
            else:
                paragraph_text = match.group(0)
            paragraph_num = self._parse_paragraph_number(paragraph_text)
            if paragraph_num:
                args['paragraph_index'] = paragraph_num
            # Default to short style
            args['style'] = 'short'
        
        elif intent_name == 'where_is_term':
            if match.groups():
                # For both Arabic and English patterns, group 1 is the search term
                from .text_utils import clean_extracted_term
                raw_term = match.group(1)
                cleaned_term = clean_extracted_term(raw_term)
                if cleaned_term:
                    args['term'] = cleaned_term
                else:
                    # Log rejected term
                    log_debug("where_is_term_rejected", f"Term rejected: '{raw_term}' (too short/long/stop-term)", session_id="fallback")
                    # Mark as invalid term - will be handled by caller
                    args['_invalid_term'] = True
        
        elif intent_name == 'paragraph_word_count':
            if match.groups():
                # For both Arabic and English patterns, the last group is the paragraph number
                paragraph_text = match.group(len(match.groups()))
                paragraph_num = self._parse_paragraph_number(paragraph_text)
                if paragraph_num:
                    args['paragraph_index'] = paragraph_num
        
        return args
    
    def _parse_paragraph_number(self, text: str) -> Optional[int]:
        """Parse paragraph number from Arabic or English text."""
        # Arabic number words (both with and without diacritics)
        arabic_numbers = {
            'الأولى': 1, 'الثانية': 2, 'الثالثة': 3, 'الرابعة': 4, 'الخامسة': 5,
            'السادسة': 6, 'السابعة': 7, 'الثامنة': 8, 'التاسعة': 9, 'العاشرة': 10,
            'أولى': 1, 'ثانية': 2, 'ثالثة': 3, 'رابعة': 4, 'خامسة': 5,
            'سادسة': 6, 'سابعة': 7, 'ثامنة': 8, 'تاسعة': 9, 'عاشرة': 10,
            'الاولى': 1, 'الثانية': 2, 'الثالثة': 3, 'الرابعة': 4, 'الخامسة': 5,  # Without diacritics
            'السادسة': 6, 'السابعة': 7, 'الثامنة': 8, 'التاسعة': 9, 'العاشرة': 10
        }
        
        # Check for Arabic number words
        for arabic_word, number in arabic_numbers.items():
            if arabic_word in text:
                return number
        
        # Extract digits from text
        numbers = extract_numbers(text)
        if numbers:
            try:
                return int(numbers[0])
            except ValueError:
                pass
        
        return None


# Global instance
_fallback_detector = FallbackIntentDetector()


def detect_intent_fallback(text: str, reason: str = "gpt_error") -> Dict[str, Any]:
    """
    Convenience function to detect intent using fallback patterns.
    
    Args:
        text: User input text
        reason: Reason for using fallback
        
    Returns:
        Dictionary with 'intent' and 'args' keys
    """
    return _fallback_detector.detect_intent_fallback(text, reason)
