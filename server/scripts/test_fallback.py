#!/usr/bin/env python3
"""
Test script for fallback intent detection functionality.

This script tests the fallback parser with various Arabic and English inputs
to ensure it correctly identifies intents when GPT fails.
"""

import sys
import os
from pathlib import Path

# Add the server app directory to Python path
server_dir = Path(__file__).parent.parent
sys.path.insert(0, str(server_dir))

from app.core.chat.fallback_intents import detect_intent_fallback
from app.core.chat.text_utils import normalize_text


def test_fallback_intents():
    """Test fallback intent detection with various inputs."""
    
    print("🧪 Testing Fallback Intent Detection")
    print("=" * 50)
    
    # Test cases: (input_text, expected_intent, expected_args)
    test_cases = [
        # count_paragraphs variations
        ("كم عدد الفقرات؟", "count_paragraphs", {}),
        ("كم فقرة؟", "count_paragraphs", {}),
        ("قديش في فقرات؟", "count_paragraphs", {}),
        ("قديش عدد الفقرات بالنص؟", "count_paragraphs", {}),
        ("how many paragraphs?", "count_paragraphs", {}),
        ("number of paragraphs", "count_paragraphs", {}),
        
        # read_paragraph variations
        ("اقرأ الفقرة الأولى", "read_paragraph", {"paragraph_index": 1}),
        ("اقرأ الفقرة الثانية", "read_paragraph", {"paragraph_index": 2}),
        ("اقري الفقرة 3", "read_paragraph", {"paragraph_index": 3}),
        ("read paragraph 1", "read_paragraph", {"paragraph_index": 1}),
        ("show paragraph 2", "read_paragraph", {"paragraph_index": 2}),
        
        # summarize_paragraph variations
        ("لخص الفقرة الأولى", "summarize_paragraph", {"paragraph_index": 1, "style": "short"}),
        ("خلاصة الفقرة الثانية", "summarize_paragraph", {"paragraph_index": 2, "style": "short"}),
        ("summarize paragraph 3", "summarize_paragraph", {"paragraph_index": 3, "style": "short"}),
        
        # where_is_term variations
        ("وين كلمة مهم؟", "where_is_term", {"term": "مهم"}),
        ("أين كلمة نظام؟", "where_is_term", {"term": "نظام"}),
        ("فين كلمة العنود؟", "where_is_term", {"term": "العنود"}),
        ("where is important?", "where_is_term", {"term": "important"}),
        ("find 'system'", "where_is_term", {"term": "system"}),
        
        # has_bullets variations
        ("هل فيه نقاط؟", "has_bullets", {}),
        ("فيه نقاط مهمة؟", "has_bullets", {}),
        ("في بولِت؟", "has_bullets", {}),
        ("are there bullets?", "has_bullets", {}),
        ("bullets?", "has_bullets", {}),
        
        # list_subtitles variations
        ("اعرض العناوين الفرعية", "list_subtitles", {}),
        ("شو العناوين الموجودة", "list_subtitles", {}),
        ("list subtitles", "list_subtitles", {}),
        ("headings?", "list_subtitles", {}),
        
        # count_words variations
        ("كم عدد الكلمات؟", "count_words", {}),
        ("قديش كلمة؟", "count_words", {}),
        ("how many words?", "count_words", {}),
        ("word count", "count_words", {}),
        
        # get_page_number variations
        ("ما رقم الصفحة؟", "get_page_number", {}),
        ("كم صفحة؟", "get_page_number", {}),
        ("what page?", "get_page_number", {}),
        ("page number", "get_page_number", {}),
        
        # unsupported cases
        ("ترجم هذا للإنجليزية", "unsupported", {}),
        ("solve this math problem", "unsupported", {}),
        ("اكتب لي بحث", "unsupported", {}),
    ]
    
    passed = 0
    failed = 0
    
    for i, (input_text, expected_intent, expected_args) in enumerate(test_cases, 1):
        print(f"\n{i:2d}. Testing: '{input_text}'")
        
        # Test fallback detection
        result = detect_intent_fallback(input_text, reason="test")
        detected_intent = result.get("intent")
        detected_args = result.get("args", {})
        
        # Check intent match
        intent_match = detected_intent == expected_intent
        
        # Check args match (for supported intents)
        args_match = True
        if expected_intent != "unsupported":
            for key, expected_value in expected_args.items():
                if detected_args.get(key) != expected_value:
                    args_match = False
                    break
        
        if intent_match and args_match:
            print(f"    ✅ PASS - Intent: {detected_intent}, Args: {detected_args}")
            passed += 1
        else:
            print(f"    ❌ FAIL - Expected: {expected_intent} {expected_args}")
            print(f"             Got:      {detected_intent} {detected_args}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"⚠️  {failed} tests failed")
        return False


def test_text_normalization():
    """Test text normalization functionality."""
    
    print("\n🔤 Testing Text Normalization")
    print("=" * 50)
    
    test_cases = [
        # Diacritics removal
        ("كَمْ عَدَدُ الْفَقَرَاتِ؟", "كم عدد الفقرات"),
        ("قَدِيشْ فِي فِقَرَاتٍ؟", "قديش في فقرات"),
        
        # Arabic digits to Western
        ("اقرأ الفقرة ١", "اقرأ الفقرة 1"),
        ("اقرأ الفقرة ٢", "اقرأ الفقرة 2"),
        ("اقرأ الفقرة ٣", "اقرأ الفقرة 3"),
        
        # Hamza normalization
        ("أين كلمة إهمال؟", "اين كلمة اهمال"),
        ("آخر فقرة", "اخر فقرة"),
        
        # ة to ه at word end
        ("الفقرة الأولى", "الفقرة الاولى"),
        ("الفقرة الثانية", "الفقرة الثانية"),
        
        # Punctuation and whitespace
        ("كم عدد الفقرات؟؟؟", "كم عدد الفقرات"),
        ("كم    عدد    الفقرات", "كم عدد الفقرات"),
        
        # Colloquial mappings
        ("قديش في فقرات؟", "كم في فقرات"),
        ("وين كلمة مهم؟", "اين كلمة مهم"),
    ]
    
    passed = 0
    failed = 0
    
    for i, (input_text, expected) in enumerate(test_cases, 1):
        print(f"\n{i:2d}. Testing: '{input_text}'")
        
        normalized = normalize_text(input_text)
        
        if normalized == expected:
            print(f"    ✅ PASS - '{normalized}'")
            passed += 1
        else:
            print(f"    ❌ FAIL - Expected: '{expected}'")
            print(f"             Got:      '{normalized}'")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All normalization tests passed!")
        return True
    else:
        print(f"⚠️  {failed} normalization tests failed")
        return False


def main():
    """Run all tests."""
    print("🚀 Phase 7b Fallback Intent Detection Tests")
    print("=" * 60)
    
    # Test text normalization
    norm_success = test_text_normalization()
    
    # Test fallback intents
    intent_success = test_fallback_intents()
    
    print("\n" + "=" * 60)
    if norm_success and intent_success:
        print("🎉 ALL TESTS PASSED! Fallback system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
