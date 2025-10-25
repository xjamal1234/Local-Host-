#!/usr/bin/env python3
"""
Comprehensive tests for where_is_term intent hardening.

This script tests the improved where_is_term functionality including:
- Text normalization and colloquial mapping
- Regex pattern matching
- Term extraction and cleaning
- Quote handling
- Stop term rejection
- Integration with fallback intent detection
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.chat.fallback_intents import detect_intent_fallback
from app.core.chat.text_utils import normalize_text, clean_extracted_term, strip_smart_quotes


def test_text_normalization():
    """Test text normalization for where_is_term queries."""
    print("🧪 Testing Text Normalization")
    print("=" * 50)
    
    test_cases = [
        # Colloquial to MSA mapping
        ("وين كلمة مهم؟", "أين كلمة مهم"),
        ("فين كلمة مهم؟", "أين كلمة مهم"),
        ("وين مهم؟", "أين مهم"),
        ("فين مهم؟", "أين مهم"),
        
        # With quotes
        ("وين كلمة 'مهم'؟", "أين كلمة 'مهم'"),
        ('وين كلمة "مهم"؟', 'أين كلمة "مهم"'),
        
        # With extra spaces
        ("وين   كلمة   مهم   ؟", "أين كلمة مهم"),
        
        # Diacritics removal
        ("أَيْنَ كَلِمَة مُهِمّ؟", "أين كلمة مهم"),
    ]
    
    passed = 0
    failed = 0
    
    for original, expected in test_cases:
        normalized = normalize_text(original)
        if normalized == expected:
            print(f"    ✅ PASS - '{original}' -> '{normalized}'")
            passed += 1
        else:
            print(f"    ❌ FAIL - '{original}' -> '{normalized}' (expected: '{expected}')")
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0


def test_smart_quotes():
    """Test smart quote stripping functionality."""
    print("\n🔤 Testing Smart Quote Stripping")
    print("=" * 50)
    
    test_cases = [
        # ASCII quotes
        ('"important"', 'important'),
        ("'important'", 'important'),
        
        # Smart quotes
        ('"important"', 'important'),
        ('"important"', 'important'),
        ('\u2018important\u2019', 'important'),
        ('\u201cimportant\u201d', 'important'),
        
        # Mixed quotes
        ('"important"', 'important'),
        ("'important'", 'important'),
        
        # No quotes
        ('important', 'important'),
        ('مهم', 'مهم'),
        
        # Empty/whitespace
        ('', ''),
        ('   ', '   '),
    ]
    
    passed = 0
    failed = 0
    
    for original, expected in test_cases:
        result = strip_smart_quotes(original)
        if result == expected:
            print(f"    ✅ PASS - '{original}' -> '{result}'")
            passed += 1
        else:
            print(f"    ❌ FAIL - '{original}' -> '{result}' (expected: '{expected}')")
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0


def test_term_cleaning():
    """Test term cleaning and validation."""
    print("\n🧹 Testing Term Cleaning")
    print("=" * 50)
    
    test_cases = [
        # Valid terms
        ("مهم", "مهم"),
        ("important", "important"),
        ("  مهم  ", "مهم"),
        ('"مهم"', "مهم"),
        ("'important'", "important"),
        ("  'مهم'  ", "مهم"),
        
        # Invalid terms (should return None)
        ("", None),  # Empty
        ("   ", None),  # Whitespace only
        ("أين", None),  # Stop term
        ("where", None),  # Stop term
        ("is", None),  # Stop term
        ("كلمة", None),  # Stop term
        ("a", None),  # Stop term
        ("the", None),  # Stop term
        
        # Too long (should return None)
        ("a" * 65, None),  # 65 characters
        ("مهم" * 25, None),  # Very long Arabic
        
        # Valid long terms
        ("a" * 64, "a" * 64),  # Exactly 64 characters
        ("مهم" * 20, "مهم" * 20),  # Long but valid Arabic
    ]
    
    passed = 0
    failed = 0
    
    for original, expected in test_cases:
        result = clean_extracted_term(original)
        if result == expected:
            status = "None" if expected is None else f"'{expected}'"
            print(f"    ✅ PASS - '{original}' -> {status}")
            passed += 1
        else:
            result_str = "None" if result is None else f"'{result}'"
            expected_str = "None" if expected is None else f"'{expected}'"
            print(f"    ❌ FAIL - '{original}' -> {result_str} (expected: {expected_str})")
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0


def test_where_is_term_patterns():
    """Test where_is_term intent detection patterns."""
    print("\n🎯 Testing where_is_term Patterns")
    print("=" * 50)
    
    # Test cases: (input_text, expected_intent, expected_args)
    test_cases = [
        # Arabic MSA
        ("أين كلمة مهم؟", "where_is_term", {"term": "مهم"}),
        ("أين مهم؟", "where_is_term", {"term": "مهم"}),
        ("أين كلمة 'مهم'؟", "where_is_term", {"term": "مهم"}),
        ('أين كلمة "مهم" ؟', "where_is_term", {"term": "مهم"}),
        ("أين كلمة   مهم   ", "where_is_term", {"term": "مهم"}),
        
        # Colloquial normalized → 'أين'
        ("وين كلمة مهم؟", "where_is_term", {"term": "مهم"}),
        ("فين كلمة مهم؟", "where_is_term", {"term": "مهم"}),
        ("وين مهم؟", "where_is_term", {"term": "مهم"}),
        ("فين مهم؟", "where_is_term", {"term": "مهم"}),
        
        # English
        ('where is "important"?', "where_is_term", {"term": "important"}),
        ("where is important", "where_is_term", {"term": "important"}),
        ("where is 'important'", "where_is_term", {"term": "important"}),
        ("where is   important   ", "where_is_term", {"term": "important"}),
        
        # Make sure we don't capture the question word
        ("أين؟", "unsupported", {}),
        ("where is", "unsupported", {}),
        ("أين", "unsupported", {}),
        ("where", "unsupported", {}),
        
        # Make sure headings query doesn't collide
        ("شو العناوين؟", "list_subtitles", {}),
        ("list headings", "list_subtitles", {}),
        ("ما العناوين؟", "list_subtitles", {}),
        ("اعرض العناوين", "list_subtitles", {}),
        
        # Bad terms (should be rejected)
        ("أين أين؟", "unsupported", {}),  # Stop term
        ("where is where", "unsupported", {}),  # Stop term
        ("أين كلمة؟", "unsupported", {}),  # Stop term
        ("where is is", "unsupported", {}),  # Stop term
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
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0


def test_no_regressions():
    """Test that other intents still work correctly."""
    print("\n🔄 Testing No Regressions")
    print("=" * 50)
    
    # Test cases for other intents
    test_cases = [
        # count_paragraphs
        ("كم عدد الفقرات؟", "count_paragraphs", {}),
        ("how many paragraphs", "count_paragraphs", {}),
        
        # read_paragraph
        ("اقرأ الفقرة الأولى", "read_paragraph", {"paragraph_index": 1}),
        ("read paragraph 2", "read_paragraph", {"paragraph_index": 2}),
        
        # summarize_paragraph
        ("لخص الفقرة الأولى", "summarize_paragraph", {"paragraph_index": 1, "style": "short"}),
        ("summarize paragraph 2", "summarize_paragraph", {"paragraph_index": 2, "style": "short"}),
        
        # has_bullets
        ("في بولت؟", "has_bullets", {}),
        ("has bullets?", "has_bullets", {}),
        
        # list_subtitles
        ("شو العناوين؟", "list_subtitles", {}),
        ("list headings", "list_subtitles", {}),
        
        # get_page_number
        ("كم رقم الصفحة؟", "get_page_number", {}),
        ("what page number", "get_page_number", {}),
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
        
        # Check args match
        args_match = True
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
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Run all tests."""
    print("🚀 Starting where_is_term Intent Hardening Tests")
    print("=" * 60)
    
    tests = [
        ("Text Normalization", test_text_normalization),
        ("Smart Quote Stripping", test_smart_quotes),
        ("Term Cleaning", test_term_cleaning),
        ("where_is_term Patterns", test_where_is_term_patterns),
        ("No Regressions", test_no_regressions),
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                total_passed += 1
            else:
                total_failed += 1
        except Exception as e:
            print(f"    ❌ ERROR in {test_name}: {str(e)}")
            total_failed += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 FINAL RESULTS: {total_passed} test suites passed, {total_failed} failed")
    
    if total_failed == 0:
        print("🎉 All tests passed! where_is_term intent hardening is working correctly.")
        return True
    else:
        print(f"⚠️  {total_failed} test suites failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
