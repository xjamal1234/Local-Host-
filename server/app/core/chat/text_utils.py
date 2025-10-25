"""
Text normalization utilities for Arabic and English text processing.

This module provides functions to normalize text for intent detection,
handling diacritics, digits, punctuation, and colloquial variations.
"""

import re
import unicodedata
from typing import Dict


# Arabic diacritics to remove
ARABIC_DIACRITICS = [
    '\u064B',  # Fathatan
    '\u064C',  # Dammatan  
    '\u064D',  # Kasratan
    '\u064E',  # Fatha
    '\u064F',  # Damma
    '\u0650',  # Kasra
    '\u0651',  # Shadda
    '\u0652',  # Sukun
    '\u0653',  # Maddah above
    '\u0654',  # Hamza above
    '\u0655',  # Hamza below
    '\u0656',  # Subscript alef
    '\u0657',  # Inverted damma
    '\u0658',  # Mark noon ghunna
    '\u0659',  # Zwarakay
    '\u065A',  # Vowel sign small v above
    '\u065B',  # Vowel sign inverted small v above
    '\u065C',  # Vowel sign dot below
    '\u065D',  # Reversed damma
    '\u065E',  # Fatha with two dots
    '\u065F',  # Wavy hamza below
    '\u0670',  # Superscript alef
]

# Arabic digits to Western digits mapping
ARABIC_TO_WESTERN_DIGITS = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
}

# Common colloquial token mappings
COLLOQUIAL_MAPPINGS = {
    # Arabic colloquial variations
    'قديش': 'كم',
    'وين': 'أين',
    'فين': 'أين',
    'إيش': 'ما',
    'شو': 'ما',
    'هيك': 'هكذا',
    'هيكي': 'هكذا',
    'مش': 'ليس',
    'مو': 'ليس',
    'ما': 'ليس',
    'عندي': 'لدي',
    'عندك': 'لديك',
    'عنده': 'لديه',
    'عندها': 'لديها',
    'عندنا': 'لدينا',
    'عندكم': 'لديكم',
    'عندهم': 'لديهم',
    
    # English variations
    'how many': 'number of',
    'how much': 'number of',
    'what is': 'what',
    'where is': 'where',
    'where are': 'where',
}


def normalize_text(text: str) -> str:
    """
    Normalize text for intent detection by:
    1. Removing diacritics and tatweel
    2. Converting Arabic digits to Western digits
    3. Normalizing hamza variants
    4. Converting ة to ه at word end
    5. Lowercasing and collapsing whitespace
    6. Removing punctuation
    7. Applying colloquial mappings
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Step 1: Remove diacritics and tatweel (ـ)
    normalized = text
    for diacritic in ARABIC_DIACRITICS:
        normalized = normalized.replace(diacritic, '')
    normalized = normalized.replace('ـ', '')  # Remove tatweel
    
    # Step 2: Convert Arabic digits to Western digits
    for arabic_digit, western_digit in ARABIC_TO_WESTERN_DIGITS.items():
        normalized = normalized.replace(arabic_digit, western_digit)
    
    # Step 3: Normalize hamza variants to ا (but preserve أين specifically)
    # First, protect أين from hamza normalization using a unique placeholder
    normalized = normalized.replace('أين', '___AYN_PROTECTED___')
    normalized = normalized.replace('إين', '___AYN_PROTECTED___')
    
    # Then normalize other hamza variants
    hamza_variants = ['أ', 'إ', 'آ', 'ؤ', 'ئ', 'ء']
    for hamza in hamza_variants:
        normalized = normalized.replace(hamza, 'ا')
    
    # Step 4: Keep ة as ة (don't convert to ه)
    # normalized = re.sub(r'ة\b', 'ه', normalized)  # Commented out
    
    # Step 5: Lowercase and normalize Unicode
    normalized = normalized.lower()
    normalized = unicodedata.normalize('NFKC', normalized)
    
    # Restore protected أين (after lowercase)
    normalized = normalized.replace('___ayn_protected___', 'أين')
    
    # Step 6: Remove punctuation and collapse whitespace
    normalized = re.sub(r'[^\w\s]', ' ', normalized)  # Remove punctuation
    normalized = re.sub(r'\s+', ' ', normalized)  # Collapse whitespace
    normalized = normalized.strip()
    
    # Step 7: Apply colloquial mappings
    words = normalized.split()
    normalized_words = []
    for word in words:
        if word in COLLOQUIAL_MAPPINGS:
            normalized_words.append(COLLOQUIAL_MAPPINGS[word])
        else:
            normalized_words.append(word)
    
    return ' '.join(normalized_words)


def extract_numbers(text: str) -> list:
    """
    Extract all numbers from text (both Arabic and Western digits).
    
    Args:
        text: Input text
        
    Returns:
        List of extracted numbers as strings
    """
    # First normalize the text
    normalized = normalize_text(text)
    
    # Find all sequences of digits
    numbers = re.findall(r'\d+', normalized)
    return numbers


def is_arabic_text(text: str) -> bool:
    """
    Check if text contains Arabic characters.
    
    Args:
        text: Input text
        
    Returns:
        True if text contains Arabic characters
    """
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    return bool(arabic_pattern.search(text))


def is_english_text(text: str) -> bool:
    """
    Check if text contains English characters.
    
    Args:
        text: Input text
        
    Returns:
        True if text contains English characters
    """
    english_pattern = re.compile(r'[a-zA-Z]')
    return bool(english_pattern.search(text))


def get_language_hint(text: str) -> str:
    """
    Determine the primary language of the text.
    
    Args:
        text: Input text
        
    Returns:
        'ar' for Arabic, 'en' for English, 'mixed' for both
    """
    has_arabic = is_arabic_text(text)
    has_english = is_english_text(text)
    
    if has_arabic and has_english:
        return 'mixed'
    elif has_arabic:
        return 'ar'
    elif has_english:
        return 'en'
    else:
        return 'unknown'


def strip_smart_quotes(s: str) -> str:
    """
    Remove ASCII quotes and smart quotes around a term.
    
    Args:
        s: Input string
        
    Returns:
        String with quotes removed
    """
    if not s:
        return s
    
    # Remove quotes from both ends
    s = s.strip()
    
    # ASCII quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    
    # Smart quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]
    if (s.startswith('\u2018') and s.endswith('\u2019')) or (s.startswith('\u201c') and s.endswith('\u201d')):
        s = s[1:-1]
    
    return s.strip()


def clean_extracted_term(s: str) -> str:
    """
    Clean and validate an extracted search term.
    
    Args:
        s: Raw extracted term
        
    Returns:
        Cleaned term or None if invalid
    """
    if not s:
        return None
    
    # Trim and strip quotes
    cleaned = s.strip()
    cleaned = strip_smart_quotes(cleaned)
    
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Reject if too short or too long
    if len(cleaned) < 1 or len(cleaned) > 64:
        return None
    
    # Stop terms that shouldn't be search terms
    STOP_TERMS = ['اين', 'أين', 'where', 'is', 'كلمة', 'term', 'the', 'a', 'an']
    
    if cleaned.lower() in STOP_TERMS:
        return None
    
    return cleaned
