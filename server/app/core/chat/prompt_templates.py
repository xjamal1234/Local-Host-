"""
Prompt templates for intent extraction with few-shot examples.
Supports Arabic (MSA/colloquial) and English queries.
"""

# System prompt for intent extraction
INTENT_EXTRACTION_SYSTEM_PROMPT = """أنت مستخرج نوايا (intent extractor) للأوامر المسموحة فقط.

الأوامر المدعومة:
- count_paragraphs: عد الفقرات
- has_bullets: التحقق من وجود نقاط
- has_headings: التحقق من وجود عناوين
- get_main_title: الحصول على العنوان الرئيسي
- get_page_number: الحصول على رقم الصفحة
- paragraph_word_count: عد كلمات فقرة محددة
- read_paragraph: قراءة فقرة محددة
- summarize_paragraph: تلخيص فقرة محددة
- read_line: قراءة سطر محدد من فقرة
- where_is_term: البحث عن مصطلح في المستند
- list_subtitles: سرد العناوين الفرعية
- list_bullets_in_paragraph: سرد النقاط في فقرة محددة
- unsupported: أي طلب خارج الأوامر المسموحة

قواعد مهمة:
- اربط التراكيب المكافئة دلالياً بنفس النية؛ تجاهل التشكيل والعلامات والمسافات الزائدة
- عالج الأرقام العربية والهندية بالتساوي
- إذا كان الطلب خارج الأوامر المسموحة، أرجع intent="unsupported" مع النص الأصلي

يجب أن يطابق الإخراج JSON Schema المقدم بدقة تامة.
"""

# Few-shot examples covering Arabic MSA, colloquial, and English
FEW_SHOT_EXAMPLES = [
    # count_paragraphs - comprehensive variations
    {
        "user": "كم عدد الفقرات؟",
        "assistant": '{"intent": "count_paragraphs", "args": {}}'
    },
    {
        "user": "كم فقرة؟",
        "assistant": '{"intent": "count_paragraphs", "args": {}}'
    },
    {
        "user": "قديش في فقرات؟",
        "assistant": '{"intent": "count_paragraphs", "args": {}}'
    },
    {
        "user": "قديش عدد الفقرات بالنص؟",
        "assistant": '{"intent": "count_paragraphs", "args": {}}'
    },
    {
        "user": "how many paragraphs?",
        "assistant": '{"intent": "count_paragraphs", "args": {}}'
    },
    {
        "user": "number of paragraphs",
        "assistant": '{"intent": "count_paragraphs", "args": {}}'
    },
    {
        "user": "كم فقرة موجودة",
        "assistant": '{"intent": "count_paragraphs", "args": {}}'
    },
    
    # has_bullets - comprehensive variations
    {
        "user": "هل فيه نقاط؟",
        "assistant": '{"intent": "has_bullets", "args": {}}'
    },
    {
        "user": "فيه نقاط مهمة؟",
        "assistant": '{"intent": "has_bullets", "args": {}}'
    },
    {
        "user": "في بولِت؟",
        "assistant": '{"intent": "has_bullets", "args": {}}'
    },
    {
        "user": "are there any bullet points?",
        "assistant": '{"intent": "has_bullets", "args": {}}'
    },
    {
        "user": "bullets?",
        "assistant": '{"intent": "has_bullets", "args": {}}'
    },
    {
        "user": "فيه قائمة نقاط",
        "assistant": '{"intent": "has_bullets", "args": {}}'
    },
    
    # has_headings
    {
        "user": "هل يوجد عناوين؟",
        "assistant": '{"intent": "has_headings", "args": {}}'
    },
    {
        "user": "does it have headings?",
        "assistant": '{"intent": "has_headings", "args": {}}'
    },
    {
        "user": "في عناوين فرعية",
        "assistant": '{"intent": "has_headings", "args": {}}'
    },
    
    # get_main_title
    {
        "user": "ما هو العنوان الرئيسي؟",
        "assistant": '{"intent": "get_main_title", "args": {}}'
    },
    {
        "user": "what's the main title?",
        "assistant": '{"intent": "get_main_title", "args": {}}'
    },
    {
        "user": "شو العنوان",
        "assistant": '{"intent": "get_main_title", "args": {}}'
    },
    
    # get_page_number
    {
        "user": "ما رقم الصفحة؟",
        "assistant": '{"intent": "get_page_number", "args": {}}'
    },
    {
        "user": "what page is this?",
        "assistant": '{"intent": "get_page_number", "args": {}}'
    },
    {
        "user": "كم رقم الصفحة",
        "assistant": '{"intent": "get_page_number", "args": {}}'
    },
    
    # paragraph_word_count
    {
        "user": "كم عدد الكلمات في الفقرة الأولى؟",
        "assistant": '{"intent": "paragraph_word_count", "args": {"paragraph_index": 1}}'
    },
    {
        "user": "how many words in paragraph 2?",
        "assistant": '{"intent": "paragraph_word_count", "args": {"paragraph_index": 2}}'
    },
    {
        "user": "كم كلمة بالفقرة الثالثة",
        "assistant": '{"intent": "paragraph_word_count", "args": {"paragraph_index": 3}}'
    },
    
    # read_paragraph
    {
        "user": "اقرأ الفقرة الأولى",
        "assistant": '{"intent": "read_paragraph", "args": {"paragraph_index": 1}}'
    },
    {
        "user": "read paragraph 2",
        "assistant": '{"intent": "read_paragraph", "args": {"paragraph_index": 2}}'
    },
    {
        "user": "اقرألي الفقرة الثانية",
        "assistant": '{"intent": "read_paragraph", "args": {"paragraph_index": 2}}'
    },
    {
        "user": "وش مكتوب في الفقرة الثالثة",
        "assistant": '{"intent": "read_paragraph", "args": {"paragraph_index": 3}}'
    },
    
    # summarize_paragraph
    {
        "user": "لخص الفقرة الأولى",
        "assistant": '{"intent": "summarize_paragraph", "args": {"paragraph_index": 1, "style": "short"}}'
    },
    {
        "user": "summarize paragraph 2 in detail",
        "assistant": '{"intent": "summarize_paragraph", "args": {"paragraph_index": 2, "style": "detailed"}}'
    },
    {
        "user": "لخص الفقرة الثانية بالتفصيل",
        "assistant": '{"intent": "summarize_paragraph", "args": {"paragraph_index": 2, "style": "detailed"}}'
    },
    {
        "user": "خلاصة الفقرة الأولى",
        "assistant": '{"intent": "summarize_paragraph", "args": {"paragraph_index": 1, "style": "short"}}'
    },
    
    # read_line
    {
        "user": "اقرأ السطر الأول من الفقرة الثانية",
        "assistant": '{"intent": "read_line", "args": {"paragraph_index": 2, "line_index": 1}}'
    },
    {
        "user": "read line 3 from paragraph 1",
        "assistant": '{"intent": "read_line", "args": {"paragraph_index": 1, "line_index": 3}}'
    },
    {
        "user": "السطر الثاني من الفقرة الثالثة",
        "assistant": '{"intent": "read_line", "args": {"paragraph_index": 3, "line_index": 2}}'
    },
    
    # where_is_term - comprehensive variations
    {
        "user": "وين كلمة 'مهم'؟",
        "assistant": '{"intent": "where_is_term", "args": {"term": "مهم"}}'
    },
    {
        "user": "أين كلمة 'مهم'؟",
        "assistant": '{"intent": "where_is_term", "args": {"term": "مهم"}}'
    },
    {
        "user": "فين كلمة 'مهم'؟",
        "assistant": '{"intent": "where_is_term", "args": {"term": "مهم"}}'
    },
    {
        "user": "where is the word 'important'?",
        "assistant": '{"intent": "where_is_term", "args": {"term": "important"}}'
    },
    {
        "user": "find 'important'",
        "assistant": '{"intent": "where_is_term", "args": {"term": "important"}}'
    },
    {
        "user": "ابحث عن كلمة العنود",
        "assistant": '{"intent": "where_is_term", "args": {"term": "العنود"}}'
    },
    {
        "user": "فين كلمة نظام",
        "assistant": '{"intent": "where_is_term", "args": {"term": "نظام"}}'
    },
    
    # list_subtitles
    {
        "user": "اعرض العناوين الفرعية",
        "assistant": '{"intent": "list_subtitles", "args": {}}'
    },
    {
        "user": "list all subtitles",
        "assistant": '{"intent": "list_subtitles", "args": {}}'
    },
    {
        "user": "شو العناوين الموجودة",
        "assistant": '{"intent": "list_subtitles", "args": {}}'
    },
    
    # list_bullets_in_paragraph
    {
        "user": "ما هي النقاط في الفقرة الأولى؟",
        "assistant": '{"intent": "list_bullets_in_paragraph", "args": {"paragraph_index": 1}}'
    },
    {
        "user": "list bullets in paragraph 2",
        "assistant": '{"intent": "list_bullets_in_paragraph", "args": {"paragraph_index": 2}}'
    },
    {
        "user": "عدد النقاط بالفقرة الثانية",
        "assistant": '{"intent": "list_bullets_in_paragraph", "args": {"paragraph_index": 2}}'
    },
    
    # unsupported examples
    {
        "user": "ترجم هذا للإنجليزية",
        "assistant": '{"intent": "unsupported", "args": {"original_query": "ترجم هذا للإنجليزية"}}'
    },
    {
        "user": "solve this math problem",
        "assistant": '{"intent": "unsupported", "args": {"original_query": "solve this math problem"}}'
    },
    {
        "user": "اكتب لي بحث عن الذكاء الاصطناعي",
        "assistant": '{"intent": "unsupported", "args": {"original_query": "اكتب لي بحث عن الذكاء الاصطناعي"}}'
    },
]


def build_few_shot_messages(user_query: str, locale: str = "ar") -> list:
    """
    Build few-shot messages for intent extraction.
    
    Args:
        user_query: The actual user query to extract intent from
        locale: Locale hint (currently unused, keeping for future)
        
    Returns:
        List of messages in OpenAI chat format
    """
    messages = [
        {"role": "system", "content": INTENT_EXTRACTION_SYSTEM_PROMPT}
    ]
    
    # Add few-shot examples (use a subset to avoid token limits)
    # Select examples intelligently based on query characteristics
    for example in FEW_SHOT_EXAMPLES[:15]:  # Limit to avoid excessive tokens
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})
    
    # Add actual user query
    messages.append({"role": "user", "content": user_query})
    
    return messages

