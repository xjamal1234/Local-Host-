"""NOOR Layout Schema v1 for GPT Structured Outputs."""

NOOR_LAYOUT_V1 = {
    "type": "object",
    "properties": {
        "docId": {"type": "string"},
        "schema_version": {"type": "string"},
        "source": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "ocr_json_path": {"type": "string"}
            },
            "required": ["image_path", "ocr_json_path"],
            "additionalProperties": False
        },
        "page": {
            "type": "object",
            "properties": {"number": {"type": ["integer", "null"]}},
            "required": ["number"],
            "additionalProperties": False
        },
        "title": {"type": ["string", "null"]},
        "subtitles": {"type": "array", "items": {"type": "string"}},
        "paragraphs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": [
                            "title", "subtitle", "heading", "subheading",
                            "paragraph", "list", "table", "figure",
                            "footer", "header", "page_number", "other"
                        ]
                    },
                    "bbox": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "minItems": 4, "maxItems": 4
                    },
                    "text": {"type": "string"},
                    "lines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "text": {"type": "string"},
                                "words": {"type": "array", "items": {"type": "string"}},
                                "list_marker": {"type": ["string", "null"]}
                            },
                            "required": ["id", "text", "words", "list_marker"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["id", "role", "bbox", "text", "lines"],
                "additionalProperties": False
            }
        },
        "metadata": {
            "type": "object",
            "properties": {
                "language": {"type": "string", "enum": ["auto", "en", "ar", "mixed"]},
                "has_lists": {"type": "boolean"},
                "has_tables": {"type": "boolean"},
                "has_figures": {"type": "boolean"}
            },
            "required": ["language", "has_lists", "has_tables", "has_figures"],
            "additionalProperties": False
        },
        "metrics": {
            "type": "object",
            "properties": {
                "paragraph_count": {"type": "integer"},
                "line_count": {"type": "integer"},
                "word_count": {"type": "integer"}
            },
            "required": ["paragraph_count", "line_count", "word_count"],
            "additionalProperties": False
        },
        "notes": {"type": "array", "items": {"type": "string"}}
    },
    "required": [
        "docId", "schema_version", "source", "page", "title", "subtitles",
        "paragraphs", "metadata", "metrics", "notes"
    ],
    "additionalProperties": False
}


