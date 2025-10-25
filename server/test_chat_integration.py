"""
Quick integration test for chat functionality.

Run this after server is started to verify chat system works.
"""

import asyncio
import httpx


async def test_chat_rest_api():
    """Test the REST API endpoint."""
    
    print("=" * 60)
    print("Testing Chat REST API")
    print("=" * 60)
    
    # Test data
    test_queries = [
        ("0003", "كم عدد الفقرات؟"),
        ("0003", "اقرأ الفقرة الأولى"),
        ("0003", "وين كلمة العنود؟"),
        ("0003", "how many paragraphs?"),
        ("0003", "ترجم للإنجليزية"),  # Unsupported
        ("9999", "اقرأ الفقرة الأولى"),  # Non-existent doc
    ]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for doc_id, query in test_queries:
            print(f"\n{'─' * 60}")
            print(f"Query: {query}")
            print(f"DocId: {doc_id}")
            print(f"{'─' * 60}")
            
            try:
                response = await client.post(
                    "http://localhost:8080/api/v1/chat",
                    json={
                        "docId": doc_id,
                        "text": query,
                        "locale": "ar"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✓ Success")
                    print(f"Intent: {result['meta']['intent']}")
                    print(f"Args: {result['meta']['args']}")
                    print(f"Response: {result['text'][:200]}...")
                else:
                    print(f"✗ Error {response.status_code}")
                    print(f"Detail: {response.json().get('detail', 'Unknown error')}")
                    
            except Exception as e:
                print(f"✗ Exception: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    print("Starting Chat Integration Test")
    print("Make sure the server is running on http://localhost:8080")
    print("")
    
    asyncio.run(test_chat_rest_api())

