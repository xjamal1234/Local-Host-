#!/usr/bin/env python3
"""
WebSocket Chat + Lifecycle Events Test

Tests the new lifecycle events (processing_started, processing_completed)
and enveloped chat_user/chat_assistant message types over WebSocket.
"""

import asyncio
import websockets
import json


async def test_lifecycle_and_chat():
    """Test lifecycle events and chat functionality with enveloped format."""
    # NOTE: Add ?token=<your_token> if NOOR_CHAT_TOKEN is set
    uri = "ws://localhost:8080/ws/guidance"
    
    print("\n" + "=" * 70)
    print("WebSocket Lifecycle + Chat Test")
    print("=" * 70)
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print(f"✓ Connected\n")
        
        # Test 1: Send enveloped chat message for existing document
        print("-" * 70)
        print("Test 1: Chat with existing document (enveloped format)")
        print("-" * 70)
        
        test_message = {
            "type": "chat_user",
            "data": {
                "docId": "0003",
                "text": "كم عدد الفقرات؟"
            }
        }
        
        print(f"→ Sending: {json.dumps(test_message, ensure_ascii=False)}")
        await websocket.send(json.dumps(test_message))
        
        # Receive response (may get heartbeat first)
        try:
            response_text = await asyncio.wait_for(websocket.recv(), timeout=10)
            response = json.loads(response_text)
            
            # Skip heartbeat if received
            if response.get("type") == "hb":
                print(f"  (Received heartbeat, waiting for chat response...)")
                response_text = await asyncio.wait_for(websocket.recv(), timeout=10)
                response = json.loads(response_text)
            
            # Check for lifecycle events
            if response.get("type") == "processing_started":
                print(f"📡 Lifecycle: processing_started")
                print(f"   Data: {response.get('data')}")
                response_text = await asyncio.wait_for(websocket.recv(), timeout=30)
                response = json.loads(response_text)
            
            if response.get("type") == "processing_completed":
                print(f"✓ Lifecycle: processing_completed")
                data = response.get("data", {})
                print(f"   docId: {data.get('docId')}, gpt_ready: {data.get('gpt_ready')}")
                response_text = await asyncio.wait_for(websocket.recv(), timeout=10)
                response = json.loads(response_text)
            
            # Check chat response
            if response.get("type") == "chat_assistant":
                print(f"✓ Received enveloped chat_assistant response")
                data = response.get("data", {})
                text = data.get("text", "")
                meta = data.get("meta", {})
                print(f"   Text: {text[:100]}{'...' if len(text) > 100 else ''}")
                print(f"   Intent: {meta.get('intent')}, Args: {meta.get('args')}")
            else:
                print(f"⚠ Unexpected response type: {response.get('type')}")
                print(f"   Full response: {response}")
        
        except asyncio.TimeoutError:
            print("✗ Timeout waiting for response")
        
        # Test 2: Chat with non-existent document
        print("\n" + "-" * 70)
        print("Test 2: Chat with non-existent document")
        print("-" * 70)
        
        test_message = {
            "type": "chat_user",
            "data": {
                "docId": "9999",
                "text": "اقرأ الفقرة الأولى"
            }
        }
        
        print(f"→ Sending: {json.dumps(test_message, ensure_ascii=False)}")
        await websocket.send(json.dumps(test_message))
        
        try:
            response_text = await asyncio.wait_for(websocket.recv(), timeout=5)
            response = json.loads(response_text)
            
            # Skip heartbeat if received
            if response.get("type") == "hb":
                response_text = await asyncio.wait_for(websocket.recv(), timeout=5)
                response = json.loads(response_text)
            
            if response.get("type") == "chat_assistant":
                data = response.get("data", {})
                text = data.get("text", "")
                if "جارٍ المعالجة" in text or "processing" in text.lower():
                    print(f"✓ Got expected 'processing not ready' message")
                    print(f"   Text: {text}")
                else:
                    print(f"⚠ Unexpected text for non-existent doc")
                    print(f"   Text: {text}")
        
        except asyncio.TimeoutError:
            print("✗ Timeout waiting for response")
        
        print("\n" + "=" * 70)
        print("Test Complete!")
        print("=" * 70)


if __name__ == "__main__":
    print("Starting WebSocket Lifecycle + Chat Test")
    print("Make sure the server is running on ws://localhost:8080")
    print("And that document 0003 exists (gpt_0003.json)")
    
    asyncio.run(test_lifecycle_and_chat())

