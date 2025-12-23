import httpx
import json
import os
import base64
from dotenv import load_dotenv

load_dotenv()

async def test_docling():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Document</title></head>
    <body>
        <h1>Hello World</h1>
        <p>This is a test HTML document.</p>
    </body>
    </html>
    """
    
    endpoint = os.environ["DOCLING_URL"].rstrip("/") + "/v1/convert/source"
    headers = {
        "X-Api-Key": os.environ["DOCLING_API_KEY"],
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    
    # v1 API format - kind: "file" for base64
    payload = {
        "sources": [{
            "kind": "file",
            "base64_string": base64.b64encode(html_content.encode()).decode('utf-8'),
            "filename": "test.html"
        }],
        "options": {
            "to_formats": ["md"],
            "do_ocr": False,
        }
    }
    
    async with httpx.AsyncClient() as client:
        r = await client.post(endpoint, headers=headers, json=payload, timeout=60)
        
        if r.status_code != 200:
            print(f"Status: {r.status_code}")
            print(f"Response: {r.text}")
            r.raise_for_status()
            
        data = r.json()
    
    print("=== FULL RESPONSE ===")
    print(json.dumps(data, indent=2))
    
    # Check for metadata/format detection
    if "documents" in data:
        for doc in data["documents"]:
            print(f"\n=== DOCUMENT: {doc.get('filename', 'unknown')} ===")
            if "origin" in doc:
                print("Origin metadata:")
                print(json.dumps(doc["origin"], indent=2))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_docling())