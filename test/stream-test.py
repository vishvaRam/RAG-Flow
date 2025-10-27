import requests
import json

url = "http://localhost:4545/chat"
payload = {
    "messages": [{"role": "user", "content": "Newtons laws of motion"}],
    "stream": True,
    "temperature": 0.01
}

with requests.post(url, json=payload, stream=True) as r:
    buffer = ""
    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            buffer += chunk
            
            # Try to parse complete JSON objects from buffer
            while buffer:
                try:
                    # Find the end of a JSON object
                    obj, idx = json.JSONDecoder().raw_decode(buffer)
                    buffer = buffer[idx:].lstrip()
                    
                    # Handle different message types
                    if 'content' in obj:
                        print(obj['content'], end='', flush=True)
                    elif 'sources' in obj:
                        print("\n\n📚 Sources:")
                        for src in obj['sources']:
                            print(f"  - {src['subject']} > {src['topic']} ({src['chunk_id']})")
                    elif 'done' in obj:
                        print("\n✅ Stream complete")
                        break
                    elif 'error' in obj:
                        print(f"\n❌ Error: {obj['error']}")
                        break
                except json.JSONDecodeError:
                    # Incomplete JSON, wait for more data
                    break
