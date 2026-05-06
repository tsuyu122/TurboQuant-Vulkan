import urllib.request, urllib.error, json
url = 'http://127.0.0.1:11434/api/chat'
body = json.dumps({
    'model': 'gemma4:e4b',
    'messages': [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Say hello.'}
    ],
    'stream': False,
    'options': {'num_predict': 64, 'temperature': 0.3}
}).encode('utf-8')
req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read().decode('utf-8')
        print('STATUS', resp.status)
        print(data)
except urllib.error.HTTPError as e:
    print('HTTPError', e.code)
    print(e.read().decode('utf-8', errors='replace'))
except Exception as e:
    print('ERROR', type(e).__name__, e)
