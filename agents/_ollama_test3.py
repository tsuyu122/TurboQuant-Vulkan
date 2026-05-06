import urllib.request, urllib.error, json

def test(name, url, body):
    print('---', name, url)
    req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read().decode('utf-8')
            print('STATUS', resp.status)
            print(data)
    except urllib.error.HTTPError as e:
        print('HTTPError', e.code)
        print(e.read().decode('utf-8', errors='replace'))
    except Exception as e:
        print('ERROR', type(e).__name__, e)

models = ['gemma4:e4b', 'gemma4:e4b-it-q4_K_M']
urls = ['http://127.0.0.1:11434/api/chat', 'http://127.0.0.1:11434/api/completions']
for model in models:
    for url in urls:
        print('Testing model', model)
        body = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Say hello.'}
            ]
        }
        test(f'{model} chat', 'http://127.0.0.1:11434/api/chat', body)
        body2 = {
            'model': model,
            'prompt': 'Say hello.',
            'max_tokens': 64,
            'temperature': 0.3
        }
        test(f'{model} completions', 'http://127.0.0.1:11434/api/completions', body2)
