import requests, time
BASE = "http://localhost:1235/v1"
models = requests.get(f"{BASE}/models").json()["data"]
name = models[0]["id"]
print("model:", name)

for i in range(3):
    t = time.time()
    r = requests.post(f"{BASE}/chat/completions", json={
        "model": name,
        "messages": [{"role": "user", "content": "یک پاراگراف کوتاه درباره مدیریت دانش سازمانی بنویس."}],
        "max_tokens": 200, "temperature": 0}).json()
    dt = time.time() - t
    n = r["usage"]["completion_tokens"]
    print(f"run {i+1}: {n/dt:.1f} tok/s")
print(r["choices"][0]["message"]["content"]) # type: ignore