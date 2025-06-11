from together import Together
base_url="https://api.together.xyz/v1"
api_key="your-key"

client = Together(api_key=api_key, base_url=base_url)
response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B-fp8",
    messages=[{"role": "user", "content": "What are some fun things to do in New York?"}]
)
print(response.choices[0].message.content)
