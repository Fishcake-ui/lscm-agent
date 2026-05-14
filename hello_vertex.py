from anthropic import AnthropicVertex

client = AnthropicVertex(region="global", project_id="poised-beach-467216-k1")
message = client.messages.create(
 max_tokens=1024,
 messages=[{"role": "user", "content": "Hello! Can you help me?"}],
 model="claude-haiku-4-5@20251001"
)
print(message.content[0].text)