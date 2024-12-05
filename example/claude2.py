from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os

anthropic_api_key = os.environ["An_Key"]

anthropic = Anthropic(
    api_key= anthropic_api_key,
)
completion = anthropic.completions.create(
    model="claude-2.1",
    max_tokens_to_sample=350,
    prompt=f"{HUMAN_PROMPT} 如何在一周内学会Python？{AI_PROMPT}",
)
print(completion.completion)