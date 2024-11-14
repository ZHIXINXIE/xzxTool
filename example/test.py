from xzxTool.chat import chat

chat = chat(model_name="llama2")
chat.config_system_prompt("You are a helpful assistant.")
print(chat.chat("Hello, how are you?"))