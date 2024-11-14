from transformers import AutoModelForCausalLM, AutoTokenizer
from xzxTool.config import model_path_dict
class chat:
    def __init__(self, model_name):
        model_path = model_path_dict[model_name]
        self.model, self.tokenizer = self.init_model(model_path)

    def init_model(self,model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

    def config_system_prompt(self,system_prompt):
        self.system_prompt = system_prompt

    def chat(self,message,format=True):
        if format:
            prompt = "<s>[INST] <<SYS>>{{ %s }}<</SYS>>{{ %s }} [/INST]"%(self.system_prompt,message)
        else:
            prompt = message
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=1024)
        outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = outputs[len(prompt)-1:]
        return response

def chat_model(model,tokenizer,prompt):
    # without setting system prompt, the prompt should contain the system prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024)
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = outputs[len(prompt)-1:]
    return response