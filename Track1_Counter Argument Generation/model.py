from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config
import torch
import torch.nn as nn

class cag_gpt_model(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

    def from_pretrained(self, path):
        self.model = GPT2LMHeadModel.from_pretrained(path).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def forward(self, topic, source, target):
        input_prompt = f'Rebute the argument, {{ {topic} }}, {{ {source} }}, {{ {target} }}'
        input_target = f'{{ {target} }}' 
        encoded_prompt = self.tokenizer(input_prompt, return_tensors = 'pt').to(self.device)
        encoded_target = self.tokenizer.encode(input_target)
        labels_input = torch.LongTensor([-100]*(len(encoded_prompt['input_ids'].view(-1)) - len(encoded_target)) + encoded_target).to(self.device)
        output = self.model(**encoded_prompt, labels = labels_input)
        return output.loss
   
    def generator(self, topic, source, num_beams, max_length, eos_token_id = 92):
        prompt = f'Rebute the argument, {{ {topic} }}, {{ {source} }},'
        input_ids = self.tokenizer(prompt, return_tensors = 'pt')['input_ids'].to(self.device)
        output = self.model.generate(input_ids, num_beams = num_beams, max_length = max_length, eos_token_id = eos_token_id)
        result = self.tokenizer.decode(output.view(-1), skip_special_tokens = True)
        return result[len(prompt):]


