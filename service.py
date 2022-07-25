import torch
from transformers import AutoModelWithLMHead, AutoTokenizer

import bentoml
from bentoml.io import Text, JSON

class ChatBotRunnable(bentoml.Runnable):
    SUPPORT_NVIDA_GPU = False
    SUPPORT_CPU_MULTI_THREADING = False

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        self.model = AutoModelWithLMHead.from_pretrained('microsoft/DialoGPT-medium')
    
    @bentoml.Runnable.method(batchable=False)
    def get_response(self, input_text):
        user_input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token, 
                                               return_tensors = 'pt')
        chat_history_ids = self.model.generate(
            user_input_ids, max_length=200,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )
        return self.tokenizer.decode(chat_history_ids[:, user_input_ids.shape[-1]:][0],
                                     skip_special_tokens=True)

my_runner = bentoml.Runner(ChatBotRunnable)
svc = bentoml.Service('chatbot', runners=[my_runner])

@svc.api(input=Text(), output=JSON())
def talk(input_text):
    response = my_runner.get_response.run(input_text)
    return {'reponse': response}
