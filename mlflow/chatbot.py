import os
import re
import mlflow.pyfunc

import torch
from transformers import AutoModelWithLMHead, AutoTokenizer

from config.modelconfig import Args

class Chatbot(mlflow.pyfunc.PythonModel):
    def __init__(self, model_path='microsoft/DialoGPT-medium'):
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        self.model = AutoModelWithLMHead.from_pretrained(model_path)
    
    def predict(self, context, model_input):
        user_input = str(model_input['text'])
        user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, 
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

if __name__ == '__main__':
    # run model training
    # save params, metrics, model artifact

    model = Chatbot(model_input='output-small') # use saved model artifact path to init
    mlflow.set_experiment('chatbot')

    with mlflow.start_run():
        # log training parameters
        args = Args()
        params = vars(Args)
        mlflow.log_params(params)

        # log evaluation metrics
        text_file = open("output-small/eval_results.txt", "r")
        text = text_file.read()
        text_file.close()

        perplexity_score = int(re.findall(r'\b\d+\b', text)[0])
        mlflow.log_metric({'perplexity': perplexity_score})

        # log model artifact
        mlflow.pyfunc.log_model('chatbot', python_model=model, conda_env=os.path.join('conda.yaml'))
        mlflow.end_run()