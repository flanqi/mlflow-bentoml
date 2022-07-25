"""Preprocess data into correct format"""
import json
import pandas as pd

def preprocess(input_path='data/message_1.json'):
    """Turns JSON messages into dialogues"""
    # load data into dict
    f = open(input_path)
    data = json.load(f)
    
    # Iterating through the json
    sender_names = []
    contents = []
    prev_name = ''
    prev_message = ''
    for m in reversed(data['messages']):
        try:
            name = m['sender_name']
            content = m['content']
            if name == prev_name: # same person, append prev conversation
                content = prev_message + content
                contents[-1] = content
            else: # move onto the next person
                sender_names.append(name)
                contents.append(content)
        except:
            pass

    # Closing file
    f.close()

    return sender_names, contents

def transform(contents):
    """Transforms messages into trainable format"""
    context_1 = contents[:-2]
    context_0 = contents[1:-1]
    response = contents[2:]

    context = pd.DataFrame({'response': response, 
                            'context/0': context_0, 
                            'context/1': context_1})
    return context

def prepare(input_path='data/message_1.json', output_path='data/contexts.csv'):
    """Orchestration function"""
    _, contents = preprocess(input_path)
    context = transform(contents)
    context.to_csv(output_path, index=False)