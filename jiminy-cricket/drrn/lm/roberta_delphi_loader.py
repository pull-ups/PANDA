from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import numpy as np


model = '/home/raja/jiminy-cricket/examples/experiments/calm-textgame/ethics/roberta-delphi'

tokenizer = AutoTokenizer.from_pretrained(model)
config = AutoConfig.from_pretrained(model)#, num_labels=3)
cclm = AutoModelForSequenceClassification.from_pretrained(model, config=config)
#cclm.load_state_dict(torch.load(model + '/pytorch_model.bin'), strict=False)

device = 'cuda'

cclm.to(torch.device(device))

def get_ids_mask(sentences, max_length):
    tokenized = [tokenizer.tokenize(s) for s in sentences]
    tokenized = [t[-(max_length - 1):] + ['SEP'] for t in tokenized]

    ids = [tokenizer.convert_tokens_to_ids(t) for t in tokenized]
    ids = np.array([np.pad(i, (0, max_length - len(i)),
                           mode='constant') for i in ids])
    amasks = []
    for seq in ids:
        seq_mask = [float(i > 0) for i in seq]
        amasks.append(seq_mask)
    return ids, amasks


def load_process_sentences(sentences, max_length=512):
    sentences = ["[CLS] " + s for s in sentences]
    ids, amasks = get_ids_mask(sentences, max_length)
    inputs = torch.tensor(ids, device=device)
    masks = torch.tensor(amasks, device=device)
    return inputs, masks



sentence = 'You see a dragon in front of you. Do you try to kill it? I hit the dragon with my sword.'

sentences = [sentence] * 4

input_ids, input_mask = load_process_sentences(sentences, 512)

with torch.no_grad():
    output = cclm(input_ids, attention_mask=input_mask)[0]
    output = torch.softmax(output, dim=1)
    scores = output[:, 0]

print()
