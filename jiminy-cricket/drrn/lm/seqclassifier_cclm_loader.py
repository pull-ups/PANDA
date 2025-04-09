from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

import torch
import numpy as np

#model = 'roberta-base'
#config = AutoConfig.from_pretrained(model, num_labels=1)
#tokenizer = AutoTokenizer.from_pretrained(model)
#cclm = AutoModelForSequenceClassification.from_pretrained(model, config=config).cuda()

#model = 't5-jc'
model = '../../ethics/t5-large-tuned-jc-new'
config = T5Config.from_pretrained(model)
tokenizer = T5Tokenizer.from_pretrained('t5-large', config=config)
cclm = T5ForConditionalGeneration.from_pretrained(model)#.from_pretrained(model)


device = 'cuda:1'

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

imm_tokens_all = [['0', '▁0'], ['1', '▁1'], ['2', '▁2']]

for imm_tokens in imm_tokens_all:
    with torch.no_grad():
        #output = cclm(input_ids, attention_mask=input_mask)
        output = cclm.generate(input_ids, max_length=10, output_scores=True, return_dict_in_generate=True)
        our = output.scores[4]
        our_soft = torch.softmax(our, dim=1)
        # imm_tokens = ['0', '▁0']
        immoral_act_label_ids = [tokenizer.convert_tokens_to_ids(x) for x in imm_tokens]
        immorality_scores = torch.zeros((our_soft.size(0))).to(device)
        for label_id in immoral_act_label_ids:
            immorality_scores += our_soft[:, label_id]

    print(imm_tokens)
    print(immorality_scores)
    print()