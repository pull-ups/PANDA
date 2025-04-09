import torch
import numpy as np
import sys
import os
import pickle


from custom_transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, T5ForConditionalGeneration, T5Config, T5Tokenizer
from transformers import T5ForConditionalGeneration as T5ForConditionalGeneration_persona
from transformers import T5Config as T5Config_persona
from transformers import AutoTokenizer as AutoTokenizer_persona





DIRECTIONS = ['north', 'south', 'east', 'west', 'northwest', 'northeast', 'southwest', 'southeast', 'down', 'up', 'in', 'out']

class EthicsModel:
    def __init__(self, model, load_path, load_dir, persona, persona_reverse, oracle=False):
        self.device = torch.device("cuda")
        print("load_path: ", load_path)
        print("persona: ", persona)
        print("persona_reverse: ", persona_reverse)

        self.oracle = oracle  # True if using oracle to know whether action is immoral
        self.envs = None  # initialize envs attribute for oracle

        if 'cm' in load_path:   
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            config = AutoConfig.from_pretrained(model, num_labels=1)
            self.morality_mode = 'cm'
            self.cclm = AutoModelForSequenceClassification.from_pretrained(model, config=config)
            self.cclm.load_state_dict(torch.load(load_path), strict=False)
        elif 'util' in load_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model)

            config = AutoConfig.from_pretrained(model, num_labels=1)
            self.morality_mode = 'util'
            self.cclm = AutoModelForSequenceClassification.from_pretrained(model, config=config)
            self.cclm.load_state_dict(torch.load(load_path), strict=False)
        elif 'persona' in load_path:   
            print("persona mode")
            self.persona = persona
            self.persona_reverse = persona_reverse
        
            model_size="xl"
            if model_size=="xl":
                print("XL model") 
                config = T5Config_persona.from_pretrained(f'google/flan-t5-{model_size}')
                self.tokenizer = AutoTokenizer_persona.from_pretrained(f'google/flan-t5-{model_size}')
                self.cclm = T5ForConditionalGeneration_persona.from_pretrained(load_path, local_files_only=True, config=config)
                self.cclm.load_state_dict(torch.load(load_path))
            else:
                print("large model")
                config = T5Config.from_pretrained(f'google/flan-t5-{model_size}')
                self.tokenizer = AutoTokenizer.from_pretrained(f'google/flan-t5-{model_size}')
                self.cclm = T5ForConditionalGeneration.from_pretrained(load_path, local_files_only=True, config=config)
            
            self.morality_mode = 'persona'
        
       
        elif 't5' in load_path:

            config = T5Config.from_pretrained(load_path)
            self.tokenizer = T5Tokenizer.from_pretrained('t5-large', config=config)
            self.morality_mode = 't5'
            self.cclm = T5ForConditionalGeneration.from_pretrained(load_path, config=config)
        elif 'roberta-delphi' in load_path:
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            config = AutoConfig.from_pretrained(load_path)
            self.cclm = AutoModelForSequenceClassification.from_pretrained(load_path, config=config)
            self.morality_mode = 'roberta-delphi'
        

        else:
            raise NotImplementedError

        self.cclm = self.cclm.to(self.device)
        self.cclm.eval()
        imm_tokens = ['0', '▁0']
        self.immoral_act_label_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in imm_tokens]

        self.cache = {}

    def modelGenerate(self, input_text, lm, tokenizer):
        input_ids = tokenizer(input_text, max_length=256, truncation=True, padding = True, return_tensors='pt')['input_ids'].to(self.device)
        
        model_output = lm.generate(input_ids)
        model_answer = tokenizer.batch_decode(model_output, skip_special_tokens=True)

        return model_answer

    

    def save_cache(self, save_dir):
        for action in self.cache:
            self.cache[action] = self.cache[action].detach().cpu()
        with open(os.path.join(save_dir, 'ethics_model.pkl'), 'wb') as f:
            pickle.dump(self.cache, f)

    def load_cache(self, load_dir):
        cache_file = os.path.join(load_dir, 'ethics_model.pkl')
        if os.path.exists(cache_file):
            print('LOADING CONDITIONING MODEL CACHE')
            with open(cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            for action in self.cache:
                self.cache[action] = torch.tensor(self.cache[action], device=self.device)
        
    def get_ids_mask(self, sentences, max_length):
        tokenized = [self.tokenizer.tokenize(s) for s in sentences]
        tokenized = [t[-(max_length - 1):] + ['SEP'] for t in tokenized]

        ids = [self.tokenizer.convert_tokens_to_ids(t) for t in tokenized]
        ids = np.array([np.pad(i, (0, max_length - len(i)),
                            mode='constant') for i in ids])
        amasks = []
        for seq in ids:
            seq_mask = [float(i > 0) for i in seq]
            amasks.append(seq_mask)
        return ids, amasks

    def load_process_sentences(self, sentences, max_length=512):
        sentences = ["[CLS] " + s for s in sentences]
        ids, amasks = self.get_ids_mask(sentences, max_length)
        inputs = torch.tensor(ids, device=self.device)
        masks = torch.tensor(amasks, device=self.device)
        return inputs, masks

    @torch.no_grad()
    def get_probs(self, state_batch, act_batch, cond_threshold, testing_flag=False, custom_envs=None):
        # ACTION ONLY CONDITIONING
        act_sizes = [len(a) for a in act_batch]
        sentences = []
        sentences_raw_actions = []
        cond_probs = []
        for env_idx, actions in enumerate(act_batch):
            for action in actions:
                if self.oracle:
                    # in the oracle case, the output depends on the action and the env state, not just on
                    # the action string
                    cache_str = action + str(state_batch[env_idx].env_hash)
                else:
                    cache_str = action
                
                if cache_str in self.cache:
                    cond_probs.append(self.cache[cache_str])
                else:
                    cond_probs.append(cache_str)  # hacky way to remember what to add to the cache later
                    if testing_flag and self.oracle:
                        print('should not get here in update method; caching is imperfect')
                    sentences_raw_actions.append((action, env_idx))  # env_idx should be valid (e.g. < 8) when oracle is turned on
                    if action in DIRECTIONS:
                        action = 'go ' + action
                    if self.morality_mode == 't5' or self.morality_mode == 'roberta-delphi':
                        raw = state_batch[env_idx].raw_state
                        curr_act = f'I {action}.'
                        sentences.append(raw + curr_act)
                    else:
                        sentences.append(f'I {action}.')
        if sentences:
            if self.oracle:
                probs = []
                assert self.envs != None, 'must set cclm.envs before using oracle'
                for action, env_idx in sentences_raw_actions:
                    if custom_envs is not None:
                        env = custom_envs[env_idx]
                    else:
                        env = self.envs[env_idx]
                    saved_state = env.env.get_state()
                    _, _, _, info = env.env.step(action)
                    if (info['label_vectors'] != []) and (sum(info['label_vectors'])[0] > 0):
                        probs.append(1)
                    else:
                        probs.append(0)
                    env.env.set_state(saved_state)
                probs = torch.FloatTensor(probs).to(self.device)
            else:
                if self.morality_mode != 't5':                
                    if self.morality_mode == 'cm':
                        #print("in delphi, sentences: ", sentences)
                        #print("length of sentences: ", len(sentences))
                        input_ids, input_mask = self.load_process_sentences(sentences, 10)
                        with torch.no_grad():
                            output = self.cclm(input_ids, attention_mask=input_mask)[0]
                        probs = torch.sigmoid(output).squeeze(dim=1)
                        #print("in delphi, probs: ", probs)
                        
                    elif self.morality_mode == 'util':
                        input_ids, input_mask = self.load_process_sentences(sentences, 10)
                        with torch.no_grad():
                            output = self.cclm(input_ids, attention_mask=input_mask)[0]
                        probs = output.squeeze(dim=1)
                    elif self.morality_mode == 'roberta-delphi':
                        input_ids, input_mask = self.load_process_sentences(sentences, 512)
                        with torch.no_grad():
                            output = self.cclm(input_ids, attention_mask=input_mask)[0]
                        output = torch.softmax(output, dim=1)
                        probs = output[:, 0]
                    elif self.morality_mode == 'persona':
                        
                        input_text = [f"[Valence] Statement: {sentence} Trait: {self.persona}" for sentence in sentences]
                        #print("Person input_text: ", input_text)
                        
                        output_text = self.modelGenerate(input_text, self.cclm, self.tokenizer)
                        #print("Person output_text: ", output_text)
                        #if element of output_text is "Positive", then element of booleans is True, else False
                        booleans=[]
                        for i in range(len(output_text)):    
                            try:
                                #valence=int(output_text[i].split(":")[1].strip())
                                if "positive" == output_text[i].lower().strip():
                                    valence=1
                                elif "negative" == output_text[i].lower().strip():
                                    valence=-1
                                else:
                                    valence=0
                            except:
                                valence=0
                            
                            booleans.append(valence)
                        #print("in conditioning_model.py")
                        #print(output_text)
                        #print(booleans)
                    # elif self.morality_mode == 'persona':
                        
                    #     input_text = [f"[Valence] Statement: {sentence} Trait: {self.persona}" for sentence in sentences]
                    #     #print("Person input_text: ", input_text)
                        
                    #     #[Valence]: Statement: Not sharing your online banking password with a friend. Trait: Machiavellianism
                    #     output_text = self.modelGenerate(input_text, self.cclm, self.tokenizer)
                    #     #print("Person output_text: ", output_text)
                    #     #if element of output_text is "Positive", then element of booleans is True, else False
                    #     booleans=[]
                    #     for i in range(len(output_text)):    
                    #         try:
                    #             valence=int(output_text[i].split(":")[1].strip())
                    #         except:
                    #             valence=0
                            
                    #         booleans.append(valence)
                else:
                    input_ids, input_mask = self.load_process_sentences(sentences, 512)
                    with torch.no_grad():
                        output = self.cclm.generate(input_ids, max_length=10, output_scores=True, return_dict_in_generate=True)
                        #our = output.scores[0] #keisuke checkpoint
                        our = output.scores[4] #new model not keisuke checkpoint
                        our_soft = torch.softmax(our, dim=1)
                        probs = torch.zeros((our_soft.size(0))).to(self.device)
                        for label_id in self.immoral_act_label_ids:
                            probs += our_soft[:, label_id]

        counter = 0

        if self.morality_mode == 'persona':
            for i, item in enumerate(cond_probs):
                if type(item) == str:
                    prob = torch.tensor(booleans[counter]).to(self.device)
                    self.cache[item] = prob
                    cond_probs[i] = prob
                    counter += 1
        else:
            for i, item in enumerate(cond_probs):
                if type(item) == str:
                    prob = probs[counter]
                    self.cache[item] = prob
                    cond_probs[i] = prob
                    counter += 1

        
        cond_probs = torch.tensor(cond_probs, device=self.device)
        #print("in delphi, cond_probs: ", cond_probs)
        
        if self.morality_mode != 'persona':
            if cond_threshold:
                if self.morality_mode == 'cm':
                    cond_probs = cond_probs > cond_threshold
                elif self.morality_mode == 'util':
                    cond_probs = cond_probs < cond_threshold
                
                
            
        return cond_probs.split(act_sizes)
