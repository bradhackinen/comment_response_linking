from copy import deepcopy
import numpy as np
import pandas as pd
import re
from unidecode import unidecode
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer,RobertaModel,get_cosine_schedule_with_warmup
from transformers.utils import logging
from ast import literal_eval
import sqlite3

from regcomment_influence import config

# Hide tokenization length warnings
# (triggered before the sequence length is shortened in encode_headtail)
logging.set_verbosity(40)

db_connection = sqlite3.connect(config.frdocs_db_file)
frdocs_cursor = db_connection.cursor()

# Verify that the database is connected
assert frdocs_cursor.execute('SELECT COUNT(1) FROM text').fetchone()[0]


def add_footnotes(obs_df):

    obs_df = obs_df.copy()

    for frdoc_number in obs_df['frdoc_number'].unique():
        # Build map from paragraph-->footnotes
        footnotes_df = pd.read_sql_query('''
                        SELECT footnotes.text_id,text.text_id AS footnote_ids
                        FROM text_footnotes AS footnotes
                        INNER JOIN text
                        ON text.footnote == footnotes.footnote
                        WHERE (text.frdoc_number==?)
                        AND (footnotes.frdoc_number==?)
                        ''',db_connection,
                        params=(frdoc_number,frdoc_number))
        
        if len(footnotes_df):

            footnotes_df = (footnotes_df
                                .groupby('text_id')[['footnote_ids']]
                                .agg(list)
                                .reset_index())

            footnote_map = {p:f for p,f in footnotes_df[['text_id','footnote_ids']].values}

            # Add footnote paragraphs
            frdoc_obs = obs_df['frdoc_number']==frdoc_number
            obs_df.loc[frdoc_obs,'text_ids'] = (obs_df.loc[frdoc_obs,'text_ids']
                                                        .apply(lambda pars: 
                                                                sorted(
                                                                    set(pars) 
                                                                    | set([f for p in pars 
                                                                        for f in footnote_map.get(p,[])]))
                                                            ))

    return obs_df



def add_lead_in(obs_df,mode='all'):

    assert mode in ['all','list_only']
    
    obs_df = obs_df.copy()

    for frdoc_number in obs_df['frdoc_number'].unique():
        # Build map from paragraph-->lead-in paragraph
        lead_in_df = (pd.read_sql_query(f'''
                SELECT text_id,numbering,bullet
                FROM text
                WHERE (frdoc_number == ?)
                AND (reg == 0)
                AND (footnote IS NULL)
                AND (tag IN ("P","FP"))
                ORDER BY text_id
                ''',db_connection,
                params=(frdoc_number,))
            .assign(frdoc_number=frdoc_number))
        
        if mode =='all':
            lead_in_df['lead_in'] = lead_in_df['text_id'].shift(1)
        else:
            lead_in_df['lead_in'] = lead_in_df['paragraph'] 


        for c in ['numbering','bullet']:
            x = lead_in_df['text_id'].copy()
            x.loc[lead_in_df[c].notnull()] = np.nan
            x = x.fillna(method='ffill')

            lead_in_df.loc[lead_in_df[c].notnull(),'lead_in'] = x.loc[lead_in_df[c].notnull()]

        lead_in_df = lead_in_df.dropna(subset=['lead_in'])

        lead_in_map = {p:l for p,l in lead_in_df[['text_id','lead_in']].values.astype(int)}

        # Add lead in paragraphs
        frdoc_obs = obs_df['frdoc_number']==frdoc_number
        obs_df.loc[frdoc_obs,'text_ids'] = (obs_df.loc[frdoc_obs,'text_ids']
                                                    .apply(lambda pars: 
                                                            sorted(
                                                                set(pars) 
                                                                | set([lead_in_map[p] for p in pars 
                                                                       if p in lead_in_map]))
                                                           ))

    return obs_df


# Run a little unit test
# obs_df = (pd.read_sql_query('''
#                 SELECT DISTINCT frdoc_number,text_id,text
#                 FROM text
#                 WHERE (frdoc_number == ?)
#                 AND (reg == 0)
#                 AND (footnote IS NULL)
#                 AND (tag IN ("P","FP"))
#                 AND text_id <= 30
#                 ''',db_connection,
#                 params=('E9-12929',))
#             .assign(
#                 text_ids = lambda x: [[p] for p in x['text_id']]
#             ))

# # TODO: Change test paragraphs now that we are using text_id instead of paragraph number
# assert add_footnotes(obs_df).set_index('text_id').loc[14,'text_ids'] == [14,15]
# assert add_lead_in(obs_df).set_index('text_id').loc[18,'text_ids'] == [14,18]


def get_input_text(frdoc_number,text_ids,prompt=None,max_footnote_chars=200,
                   title=False,abstract=False,agencies=False,sep='</s></s>'):
    
    # Make sure text_ids is not a string or bytes
    # (This can happen accidently and quietely when loading response data)
    if isinstance(text_ids,str) | isinstance(text_ids,bytes):
        raise ValueError(f'Invalid input type for text_ids ({type(text_ids)})')


    chunks = []

    if prompt:
        chunks.append(prompt)

    if title:
        title = frdocs_cursor.execute('''
                            SELECT title
                            FROM frdocs
                            WHERE (frdoc_number == ?)
                            ''',(frdoc_number,)
                            ).fetchone()[0]
        chunks.append(f'Title: {title}')

    if abstract:
        abstract = frdocs_cursor.execute('''
                            SELECT abstract
                            FROM frdocs
                            WHERE (frdoc_number == ?)
                            ''',(frdoc_number,)
                            ).fetchone()[0]
        
        chunks.append(f'Abstract: {abstract}')
    
    if agencies:
        agencies = frdocs_cursor.execute('''
                            SELECT agency
                            FROM frdoc_agencies
                            WHERE (frdoc_number == ?)
                            ''',(frdoc_number,)
                            ).fetchall()
        chunks.append(f'Agencies: {", ".join(a[0] for a in agencies)}')

    
    text_ids = sorted(set(text_ids))


    par_df = pd.read_sql_query(f'''
                    SELECT header,footnote,numbering,bullet,text
                    FROM text
                    WHERE (frdoc_number==?)
                    AND (text_id IN  ({','.join(['?']*len(text_ids))}))
                    ORDER BY text_id
                    ''',db_connection,           
                    params=tuple([frdoc_number]+text_ids))

    par_df['new_header'] = par_df['header'] != par_df['header'].shift(1)

    for i,(header,footnote,numbering,bullet,text,new_header) in enumerate(par_df.values):
        
        if pd.notnull(footnote):
            if len(text) > max_footnote_chars:
                text = text[:max_footnote_chars] + '...'
                
            chunks.append(f'Footnote: {text}')
        else:
            if ((i==0) or new_header):
                chunks.append(header.replace('\n',' / '))

            if numbering:
                chunks.append(f'{numbering} {text}')

            elif bullet:
                chunks.append(f'{bullet} {text}')

            else:
                chunks.append(text)
    
    return sep.join(chunks)

# frdoc_number = '2010-17338'
# get_input_text(frdoc_number,[17,18])

# print(get_input_text('2020-28615',[71,72,73,74],sep='\n'))



def encode_headtail(string,tokenizer,max_length=512,midpoint=128,pad=True):


    if pd.isnull(string):
        string = ''

    tokenized = tokenizer.encode_plus(string)
    # bos_token_id = tokenizer.bos_token_id
    # sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    ids = tokenized['input_ids']
    mask = tokenized['attention_mask']

    if len(ids) > max_length:
        m0 = midpoint
        m1 = max_length - midpoint - 1
        tokenized['input_ids'] = ids[:m0] + [pad_token_id] + ids[-m1:]
        tokenized['attention_mask'] = [1]*max_length

    elif pad and (len(ids) < max_length):
        n_pad = (max_length - len(ids))
        tokenized['input_ids'] = ids + [pad_token_id]*n_pad
        tokenized['attention_mask'] = mask + [0]*n_pad

    return tokenized


class ParagraphData(Dataset):
    def __init__(self,obs_df,model):

        if model.lead_in:
            obs_df = add_lead_in(obs_df,model.lead_in)

        if model.footnotes:
            obs_df = add_footnotes(obs_df)

        self.frdoc_numbers = obs_df['frdoc_number'].values
        self.text_ids = obs_df['text_ids'].values
        
        if 'prompt' in obs_df.columns:
            self.prompts = obs_df['prompts'].values
        else:
            self.prompts = None
        
        self.model = model

    def __len__(self):
        return len(self.frdoc_numbers)

    def __getitem__(self,index):

        frdoc_number = self.frdoc_numbers[index]
        text_ids = self.text_ids[index]
        prompt = self.prompts[index] if self.prompts else None

        input_text = get_input_text(frdoc_number,text_ids,prompt)

        obs = self.model.preprocess_string(input_text)
        obs['input_text'] = input_text
        obs['index'] = torch.tensor(index)

        return obs


class LabeledParagraphData(Dataset):
    def __init__(self,obs_df,model):

        if model.lead_in:
            obs_df = add_lead_in(obs_df,model.lead_in)

        if model.footnotes:
            obs_df = add_footnotes(obs_df)

        self.frdoc_numbers = obs_df['frdoc_number'].values
        self.text_ids = obs_df['text_ids'].values

        if 'prompt' in obs_df.columns:
            self.prompts = obs_df['prompts'].values
        else:
            self.prompts = None

        self.label_ids = [model.label_id_map[label] for label in obs_df['label']]
        self.model = model

    def __len__(self):
        return len(self.frdoc_numbers)

    def __getitem__(self,index):

        frdoc_number = self.frdoc_numbers[index]
        text_ids = self.text_ids[index]
        prompt = self.prompts[index] if self.prompts else None

        input_text = get_input_text(frdoc_number,text_ids,prompt)

        obs = self.model.preprocess_string(input_text)
        obs['input_text'] = input_text
        obs['label_id'] = torch.tensor(self.label_ids[index])
        obs['index'] = torch.tensor(index)

        return obs


class ParagraphClassifier(nn.Module):
    def __init__(self,
                    labels,
                    model_class=RobertaModel,
                    model_name='roberta-base',
                    max_len=512,
                    midpoint=128,
                    device='cpu',
                    lead_in='all',
                    footnotes=True,
                    **unused_args):

        super().__init__()

        self.model_class = model_class
        self.model_name = model_name
        self.max_len = max_len
        self.midpoint = midpoint
        self.labels = labels
        self.label_id_map = {label:i for i,label in enumerate(labels)}
        self.lead_in = lead_in
        self.footnotes=footnotes

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.transformer = model_class.from_pretrained(model_name)
        except OSError:
            self.transformer = model_class.from_pretrained(model_name,from_tf=True)

        self.linear_out = torch.nn.Linear(self.transformer.config.hidden_size,len(labels),bias=True)

        self.to(device)

    def to(self,device):
        super().to(device)
        self.device = device

    def forward(self,batch):

        transformer_out = self.transformer(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            return_dict=True)

        return transformer_out['pooler_output']

    def preprocess_string(self,string):

        # Encode string to create input and attention mask
        encoded = encode_headtail(string,self.tokenizer,max_length=self.max_len,midpoint=self.midpoint,pad=True)

        return {
                'processed':string,
                'input_ids':torch.tensor(encoded['input_ids']).long(),
                'attention_mask':torch.tensor(encoded['attention_mask']),
                }

    def train_classifier(self,train_df,epochs=1,lr=2e-5,eps=1e-8,
                         weight_decay=0,batch_size=8,num_workers=1,
                         max_grad_norm=1,mixup=False,mixup_alpha=5,
                         progress_bar=True,grad_accumulation=1,
                         max_obs_per_class=None,
                         **unused_args):

        optimizer = torch.optim.AdamW(self.parameters(),lr=lr,eps=eps,weight_decay=weight_decay)

        n_steps = epochs*(len(train_df)//(batch_size*grad_accumulation) + 1)

        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(0.1*n_steps),
                                                    num_training_steps=n_steps
                                                    )

        self.history = []
        for epoch in range(epochs):
            if max_obs_per_class:
                epoch_train_df = (train_df
                                    .sample(frac=1)
                                    .groupby('label')
                                    .head(max_obs_per_class)
                                    .reset_index())
            else:
                epoch_train_df = train_df

            training_loader = DataLoader(
                                LabeledParagraphData(epoch_train_df,self),
                                batch_size=batch_size,
                                num_workers=num_workers,
                                drop_last=False,
                                shuffle=True
                                )

            with tqdm(total=len(epoch_train_df),desc=f'Training epoch {epoch}',disable=not progress_bar) as pbar:
                
                a = 0
                optimizer.zero_grad()                
                for batch_number,batch in enumerate(training_loader):

                    h = {'epoch':epoch,'batch':batch_number,'train_loss':0,'n_obs':0}

                    batch = {k:v.to(self.device) for k,v in batch.items() if torch.is_tensor(v)}
                    n_batch = batch['label_id'].shape[0]

                    self.train()
                    X = self(batch)

                    if mixup:
                        mixup = int(mixup)

                        Y = F.one_hot(batch['label_id'],num_classes=len(self.labels)).float()


                        M = F.softmax(mixup_alpha*torch.rand((mixup,n_batch),device=self.device),dim=1)
                        
                        X = M@X
                        Y = M@Y

                        logits = self.linear_out(X)
                        loss = F.cross_entropy(logits,Y,reduction='mean')

                    else:
                        logits = self.linear_out(X)
                        loss = F.cross_entropy(logits,batch['label_id'],reduction='mean')
                    
                    loss /= grad_accumulation

                    loss.backward()
                    a += 1

                    if a >= grad_accumulation:

                        torch.nn.utils.clip_grad_norm_(self.parameters(),max_norm=max_grad_norm)

                        optimizer.step()
                        scheduler.step()
                        
                        optimizer.zero_grad()
                        a = 0

                    h['n_obs'] = n_batch
                    h['lr'] = scheduler.get_last_lr()[0]
                    h['loss'] = loss.detach().cpu().item()
                    
                    self.history.append(h)

                    pbar.update(n_batch)

        return pd.DataFrame(self.history)

    @torch.no_grad()
    def test_classifier(self,test_df):

        predicted_df = self.predict(test_df)

        return score_predicted(predicted_df['predicted'],test_df['label'])


    @torch.no_grad()
    def predict(self,obs_df,batch_size=64,num_workers=1,progress_bar=True):

        input_loader = DataLoader(
                            ParagraphData(obs_df,self),
                            batch_size=batch_size,
                            drop_last=False,
                            num_workers=num_workers
                            )

        probs = []

        with tqdm(total=len(obs_df),desc='Predicting',disable=not progress_bar) as pbar:
            for batch in input_loader:

                batch = {k:v.to(self.device) for k,v in batch.items() if torch.is_tensor(v)}

                self.eval()
                X = self(batch)
                logits = self.linear_out(X)
                batch_probs = F.softmax(logits,dim=1)

                probs.append(batch_probs.to('cpu').numpy())

                pbar.update(batch['input_ids'].shape[0])

        probs = np.vstack(probs)

        predicted_df = pd.DataFrame(index=obs_df.index)

        predicted_df['predicted'] = [self.labels[i] for i in np.argmax(probs,axis=1)]
        predicted_df['predicted_prob'] = np.max(probs,axis=1)

        for i,label in enumerate(self.labels):
            predicted_df[f'{label}_prob'] = probs[:,i]

        return predicted_df


def score_predicted(predicted,gold,labels=None):
        
        if labels is None:
            labels = sorted(set(predicted) | set(gold))

        confusion_df = pd.DataFrame()
        for gold_label in labels:
            for predicted_label in labels:
                confusion_df.loc[gold_label,predicted_label] \
                    = ((gold==gold_label) & (predicted==predicted_label)).sum()

        class_scores = []
        for label in labels:
            tp = ((predicted == label) & (gold == label)).sum()
            fp = ((predicted == label) & (gold != label)).sum()
            tn = ((predicted != label) & (gold != label)).sum()
            fn = ((predicted != label) & (gold == label)).sum()

            tpr = tp/(tp + fn) if tp + fn else 0
            tnr = tn/(tn + fp) if tn + fp else 0
            ppv = tp/(tp + fp) if tp + fp else 0
            npv = tn/(tn + fn) if tn + fn else 0
            f1 = 2*ppv*tpr / (ppv + tpr) if ppv + tpr else 0

            class_scores.append({
                                'label':label,
                                'TP':tp,
                                'FP':fp,
                                'TN':tn,
                                'FN':fn,
                                'TPR':tpr,
                                'TNR':tnr,
                                'PPV':ppv,
                                'NPV':npv,
                                'F1':f1,
                                })
        
        class_scores_df = pd.DataFrame(class_scores)

        if class_scores_df['F1'].min() > 0:
            macro_F1 = np.exp(np.log(class_scores_df['F1']).mean())
        else:
            macro_F1 = 0

        scores = {
            'confusion_matrix':confusion_df,
            'class_scores':class_scores_df,
            'macro_F1':macro_F1
        }

        return scores



# Run some unit tests

