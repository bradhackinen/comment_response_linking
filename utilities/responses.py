from pathlib import Path
import pandas as pd
import re
import sqlite3
import numpy as np
from ast import literal_eval

from regcomment_influence import config


raw_dir = Path(config.data_dir)/'raw_data'/'annotated_rule_paragraphs'

db = sqlite3.connect(config.frdocs_db_file)



def get_sample_df(frdoc_number,sample_text_ids=None,sample_context_size=10):

    if sample_text_ids:

        start = min(sample_text_ids)
        end = max(sample_text_ids)

        sample_df = (pd.read_sql_query(f'''
                        SELECT text_id,footnote,text
                        FROM text
                        WHERE (frdoc_number == ?)
                        AND (text_id >= ?)
                        AND (text_id < ?)
                        AND (reg == 0)
                        AND (tag IN ("P","FP"))
                        AND (footnote IS NULL)
                        ''',db,
                        params=(frdoc_number,start - sample_context_size,end + sample_context_size))
                    .assign(frdoc_number=frdoc_number))
    else:    
        sample_df = (pd.read_sql_query(f'''
                        SELECT DISTINCT text_id,footnote,text
                        FROM text
                        WHERE (frdoc_number == ?)
                        AND (reg == 0)
                        AND (tag IN ("P","FP"))
                        AND (footnote IS NULL)
                        ''',db,
                        params=(frdoc_number,))
                    .assign(frdoc_number=frdoc_number))
        
    return sample_df


def predict_responses(frdoc_number,span_classifier,change_classifier,
                sample_text_ids=None,sample_context_size=10,min_sample_overlap=0.5,
                progress_bar=True):

    if sample_text_ids is not None:
        sample_text_ids = set(sample_text_ids)
    
    sample_df = get_sample_df(frdoc_number=frdoc_number,
                                sample_text_ids=sample_text_ids,
                                sample_context_size=sample_context_size)
    
    if len(sample_df):

        sample_df['text_ids'] = sample_df['text_id'].apply(lambda x: [x])

        spans_df = span_classifier.predict(sample_df,progress_bar=progress_bar)

        # Enumerate topics
        span_string = ''.join(spans_df['predicted'])

        # Make some "corrections" to the span patterns
        span_string = re.sub(r'(?<=T[CO]T)O','C',span_string)
        span_string = re.sub(r'TO(?=T[OC])','TC',span_string)
        span_string = re.sub(r'TOC','TCC',span_string)
        span_string = re.sub(r'TOOC','TCCC',span_string)

        responses = []
        for i,m in enumerate(re.finditer(r'TC*',span_string)):
            text_ids = set(sample_df.loc[m.start():m.end()-1,'text_id'])
            # print(text_ids)
            # print(sample_text_ids)

            if (sample_text_ids is None) or (len(text_ids & sample_text_ids)/len(text_ids) >= min_sample_overlap):
                responses.append({
                                'frdoc_number':frdoc_number,
                                'response_id':i+1,
                                'text_ids':sorted(text_ids)
                                })

        if responses:

            responses_df = pd.DataFrame(responses)
        
            change_df = change_classifier.predict(responses_df,progress_bar=progress_bar)

            responses_df = (pd.concat([
                                responses_df,
                                change_df[['predicted','Y_prob']]
                                ],axis=1))
            return responses_df
            
    return pd.DataFrame()


def predict_osr(frdoc_number,osr_classifier,
                sample_text_ids=None,sample_context_size=10,min_sample_overlap=0.5,
                progress_bar=True):

    
    sample_df = get_sample_df(frdoc_number=frdoc_number,
                                sample_text_ids=sample_text_ids,
                                sample_context_size=sample_context_size)
    
    if len(sample_df):

        sample_text_ids = set(sample_df['text_id'])

        sample_df['text_ids'] = sample_df['text_id'].apply(lambda x: [x])

        osr_df = osr_classifier.predict(sample_df,progress_bar=progress_bar)

        osr_df = pd.concat([sample_df[['frdoc_number','text_id']],osr_df],axis=1)

        return osr_df
            
    return pd.DataFrame()





# def ts_predict(frdoc_number,osr_classifier,span_classifier,change_classifier,
#                 sample_text_ids=None,sample_context_size=10,osr_context=(20,10),min_sample_overlap=0.5,
#                 progress_bar=True):
    
#     if sample_text_ids:
#         sample_text_ids = set(sample_text_ids)
#         start = min(sample_text_ids)
#         end = max(sample_text_ids)

#         sample_df = (pd.read_sql_query(f'''
#                         SELECT text_id,footnote,text
#                         FROM text
#                         WHERE (frdoc_number == ?)
#                         AND (text_id >= ?)
#                         AND (text_id < ?)
#                         AND (reg == 0)
#                         AND (tag IN ("P","FP"))
#                         AND (footnote IS NULL)
#                         ''',db,
#                         params=(frdoc_number,start - sample_context_size,end + sample_context_size))
#                     .assign(frdoc_number=frdoc_number))
#     else:    
#         sample_df = (pd.read_sql_query(f'''
#                         SELECT DISTINCT text_id,footnote,text
#                         FROM text
#                         WHERE (frdoc_number == ?)
#                         AND (reg == 0)
#                         AND (tag IN ("P","FP"))
#                         AND (footnote IS NULL)
#                         ''',db,
#                         params=(frdoc_number,))
#                     .assign(frdoc_number=frdoc_number))
        
#     if len(sample_df):

#         sample_text_ids = set(sample_df['text_id'])

#         sample_df['text_ids'] = sample_df['text_id'].apply(lambda x: [x])

#         osr_df = osr_classifier.predict(sample_df,progress_bar=progress_bar)

#         osr_df = pd.concat([sample_df[['frdoc_number','text_id']],osr_df],axis=1)

#         # if osr_context:
#         #     # TODO: Make this work someday?
#         #     osr = list(osr_df['predicted'])

#         #     sample_df['prompt'] = [f'Context labels: {" ".join(osr[i-osr_context[0]:i])} [{x}] {" ".join(osr[i+1:i+1+osr_context[1]])}'
#         #                             for i,x in enumerate(osr)]

#         spans_df = span_classifier.predict(sample_df,progress_bar=progress_bar)

#         # Enumerate topics
#         span_string = ''.join(spans_df['predicted'])

#         # Make some "corrections" to the span patterns
#         span_string = re.sub(r'(?<=T[CO]T)O','C',span_string)
#         span_string = re.sub(r'TO(?=T[OC])','TC',span_string)
#         span_string = re.sub(r'TOC','TCC',span_string)
#         span_string = re.sub(r'TOOC','TCCC',span_string)

#         responses = []
#         for i,m in enumerate(re.finditer(r'TC*',span_string)):
#             text_ids = set(sample_df.loc[m.start():m.end()-1,'text_id'])


#             if len(text_ids & sample_text_ids)/len(text_ids) >= min_sample_overlap:
#                 responses.append({
#                                 'frdoc_number':frdoc_number,
#                                 'response_id':i+1,
#                                 'text_ids':sorted(text_ids)
#                                 })


#         if responses:

#             responses_df = pd.DataFrame(responses)
        
#             change_df = change_classifier.predict(responses_df,progress_bar=progress_bar)

#             responses_df = (pd.concat([
#                                 responses_df,
#                                 change_df[['predicted','Y_prob']]
#                                 ],axis=1))
#             return osr_df,responses_df
        
#         else:
#             return osr_df,pd.DataFrame()
    
#     return pd.DataFrame(),pd.DataFrame()


def jaccard(a,b):
    a = set(a)
    b = set(b)
    ab = a & b
    if ab:
        return len(ab) / len(a | b)
    else:
        return 0


def score_predicted(pred_df,gold_df,
                    jaccard_threshold=0.5,
                    labels=None,round_digits=1):

    if labels is None:
        labels = list(gold_df['label'].dropna().unique())

    dfs = {'*':{}}
    for v in labels:
        dfs[v] = {}
    
    pred_df = pred_df.copy()
    gold_df = gold_df.copy()

    pred_df['pred_id'] = [f'{d}-R{r}' for d,r in pred_df[['frdoc_number','response_id']].values]
    gold_df['gold_id'] = [f'{d}-R{r}' for d,r in gold_df[['frdoc_number','response_id']].values]

    pred_df = pred_df.drop('response_id',axis=1)
    gold_df = gold_df.drop('response_id',axis=1)

    # Find True Positives by matching predicted to gold,
    # taking best predicted match for each gold response
    tp_df = (pd.merge(
                    pred_df[['frdoc_number','pred_id','text_ids']]
                        .explode('text_ids'),
                    gold_df[['frdoc_number','gold_id','text_ids']]
                        .explode('text_ids'),
                    'inner',on=['frdoc_number','text_ids'])
                .drop('text_ids',axis=1)
                .drop_duplicates()
                .merge(pred_df,'left',on=['frdoc_number','pred_id'])
                .rename(columns={'text_ids':'pred_text_ids'})
                .merge(gold_df,'left',on=['frdoc_number','gold_id']))


    # Drop matches below the jaccard threshold
    tp_df['jaccard'] = [jaccard(p,g) for p,g in tp_df[['pred_text_ids','text_ids']].values]
    tp_df = tp_df[tp_df['jaccard'] >= jaccard_threshold].copy()

    # Assign the best available match to each gold response as a True Positive
    tp_df['abs_error'] = ((tp_df['label'] == 'Y') - tp_df['Y_prob']).abs()
    tp_df = (tp_df
                .sort_values(['jaccard','abs_error'],ascending=[False,True])
                .groupby('gold_id').first()
                .reset_index()
                .groupby('pred_id').first()
                .reset_index())
    
    dfs['*']['TP'] = tp_df
    for v in labels:
        dfs[v]['TP'] = tp_df.query(f'(label == "{v}") & (predicted == "{v}")')

    # Find False Positives by identifying unmatched predicted responses
    dfs['*']['FP'] = pred_df[~pred_df['pred_id'].isin(tp_df['pred_id'])]
    for v in labels:
        dfs[v]['FP'] = pred_df[(pred_df['predicted']==v) & ~pred_df['pred_id'].isin(dfs[v]['TP']['pred_id'])]
    
    # Find False Negatives by identifying unmatched gold responses
    dfs['*']['FN'] = gold_df[~gold_df['gold_id'].isin(tp_df['gold_id'])]
    for v in labels:
        dfs[v]['FN'] = gold_df[(gold_df['label']==v) & ~gold_df['gold_id'].isin(dfs[v]['TP']['gold_id'])]

    scores = {}
    for v in ['*'] + labels:

        tp = len(dfs[v]['TP'])
        fp = len(dfs[v]['FP'])
        fn = len(dfs[v]['FN'])

        tpr = 100*tp/(tp + fn) if tp + fn else 0
        ppv = 100*tp/(tp + fp) if tp + fp else 0
        f1 = 2*ppv*tpr / (ppv + tpr) if ppv + tpr else 0

        if round_digits:
            tpr = round(tpr,round_digits)
            ppv = round(ppv,round_digits)
            f1 = round(f1,round_digits)


        scores[v] = {
            'TP':tp,
            'FP':fp,
            'FN':fn,
            'TPR':tpr,
            'PPV':ppv,
            'F1':f1,
        }

    return scores,dfs
  

def load_responses(f):
    df = pd.read_csv(f)

    for c in ['text_ids','S_text_ids','R_text_ids']:
        if c in df.columns:
            df[c] = df[c].apply(literal_eval)

    return df