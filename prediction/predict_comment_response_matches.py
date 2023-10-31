from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3
import torch
import random
from tqdm import tqdm

from nama.embedding_similarity import load_similarity_model

from regcomment_influence.response_extraction.paragraph_classifier import get_input_text, add_footnotes
from regcomment_influence.response_linking import comments
from regcomment_influence import config



def init_tables(db):
    
    db.execute('''
                CREATE TABLE IF NOT EXISTS comment_responses
                (
                    comment_id TEXT NOT NULL,
                    frdoc_number TEXT NOT NULL,
                    response_id INT NOT NULL,
                    score REAL NOT NULL,
                    norm_score REAL NOT NULL
                )
                ''')

    db.execute('''
                CREATE INDEX IF NOT EXISTS idx_comment_responses_comment_id
                ON comment_responses (comment_id)
                ''')

    db.execute('''
                CREATE INDEX IF NOT EXISTS idx_comment_responses_frdoc_number
                ON comment_responses (frdoc_number)
                ''')

    db.execute('''
                CREATE TABLE IF NOT EXISTS comment_response_sample
                (
                    frdoc_number TEXT NOT NULL,
                    min_score REAL NOT NULL,
                    min_norm_score REAL NOT NULL,
                    r_max_tol REAL NOT NULL,
                    status TEXT
                )
                ''')



def get_remaining_frdocs(db,shuffle=True):

    existing = {r[0] for r in db.execute('''SELECT DISTINCT frdoc_number
                                            FROM comment_response_sample
                                            ''').fetchall()}
    
    with_comments = {r[0] for r in db.execute('''SELECT DISTINCT responses.frdoc_number
                                                FROM responses
                                                
                                                INNER JOIN rulemaking.frdoc_input_comment_counts AS comments
                                                ON comments.frdoc_number == responses.frdoc_number
                                                
                                                WHERE comment_count > 0
                                                ''').fetchall()}


    remaining = sorted(with_comments - existing)

    
    if shuffle:
         random.shuffle(remaining)

    print(f'Found existing comment_response data for {len(existing)} frdocs.')

    return remaining


def add_frdoc_comment_responses(frdoc_number,db,similarity_model,min_score,min_norm_score,r_max_tol,batch_size,embeddings_device):
     
    # Load response text
    responses_df = (pd.read_sql_query('''
                                        SELECT response_id,text_id
                                        FROM response_paragraphs
                                        WHERE (frdoc_number == ?)
                                        ''',db,params=(frdoc_number,))
                            .groupby('response_id')[['text_id']].agg(lambda x: list(x))
                            .rename(columns={'text_id':'text_ids'})
                            .reset_index()
                            .assign(frdoc_number=frdoc_number)
                        )
    
    responses_df = add_footnotes(responses_df)

    # Get response text
    response_text = [get_input_text(d,p,sep='\n')
                        for d,p in responses_df[['frdoc_number','text_ids']].values]
    
    response_ids = responses_df['response_id'].values

    response_embeddings = similarity_model.embed(response_text,to=embeddings_device)
    V_r = response_embeddings.V


    chunks_df = (comments.get_input_comment_chunks(frdoc_number)
                    .groupby('paragraph_ids')[['comment_id']].agg(lambda x: list(x))
                    .rename(columns={'comment_id':'comment_ids'})
                    .reset_index())

    if not len(chunks_df):
        print(f'Warning: No comment chunks found for {frdoc_number}')
        db.execute('INSERT INTO comment_response_sample VALUES (?,?,?,?,?)',(frdoc_number,min_score,min_norm_score,r_max_tol,'no chunks'))
        db.commit()

        return
    
    chunk_paragraph_ids = chunks_df['paragraph_ids'].values
    chunk_comment_ids = chunks_df['comment_ids'].values
    
    # Find response-chunk matches in batches
    chunk_matches = []
    for i in tqdm(range(0,len(chunks_df),batch_size),desc=frdoc_number,delay=3):
        chunk_text = [comments.get_comment_chunk_text(pars) for pars in chunk_paragraph_ids[i:i+batch_size]]
        chunk_embeddings = similarity_model.embed(chunk_text,batch_size=batch_size,progress_bar=False,to=embeddings_device)

        V_c = chunk_embeddings.V

        scores = similarity_model.score_model(V_c@V_r.T)

        # Find best response for each comment
        r_max,_ = torch.max(scores,dim=1,keepdim=True) # Max across all responses


        # select matches
        match_locs = torch.nonzero((scores > min_score) & (scores >= r_max - r_max_tol))

        for m_c,m_r in match_locs:            
            chunk_matches.append(dict(
                                frdoc_number=frdoc_number,
                                comment_ids=chunk_comment_ids[i+m_c],
                                response_id=response_ids[m_r],
                                score=scores[m_c,m_r].item(),
                                ))

    if chunk_matches:
        matches_df = (pd.DataFrame(chunk_matches)
                        .explode('comment_ids')
                        .rename(columns={'comment_ids':'comment_id'})
                        .groupby(['comment_id','response_id'])[['score']].max()
                        .reset_index()
                        .assign(
                            norm_score=lambda df: df['score']/df.groupby('response_id')['score'].transform('max'),
                            frdoc_number=frdoc_number))
        
        matches_df = matches_df[matches_df['norm_score']>min_norm_score]

        db.executemany('INSERT INTO comment_responses VALUES (?,?,?,?,?)',
                    matches_df[['comment_id','frdoc_number','response_id','score','norm_score']].values)


    db.execute('INSERT INTO comment_response_sample VALUES (?,?,?,?,?)',
                    (frdoc_number,min_score,min_norm_score,r_max_tol,None))
   
    db.commit()

    # Clear the embeddings off the GPU
    del response_embeddings




def main(args):

    db = sqlite3.connect(Path(config.data_dir)/'processed'/'responses.db')
    db.execute(f'ATTACH "{config.rulemaking_db_file}" AS rulemaking')


    if args.overwrite_all:
         db.execute('DROP TABLE IF EXISTS comment_responses')
         db.execute('DROP TABLE IF EXISTS comment_response_sample')
         db.commit()

    init_tables(db)

    remaining = get_remaining_frdocs(db,shuffle=args.shuffle)
    print(f'Matching comments to {len(remaining)} remaining frdocs')

    similarity_model = load_similarity_model(args.similarity_model)
    similarity_model.to(args.devices[0])
   
    if args.fp16:
        similarity_model.half()

    for frdoc_number in tqdm(remaining,'Finding comment-response matches',smoothing=0):
        add_frdoc_comment_responses(
                            frdoc_number=frdoc_number,
                            db=db,
                            similarity_model=similarity_model,
                            min_score=args.min_score,
                            min_norm_score=args.min_norm_score,
                            r_max_tol=args.r_max_tol,
                            batch_size=args.batch_size,
                            embeddings_device=args.devices[1])



parser = ArgumentParser()

parser.add_argument('--similarity_model',type=str,default=Path(config.data_dir)/'models'/'nama_sbert_256'/f'nama_sbert_1.bin')
parser.add_argument('--overwrite_all',action='store_true',dest='overwrite_all')
parser.add_argument('--shuffle',action='store_true',dest='shuffle')
parser.add_argument('--db_file',type=str,default=Path(config.data_dir)/'processed'/'responses.db')
parser.add_argument('--devices',type=str,nargs=2,default=['cpu','cpu'])
parser.add_argument('--fp16',action='store_true',dest='fp16')
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--min_score',type=float,default=0.1)
parser.add_argument('--min_norm_score',type=float,default=0.5)
parser.add_argument('--r_max_tol',type=float,default=0.01)

args = parser.parse_args()

main(args)

