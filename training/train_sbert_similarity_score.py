import os
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sqlite3
from tqdm import tqdm

import nama
from nama.scoring import score_predicted, split_on_groups
from nama.embedding_similarity import EmbeddingSimilarityModel,load_similarity_model

from regcomment_influence import config
from regcomment_influence.response_extraction.paragraph_classifier import get_input_text, add_footnotes
from regcomment_influence.response_linking import comments



device='cuda:2'
max_responses_per_rule = 10
rule_frac = 1
n_iterations = 5

save_folder = 'nama_sbert'

model_save_dir = Path(config.data_dir)/'models'/save_folder
output_dir = Path(config.data_dir)/'processed'/'response_linking'/save_folder

for d in model_save_dir,output_dir:
    if not os.path.isdir(d):
        os.makedirs(d)


train_kwargs = {
                'max_epochs': 1,
                'warmup_frac': 0.1,
                'transformer_lr':1e-5,
                'score_lr':0,
                'alpha':50,
                'use_counts':False,
                'batch_size':6,
                'model_name':'sentence-transformers/multi-qa-mpnet-base-dot-v1',
                'd':None,
                'upper_case':False,
                'pooling':'cls'
                }


sim = EmbeddingSimilarityModel(**train_kwargs)
sim.to(device)


db = sqlite3.connect(config.frdocs_db_file)

db.execute(f'ATTACH "{config.rulemaking_db_file}" AS rulemaking')
db.execute(f'ATTACH "{Path(config.data_dir)/"processed"/"responses.db"}" AS responses')
db.execute(f'ATTACH "{config.orgs_db_file}" AS orgs')


pd.read_sql_query('SELECT * FROM rulemaking.frdoc_input_comment_counts LIMIT 5',db)


frdoc_sample_df = (pd.read_sql_query('''
                                SELECT DISTINCT responses.frdoc_number 
                                FROM responses

                                INNER JOIN rulemaking.frdoc_input_comment_counts AS comments
                                ON (comments.frdoc_number==responses.frdoc_number)

                                WHERE (comments.comment_count > 0)
                                AND (comments.comment_count < 10)
                                
                                ''',db)
                        .sample(frac=rule_frac,random_state=1))


for i in range(1,n_iterations+1):
    """
    Iteratively train similarity algorithm using an existing similarity model.

    - Each iteration i uses the model from i-1 to build training data
    - Iteration "0" is the base roberta model.
    """

    n_responses = 0
    n_comments = 0
    n_chunks = 0

    matched = []
    for j,frdoc_number in tqdm(enumerate(frdoc_sample_df['frdoc_number']),total=len(frdoc_sample_df),
                               desc=f'Building training data for iteration {i}'):
        """
        Building training data

        - For each rule (frdoc_number):

            1. Get all comment chunks
                - skip rules that have too many or too few comments
                - skip rules that have too many or too few comment chunks

            2. Get the text of all responses in the sample
            
            3. Use the similarity model from iteration i-1 to identify the most
               comment chunk for each response (each response will be 
               matched to exactly one comment chunk)

            4. Add the matched pairs to the "matched" list to use as training data
        """

        # Get comment text
        comment_chunks_df = comments.get_input_comment_chunks(frdoc_number)

        if not (10 < len(comment_chunks_df) < 1000):
            continue

        comment_chunks_df['comment_text'] = [f'Comment text: {comments.get_comment_chunk_text(p).strip()}'
                                        for p in comment_chunks_df['paragraph_ids']]

        n_comments += comment_chunks_df['comment_id'].nunique()
        n_chunks += len(comment_chunks_df)
        
        # Get response text (Sample up to 10 responses per rule)
        responses_df = (pd.read_sql_query('''
                                        SELECT * 
                                        FROM response_paragraphs

                                        WHERE (frdoc_number==?)
                                        ''',db,params=(frdoc_number,))
                                        .groupby(['frdoc_number','response_id'])
                                        [['text_id']].agg(lambda x: list(x))
                                        .reset_index()
                                        .rename(columns={'text_id':'text_ids'})
                                        .sample(frac=1,random_state=j)
                                        .head(10)
                                        )

        n_responses += len(responses_df)
        
        responses_df = add_footnotes(responses_df)

        responses_df['response_text'] = [get_input_text(d,p,sep='\n')
                                        for d,p in responses_df[['frdoc_number','text_ids']].values]
        

        # Embed comment and response text in the same embedding object
        all_text = (list(responses_df['response_text']) 
                    + list(comment_chunks_df['comment_text']))

        embeddings = sim.embed(all_text,device=device)

        # sim.to('cpu')
        # embeddings.to(device)


        # Match each response to the its nearest comment chunk
        nearest = embeddings.unite_nearest(
                        target_strings=set(comment_chunks_df['comment_text']),
                        threshold=0,
                        )
        

        # Select the response-comment chunk pair associated with each response
        responses_df['group'] = [nearest[s] for s in responses_df['response_text']]
        comment_chunks_df['group'] = [nearest[s] for s in comment_chunks_df['comment_text']]

        matched_df = pd.merge(
                    responses_df,
                    comment_chunks_df,
                    on='group')

        # Add the matched pair to the training data
        matched.append(matched_df[['frdoc_number','response_id','comment_id','paragraphs','response_text','comment_text']])
        
        # Clear the embeddings off the GPU
        embeddings.to('cpu')


    # Build the combined training data for all rules
    matched_df = pd.concat(matched)

    # Make sure no strings appear in two rules
    for c in ['response_text','comment_text']:
        matched_df = matched_df.drop_duplicates(subset=[c])

    # Convert the training data to a nama matcher
    matcher = nama.from_df(matched_df,match_format='pairs',pair_columns=['response_text','comment_text'])

    print(f'Training model iteration {i}')

    # Split test/train samples
    test,train = split_on_groups(matcher,0.2,seed=1)

    # Clear old model off the GPU
    sim.to('cpu')

    # Create new similarity model
    sim = EmbeddingSimilarityModel(**train_kwargs)
    sim.to(device)


    # Train the new model on the training data
    history_df = sim.train(train,verbose=True,**train_kwargs)
    
    # Cache embeddings for repeated prediction
    test_embeddings = sim.embed(test)

    # Score the model on the test data
    results = []
    for threshold in tqdm(np.linspace(0,1,51),desc='scoring'):
        pred = test_embeddings.unite_similar(threshold=threshold,progress_bar=False)

        scores = score_predicted(pred,test,use_counts=train_kwargs['use_counts'])

        scores.update(train_kwargs)

        scores['threshold'] = threshold
        scores['alpha'] = sim.score_model.alpha.item()
        scores['iteration'] = i

        results.append(scores)
    
    test_embeddings.to('cpu')

    # Save this model
    sim.save(model_save_dir/f'nama_sbert_{i}.bin')

    # Save a sample of the matched strings for review
    # sample_df = matched_df.merge(response_match_sample_df)
    matched_df.to_csv(output_dir/f'nama_matched_train_{i}.csv',index=False)


    # # Save the results for this and prior iterations
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir/f'nama_matched_train_results_{i}.csv',index=False)

    # Save training history (loss, etc.) for this and prior iterations
    history_df.to_csv(output_dir/f'nama_matched_train_history_{i}.csv',index=False)

    print(f'Completed iteration {i} with:\n\t{len(frdoc_sample_df)} rules\n\t{n_responses} responses\n\t{n_comments} comments\n\t{n_chunks} comment chunks')



# results_df = pd.concat([
#                     pd.read_csv(Path(config.data_dir)/'processed'/'response_linking'/f'nama_matched_train_results_{i}.csv')
#                     for i in range(5)])

# # Plot the results

# ax = plt.subplot()
# for run_vals, df in results_df.groupby('iteration'):
#     df.plot('recall','precision',ax=ax,label=f'{run_vals=}')
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#         fancybox=True, ncol=1)
# plt.show()

# ax = plt.subplot()
# for run_vals, df in results_df.groupby('iteration'):
#     df.plot('threshold','F1',ax=ax,label=f'{run_vals=}')
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#         fancybox=True, ncol=1)
# plt.show()







