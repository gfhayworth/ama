#!/usr/bin/env python
# coding: utf-8

# # Retrieve & Re-Rank QnA Model over Simple Wikipedia
# 
# This examples demonstrates the Retrieve & Re-Rank Setup and allows to search over [Simple Wikipedia](https://simple.wikipedia.org/wiki/Main_Page).
# 
# You can input a query or a question. The script then uses semantic search
# to find relevant passages in Simple English Wikipedia (as it is smaller and fits better in RAM).
# 
# For semantic search, we use `SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')` and retrieve
# 32 potentially passages that answer the input query.
# 
# Next, we use a more powerful CrossEncoder (`cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')`) that
# scores the query and all retrieved passages for their relevancy. The cross-encoder further boost the performance,
# especially when you search over a corpus for which the bi-encoder was not trained for.
# 
# Modified to load pre-scored embeddings
# https://towardsdatascience.com/machine-learning-model-deployment-on-heroku-using-flask-467acb4a34da

# In[1]:


#!pip install -U sentence-transformers


# In[1]:

import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
##import numpy as np
##import pandas as pd
#numpy==1.22.4
#pandas==1.4.3
from flask import Flask, request, render_template
import io
from google.cloud import storage
import os

if not torch.cuda.is_available():
    print("Warning: No GPU found. Please add GPU to your notebook")


#We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
top_k = 32                          #Number of passages we want to retrieve with the bi-encoder

#The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
# about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder

# wikipedia_filepath = 'simplewiki-2020-11-01.jsonl.gz'

# if not os.path.exists(wikipedia_filepath):
#     util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz', wikipedia_filepath)

# passages = []
# with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
#     for line in fIn:
#         data = json.loads(line.strip())

#         #Add all paragraphs
#         #passages.extend(data['paragraphs'])

#         #Only add the first paragraph
#         passages.append(data['paragraphs'][0])

# print("Passages:", len(passages))

# # We encode all passages into our vector space. This takes about 5 minutes (depends on your GPU speed)
# corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)

# # set-up google cloud storage
key_path = "./keys/ama-wiki-0618-6f37524a11be.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

def download_blob_into_memory(bucket_name, blob_name):
    """Downloads a blob into memory."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # blob_name = "storage-object-name"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(blob_name)
    contents = blob.download_as_string()

    print(
        "Downloaded storage object {} from bucket {}.".format(
            blob_name, bucket_name
        )
    )
    return contents


mycorpus_embeddings = torch.load(io.BytesIO(download_blob_into_memory("wiki_assets", "corpus_embeddings.pt")))
mypassages = pickle.loads(download_blob_into_memory("wiki_assets", "passages"))


# This function will search all wikipedia articles for passages that
# answer the query
def search(query):
    ##### Sematic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, mycorpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, mypassages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    predictions = hits[0:3]
    return predictions


# predictions = search(query = "When did ted cassidy die")
# prediction_text = []
# for hit in predictions:
#     prediction_text.append("\t{:.3f}\t{}".format(hit['cross-score'], mypassages[hit['corpus_id']]))
# #prediction_text

# for p in prediction_text:
#     print(p)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    qry = request.form.to_dict()
    qry = list(qry.values())
        
    predictions = search(query = qry[0])
    prediction_text = ''
    for hit in predictions:
        prediction_text+="{:.3f}\t{}\n\n".format(hit['cross-score'], 
                         mypassages[hit['corpus_id']])
    
    return render_template('index.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
