# -*- coding: utf-8 -*-
'''
Anserini: A Lucene toolkit for replicable information retrieval research

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import argparse
import time
import json
import numpy as np
import collections
from pyserini.search import SimpleSearcher
from pyserini.index import IndexReader
from pyserini.search import querybuilder
from util import load_term_weight_tfrecords
from progressbar import *
from numpy import linalg as LA
import sys
sys.path.append('./CQE/tfrecord_generation')

import tokenization
def read_corpus(corpus_path):
    id_to_doc = {}
    f = open(corpus_path, 'r')
    # with open(corpus_path, 'r') as f:
    print('Read Corpus...')

    for line in f:
        try:
            text = line.strip().split("\t")
            docid = text[0]
            doc= ' '.join(text[1:])
            id_to_doc[docid] = doc
        except:
            print("skip %s"%(line.strip()))
    f.close()
    return id_to_doc

def build_query(query, query_token_weights, threshold):
    context ='|'.join(query.split('|')[:-1])
    context_token_num = len(tokenizer.tokenize(context)) + 3
    origin_query = query.split('|')[-1]
    query =  context + '[Q]' + origin_query
    query_tokens = tokenizer.tokenize(query)
    query_token_weights = query_token_weights[1:len(query_tokens)+1]
    mean_l2 = query_token_weights.mean()
    should = querybuilder.JBooleanClauseOccur['should'].value
    boolean_query_builder = querybuilder.get_boolean_query_builder()
    is_context = True
    term_weight = 0
    term =''
    for i, token in enumerate(query_tokens): 
        if '##' in token:
            term += token[2:]
            term_weight = max(term_weight, query_token_weights[i])
        else:
            if ( (term_weight > threshold) or (i>=context_token_num)): #10.5, 12
                try:
                    term = querybuilder.get_term_query(term)
                    boost = querybuilder.get_boost_query(term, term_weight/mean_l2)
                    boolean_query_builder.add(boost, should)
                except:
                    x=1

            term = token
            term_weight = query_token_weights[i]


    return boolean_query_builder.build()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HQE for CAsT.')
    parser.add_argument('--output', required=True, default='', help='output file')
    parser.add_argument('--index', default='cast2019', help='index path')
    parser.add_argument('--topk', default=1000, help='number of hits to retrieve')
    parser.add_argument('--threshold', type=float, default=10.5, help='threshold for term weight')
    # See our MS MARCO documentation to understand how these parameter values were tuned.
    parser.add_argument('--k1', default=0.82, help='BM25 k1 parameter')
    parser.add_argument('--b', default=0.68, help='BM25 b parameter')
    parser.add_argument('--rm3', action='store_true', default=False, help='use RM3')
    parser.add_argument('--fbTerms', default=10, type=int, help='RM3 parameter: number of expansion terms')
    parser.add_argument('--fbDocs', default=10, type=int, help='RM3 parameter: number of documents')
    parser.add_argument('--originalQueryWeight', default=0.8, type=float, help='RM3 parameter: weight to assign to the original query')
    parser.add_argument("--query_text_path", type=str, required=True, help='tsv file with format qid\tquery')
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--query_emb_path", type=str)
    parser.add_argument("--data_type", type=str, default='16')
    # HQE related hyperparameters. The default is tuned on CAsT train data
    args = parser.parse_args()
    # searcher = SimpleSearcher(args.index) #SimpleSearcher.from_prebuilt_index('cast2019')
    searcher = SimpleSearcher.from_prebuilt_index(args.index)
    searcher.set_bm25(float(args.k1), float(args.b))
    print('Initializing BM25, setting k1={} and b={}'.format(args.k1, args.b))
    if args.rm3:
        searcher.set_rm3(args.fbTerms, args.fbDocs, args.originalQueryWeight)
        print('Initializing RM3, setting fbTerms={}, fbDocs={} and originalQueryWeight={}'.format(args.fbTerms, args.fbDocs, args.originalQueryWeight))

    fout = open(args.output, 'w')
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=True)
    query_token_weights, qids=load_term_weight_tfrecords([args.query_emb_path],\
                                    dim=768, data_type=args.data_type)
    widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets, maxval=len(qids)).start()
    with open(args.query_text_path, 'r') as f:
        start_time = time.time()
        count=0
        for i, line in enumerate(f):
            line = line.strip().split('\t')
            qid = line[0]
            concat_query = line[1]
            concat_query = tokenization.convert_to_unicode(concat_query)
            query = build_query(concat_query, query_token_weights[i], args.threshold)

            hits = searcher.search(query, int(args.topk))
            for rank in range(len(hits)):
                docno = hits[rank].docid

                fout.write('{} Q0 {} {} {} {}\n'.format(qid, docno, rank + 1, hits[rank].score, 'anserini'))

            count+=1
            pbar.update(i + 1)
            # if count%100==0:
            #     time_per_query = (time.time() - start_time) / (count)
            #     print('Retrieving {} queries ({:0.3f} s/query)'.format(count, time_per_query))
            #     start_time = time.time()
    time_per_query = (time.time() - start_time) / (count)
    print('Retrieving {} queries ({:0.3f} s/query)'.format(count, time_per_query))
    fout.close()
    print('Done!')



