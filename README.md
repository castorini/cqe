# Contextualized Query Embeddings for Conversational Search (CQE)
The repo is the code for our paper:
*[Contextualized Query Embeddings for Conversational Search](https://arxiv.org/abs/2104.08707)* Sheng-Chieh Lin, Jheng-Hong Yang and Jimmy Lin
In this repo, we will use the data from [CAsT repo](https://github.com/daltonj/treccastweb).
## Prepare
```shell=bash
git clone https://github.com/daltonj/treccastweb.git
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-24_H-1024_A-16.zip
export BERT_MODEL_DIR=./uncased_L-12_H-768_A-12
export CHECKPOINT=cqe_checkpoint/model.ckpt-10000
export DATA_DIR=./cast
export CORPUS_EMB=${DATA_DIR}/doc_emb
export QUERY_EMB=${DATA_DIR}/query_emb
export QUERY_NAME=cas2019.eval
export INDEX_PATH=${DATA_DIR}/indexes
export INTERMEDIATE_PATH=${DATA_DIR}/intermediate
```
## Index corpus embedding
We first split the corpus and convert the text into tfrecord for inference
```shell=bash
split -d -l 1000000 ${DATA_DIR}/collection.tsv ${DATA_DIR}/collection.part
# Convert passages in the collection
python ./CQE/tfrecord_generation/convert_collection_to_tfrecord.py \
  --output_folder=${DATA_DIR}/corpus_tfrecord \
  --vocab_file=${BERT_MODEL_DIR}/vocab.txt \
  --max_seq_length=154 \
  --corpus_path=${DATA_DIR} \
  --corpus_prefix=collection.part \
  --doc_type=passage \
for i in $(seq 0 38)
do
    srun --gres=gpu:p100:1 --mem=16G --cpus-per-task=2 --time=2:00:00 \
    python ./CQE/train/main.py --use_tpu=False \
                 --tpu_address=$tpu_address \
                 --do_output=True \
                 --bert_pretrained_dir=${BERT_MODEL_DIR} \
                 --eval_checkpoint=${CHECKPOINT} \
                 --max_doc_length=154 \
                 --doc_type=1 \
                 --eval_batch_size=100 \
                 --output_dir=${DATA_DIR}/doc_emb \
                 --data_dir=${DATA_DIR}/corpus_tfrecord \
                 --embedding_file=collection.part-${i} &
done
# indexing using faiss
python ./CQE/dr/index.py --index_path ${INDEX_PATH} \
     --corpus_emb_path ${CORPUS_EMB} --merge_index --passages_per_file 1000000 --max_passage_each_index 10000000 \
```
## CQE Embedding output and dense Search
```shell=bash
python ./CQE/tfrecord_generation/gen_query_tfrecord.py \
     --query_file ./treccastweb/2019/data/evaluation/evaluation_topics_v1.0.json \
     --vocab_file ${BERT_MODEL_DIR}/vocab.txt \
     --output_folder query_tfrecord \
     --output_filename ${QUERY_NAME}

#Then, encode the text into conversational embeddings.
python ./CQE/train/main.py --use_tpu=False \
          --tpu_address=$tpu_address \
          --do_output=True \
          --bert_pretrained_dir=${BERT_MODEL_DIR} \
          --eval_checkpoint ${CHECKPOINT} \
          --data_dir=query_tfrecord \
          --max_query_length=136 \
          --output_dir=query_emb \
          --train_model=student \
          --eval_model=student \
          --embedding_file=${QUERY_NAME} \
          --eval_batch_size=1 \
          --doc_type=0

#Dense search
for index in ${INDEX_PATH}/*
do
    python ./CQE/retrieval/dense.search.py --index_file $index --intermediate_path ${INTERMEDIATE_PATH} \
          --topk 1000 --query_emb_path ${QUERY_EMB}/embeddings-${QUERY_NAME}.tf \
          --batch_size 144 --threads 36
done
#Merge and output final result
python ./CQE/retrieval/merge.result.py --topk 1000 --intermediate_path ${INTERMEDIATE_PATH} \
                         --output ${DATA_DIR}/${QUERY_NAME}.dense.result.trec \
                         --id_to_doc_path ${DATA_DIR}/corpus_tfrecord \
                         --id_to_query_path ${DATA_DIR}/query_tfrecord/${QUERY_NAME}.id
#Evaluation
python -m pyserini.eval.trec_eval -c -mndcg_cut.3,1000 \
 ${QREL_PATH} ${DATA_DIR}/${QUERY_NAME}.dense.result.trec
python -m pyserini.eval.trec_eval -c -l 2 -mrecall.1000 \
 ${QREL_PATH} ${DATA_DIR}/${QUERY_NAME}.dense.result.trec

######################################
ndcg_cut_3              all     0.5018
ndcg_cut_1000           all     0.5592
recall_100              all     0.5215
recall_1000             all     0.7843
######################################
```
## CQE sparse search
We use CQE L2 norm to select tokens from historical context and also as the term weights for BM25 search.
```shell=bash
#Sparse search
python ./CQE/retrieval/sparse.search.py --topk 1000  --threshold 10 \
             --query_text_path ${DATA_DIR}/query_tfrecord/${QUERY_NAME}.tsv \
             --vocab_file ${BERT_MODEL_DIR}/vocab.txt \
             --query_emb_path ${QUERY_EMB}/embeddings-${QUERY_NAME}.tf \
             --output ${DATA_DIR}/${QUERY_NAME}.sparse.result.trec \
#Evaluation
python -m pyserini.eval.trec_eval -c -mndcg_cut.3,1000 \
 ${QREL_PATH} ${DATA_DIR}/${QUERY_NAME}.sparse.result.trec
python -m pyserini.eval.trec_eval -c -l 2 -mrecall.100,1000 \
 ${QREL_PATH} ${DATA_DIR}/${QUERY_NAME}.sparse.result.trec

######################################
ndcg_cut_3              all     0.2734
ndcg_cut_1000           all     0.4632
recall_100              all     0.3808
recall_1000             all     0.7740
######################################
```
## CQE fusion
We directly conduct fusion on the sparse and dense ranking lists.
```shell=bash
#Fusion
python ./CQE/retrieval/fuse.py --topk 1000 --rank_file0 ${DATA_DIR}/${QUERY_NAME}.dense.result.trec \
                               --rank_file1 ${DATA_DIR}/${QUERY_NAME}.sparse.result.trec \
                               --output ${DATA_DIR}/${QUERY_NAME}.fusion.result.trec \
#Evaluation
python -m pyserini.eval.trec_eval -c -mndcg_cut.3,1000 \
 ${QREL_PATH} ${DATA_DIR}/${QUERY_NAME}.fusion.result.trec
python -m pyserini.eval.trec_eval -c -l 2 -mrecall.100,1000 \
 ${QREL_PATH} ${DATA_DIR}/${QUERY_NAME}.fusion.result.trec

######################################
ndcg_cut_3              all     0.5069
ndcg_cut_1000           all     0.6107
recall_100              all     0.5804
recall_1000             all     0.8543
```
## CQE fusion
To optimize the top fusion ranking result (NDCG@3), we tune the threshold for term selection and conduct sparse search again.
```shell=bash
#Sparse search
python ./CQE/retrieval/sparse.search.py --topk 1000 --threshold 12 \
             --query_text_path ${DATA_DIR}/query_tfrecord/${QUERY_NAME}.tsv \
             --vocab_file ${BERT_MODEL_DIR}/vocab.txt \
             --query_emb_path ${QUERY_EMB}/embeddings-${QUERY_NAME}.tf \
             --output ${DATA_DIR}/${QUERY_NAME}.sparse.result.trec \
#Fusion
python ./CQE/retrieval/fuse.py --topk 1000 --rank_file0 ${DATA_DIR}/${QUERY_NAME}.dense.result.trec \
                               --rank_file1 ${DATA_DIR}/${QUERY_NAME}.sparse.result.trec \
                               --output ${DATA_DIR}/${QUERY_NAME}.fusion.result.trec \
#Evaluation
python -m pyserini.eval.trec_eval -c -mndcg_cut.3,1000 \
 ${QREL_PATH} ${DATA_DIR}/${QUERY_NAME}.fusion.result.trec
python -m pyserini.eval.trec_eval -c -l 2 -mrecall.100,1000 \
 ${QREL_PATH} ${DATA_DIR}/${QUERY_NAME}.fusion.result.trec

######################################
ndcg_cut_3              all     0.5173
ndcg_cut_1000           all     0.5985
recall_100              all     0.5614
recall_1000             all     0.8234
######################################
```
