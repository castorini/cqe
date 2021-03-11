python3 generate_train_data.py --conversation_file ../../trec/emb_search_for_CAsT/canard/ir_training/query.json \
                                --rewrite_query_file ../../trec/emb_search_for_CAsT/canard/ir_training/manual_resolved_query.tsv \
                                --query_ir_file ../../trec/emb_search_for_CAsT/canard/ir_training/colbert_rank_list.tsv \
                                --corpus_file ../../trec/corpus/CAsT_collection.tsv \
                                --vocab_file ../../trec/dl4marco-bert/models/uncased_L-12_H-768_A-12/vocab.txt \
                                --output_folder ../../trec/emb_search_for_CAsT/canard/ir_training/tfrecord