python generate_train_data.py --conversation_file ../../canard/query.json \
                                --rewrite_query_file ../../canard/manual_resolved_query.tsv \
                                --query_ir_file ../../canard/colbert_rank_list.tsv \
                                --corpus_file ../../../corpus/CAsT_collection.tsv \
                                --vocab_file ~/uncased_L-12_H-768_A-12/vocab.txt \
                                --output_folder ../../canard/tfrecord