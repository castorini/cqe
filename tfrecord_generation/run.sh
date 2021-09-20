## Generate train data
python3 generate_train_data.py --conversation_file ../../trec/emb_search_for_CAsT/canard/ir_training/query.json \
                                --rewrite_query_file ../../trec/emb_search_for_CAsT/canard/ir_training/manual_resolved_query.tsv \
                                --query_golden_ir_file ../../trec/emb_search_for_CAsT/canard/ir_training/colbert_rank_list.tsv \
                                --query_ir_file ~/cast/cast2019/hard/train.emb.rewrite.train10k.tsv \
                                --corpus_file ../../trec/corpus/CAsT_collection.tsv \
                                --vocab_file ../../trec/dl4marco-bert/models/uncased_L-12_H-768_A-12/vocab.txt \
                                --output_folder ../../trec/emb_search_for_CAsT/canard/ir_training/tfrecord

# python3 gen_query_tfrecord.py --query_file ../../trec/emb_search_for_CAsT/treccastweb/2019/data/cast_training/cast2019_e2e_dev_query.tsv \
# 							   --vocab_file ../../trec/dl4marco-bert/models/uncased_L-12_H-768_A-12/vocab.txt \
# 							   --output_file ~/cast/tfrecord/cast2019_e2e_dev_query