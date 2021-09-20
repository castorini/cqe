tpu_address=cast
Your_GS_Folder=gs://irresearch
ctpu up --name=cast --project=crested-bonfire-314219 --zone=us-central1-f  --tpu-size=v2-8  --tpu-only  --tf-version=1.15  --noconf
# python main.py --use_tpu=True \
#                --tpu_address=$tpu_address \
#                --do_train=False \
#                --do_eval=True \
#                --bert_pretrained_dir=$Your_GS_Folder/uncased_L-12_H-768_A-12 \
#                --init_checkpoint=$Your_GS_Folder/uncased_L-12_H-768_A-12/tct-colbertv1.hard/model.ckpt-100000 \
#                --data_dir=$Your_GS_Folder/cast/tfrecord \
#                --train_file=dataset_response_train_tower.tf \
#                --eval_file=dataset_dev_tower.tf \
#                --num_train_steps=10000 \
#                --max_query_length=236 \
#                --output_dir=$Your_GS_Folder/uncased_L-12_H-768_A-12/cqe.response.v2 \
#                --train_model=student \
#                --eval_model=student \
#                --kd_source=colbert \
#                --loss=kl \


#Output Query embeddings
for file in cast2021.eval.cqe.response.last.query
do
    python main.py --use_tpu=True \
               --tpu_address=$tpu_address \
               --do_output=True \
               --bert_pretrained_dir=$Your_GS_Folder/uncased_L-12_H-768_A-12 \
               --init_checkpoint=$Your_GS_Folder/uncased_L-12_H-768_A-12/qe.response.v2/model.ckpt-10000 \
               --data_dir=$Your_GS_Folder/cast/tfrecord \
               --train_file=dataset_train_tower.tf \
               --eval_file=dataset_dev_tower.tf \
               --num_train_steps=10000 \
               --max_query_length=236 \
               --output_dir=$Your_GS_Folder/uncased_L-12_H-768_A-12/cqe.response.v2 \
               --train_model=student \
               --eval_model=student \
               --kd_source=colbert \
               --loss=kl \
               --embedding_file=$file \
               --num_tpu_cores=1 \
               --eval_batch_size=1 \
               --doc_type=0
done
#Output Corpus embeddings
# python main.py --use_tpu=True \
#                --tpu_address=$tpu_address \
#                --do_output=True \
#                --bert_pretrained_dir=$Your_GS_Folder/uncased_L-12_H-768_A-12 \
#                --init_checkpoint=$Your_GS_Folder/uncased_L-12_H-768_A-12/cqe.v2/model.ckpt-10000 \
#                --data_dir=$Your_GS_Folder/cast/tfrecord \
#                --train_file=dataset_train_tower.tf \
#                --eval_file=dataset_dev_tower.tf \
#                --num_train_steps=10000 \
#                --output_dir=$Your_GS_Folder/uncased_L-12_H-768_A-12/cqe.v2 \
#                --train_model=student \
#                --eval_model=student \
#                --kd_source=colbert \
#                --loss=kl \
#                --embedding_file=cast0 \

# python main.py --use_tpu=True \
#                --tpu_address=$tpu_address \
#                --do_output=True \
#                --bert_pretrained_dir=$Your_GS_Folder/uncased_L-12_H-768_A-12 \
#                --init_checkpoint=$Your_GS_Folder/uncased_L-12_H-768_A-12/cqe.v2/model.ckpt-10000 \
#                --data_dir=$Your_GS_Folder/cast/tfrecord \
#                --train_file=dataset_train_tower.tf \
#                --eval_file=dataset_dev_tower.tf \
#                --num_train_steps=10000 \
#                --output_dir=$Your_GS_Folder/uncased_L-12_H-768_A-12/cqe.v2 \
#                --train_model=student \
#                --eval_model=student \
#                --kd_source=colbert \
#                --loss=kl \
#                --embedding_file=cast1 \
#                --num_tpu_cores=1 \
#                --eval_batch_size=1 \


