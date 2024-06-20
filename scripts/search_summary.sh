model_name="llama3"
model_name_or_path=/mnt/data102_d2/huggingface/models/Meta-Llama-3-8B-Instruct   # TODO: path to the model
task_type="summary"  # task type, e.g. summary
dataset_name="xsum"   # testing dataset name. e.g., xsum...
data_path=../eval/data/Summary/xsum_sample_train_1000.jsonl # TODO: path to the data. Note: If hyperparameter search is needed, please perform it on the training dataset.
learning_rate=1e-5  # Don't need to change
kl_coeff=1
num_train_epochs=10

CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com \
python ../src/grid_search.py  \
    --model_name_or_path ${model_name_or_path} \
    --data_path ${data_path} \
    --bf16 True --num_data 4000 \
    --output_path ../result \
    --output_dir ../log  \
    --per_device_train_batch_size 1  \
    --per_device_eval_batch_size 1   \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no  \
    --weight_decay 0. \
    --logging_steps 1 \
    --fsdp_transformer_layer_cls_to_wrap "'LlamaDecoderLayer'"  \
    --tf32 false  \
    --lr_scheduler_type constant  \
    --report_to none  \
    --remove_unused_columns false  \
    --model_name ${model_name}  \
    --learning_rate ${learning_rate} \
    --kl_coeff ${kl_coeff} \
    --num_train_epochs ${num_train_epochs} \
    --dataset_name ${dataset_name} \
    --task_type ${task_type}  \
    --choose_cd False \
    --choose_dola False
