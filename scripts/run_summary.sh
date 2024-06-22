model_name="llama3"
model_name_or_path=/path/to/model    #path to the huggingface model
task_type="summary"  # task type, e.g. summary
dataset_name="xsum"   # testing dataset name. e.g., xsum...
data_path=../eval/data/Summary/xsum_sample_test_3000.jsonl
learning_rate=1e-5  # Optimal parameters
kl_coeff=1
num_train_epochs=10
"""
    Ours:
        cnndm: 
            learning_rate:3e-6 
            num_train_epochs:10 
            kl_coeff:1
        xsum: 
            learning_rate:1e-5 
            num_train_epochs:10 
            kl_coeff:1
        wikihow:
            learning_rate:1e-5 
            num_train_epochs:20 
            kl_coeff:0.03
    Ours+CD:
        cnndm: 
            learning_rate:3e-6 
            num_train_epochs:10 
            kl_coeff:0.01
        xsum: 
            learning_rate:1e-5 
            num_train_epochs:10 
            kl_coeff:0.3
        wikihow:
            learning_rate:1e-5 
            num_train_epochs:20 
            kl_coeff:0.03
        
"""


python ../src/train_and_inference.py  \
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
