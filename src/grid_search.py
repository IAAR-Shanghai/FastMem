import sys
import ast
import copy
import json
import os
import logging
import statistics
import string

import regex as re
import torch
import transformers
import tqdm
from dataclasses import dataclass, field
from datasets import load_dataset
import evaluate
from evaluate import load as load_metric
from numpy import mean
from termcolor import colored
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import Trainer
from typing import Dict, Optional, Sequence

from utils import *
from NewModel_Llama_any_layer import *
from NewModel_Qwen_any_layer import *
from Trainer import NewTrainer
from train_and_inference import train_and_inference
from train_and_inference import DataCollatorForPretrainDataset

sys.setrecursionlimit(1000000)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Arguments: Algorithm Hyperparameter Settings
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    train_layer: Optional[int] = field(default=1, metadata={"help": "The number of layers to train."})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    doc_token: int = field(default=7500, metadata={"help": "The number of tokens in the document."})
    num_data: int = field(default=4000, metadata={"help": "the size of dataset."})
    output_path: str = field(default=None, metadata={"help": "Path to output."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    optimizer_params: Optional[str] = field(default="AdamW", metadata={"help": "Optimizer parameters."})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "The initial learning rate for AdamW."})
    kl_coeff: Optional[float] = field(default=0.03)
    num_train_epochs: Optional[int] = field(default=0, metadata={"help": "Total number of training epochs to perform."})
    model_name: Optional[str] = field(default="llama3")
    dataset_name: Optional[str] = field(default="nqswap")
    is_adaptive_kl: Optional[bool] = field(default=False)
    target: Optional[float] = field(default=5.0)
    task_type: Optional[str] = field(default="qa", metadata={"help": "The type of task. qa or summary"})
    max_new_tokens: Optional[int] = field(default=400, metadata={"help": "The number of new tokens."})
    choose_cd: Optional[bool] = field(default=False, metadata={"help": "Choose generate strategy."})
    choose_dola: Optional[bool] = field(default=False, metadata={"help": "Choose generate strategy."})
    profile: Optional[bool] = field(default=False, metadata={"help": "Profile the training and inference process."})  

# Calculate the evaluation metric: factkb, rouge, bertscore
def evaluate_summary(index2ex, output_data, tokenizer_factkb, factkb, bertscore, rouge):
    """
        The method of calculating metric refers to previous work:
        https://github.com/xhan77/context-aware-decoding
        output_data:
        {
            tokens:[token_id,,,]
            string: [decode result]
            input_index: [index]
            output_index = input_index
        }
    """
    all_gold = []
    all_pred = []
    all_doc = []
    all_fact_score = []
    all_summac_zs1 = []
    all_summac_conv1 = []
    cov_em_all = []
    category2em = defaultdict(list)
    id2ex_output = {}
    for i, output in enumerate(output_data):
        pred = output["string"][0]     
        gold = index2ex[i]["gold_answers"] 
        all_gold.append(gold)
        all_pred.append(pred)

        article = index2ex[i]["article"]
        summary = pred
        inputs = [[summary, article]]
        tokens = tokenizer_factkb(inputs, return_tensors="pt", padding="max_length", truncation=True)
        result = torch.softmax(factkb(**tokens).logits, dim = 1)

        fact_score = result[0][1].item()

        all_fact_score.append(fact_score)
        all_doc.append(article)
        output_dict = index2ex[i].copy()
        output_dict["pred"] = pred
        id2ex_output[i] = output_dict

    results_rouge = rouge.compute(predictions=all_pred, references=all_gold)
    results = bertscore.compute(predictions=all_pred, references=all_doc, lang="en")
    bertscore = 0
    for k, v in results.items():
        if k == "precision":
            bertscore = statistics.mean(v)
    return statistics.mean(all_fact_score), results_rouge, bertscore

# Read data
def entity_data(dataset_path):
    raw_data = []
    with open(dataset_path) as f:
        for line in f:
            ex = json.loads(line)
            raw_data.append(ex)
    return raw_data

# Grid search
def grid_search(final_data, training_args, data_args, model_args, model, last_layer_params, tokenizer, factkb=None, bertscore=None, rouge=None, generate_kwargs=None, layer_num=None, param_to_optimize=None, data_collator=None):
    total = 0
    correct = 0
    error_examples = []
    result_list = []
    # Select the 20% examples for searching
    for idx in tqdm.tqdm((list(range(len(final_data)))[:data_args.num_data:5])):
        optimizer = torch.optim.AdamW(param_to_optimize, lr=training_args.learning_rate)
        trainer = NewTrainer(model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            optimizers=(optimizer, None),
        )
        trainer.set_profile(training_args)
        logger.info(f"kl: {training_args.kl_coeff}, lr: {training_args.learning_rate}, epoch: {training_args.num_train_epochs}")

        if training_args.is_adaptive_kl:
            trainer.set_adaptive_kl()
        input_dict, metric_reference = data_format(training_args, final_data[idx])
        doc_tokens = tokenizer.encode(input_dict["instruction"])
        doc_tokens = doc_tokens[:data_args.doc_token]
        context_doc = tokenizer.decode(doc_tokens)
        input_dict["instruction"] = context_doc

        # Call train_and_inference
        output_text = train_and_inference(model, model_args, trainer, tokenizer, input_dict, last_layer_params, training_args, generate_kwargs, layer_num=layer_num)
        total += 1
        model.model.last_input_dic = []
        logger.info(f"output_text: {colored(output_text, 'green')}")

        # Post-process the generated text
        if training_args.task_type == 'qa':
            if 'memotrap' in data_args.data_path or 'hotpot' in data_args.data_path:
                answers = [metric_reference["answers"]]
                normalized_answers = [normalize_text(an) for an in answers]
                output_answer = postprocess_text(output_text, data_args.data_path)
                normalized_output = normalize_text(output_answer)
                logger.info(f"normalized_answers: {colored(normalized_answers, 'blue')}\n" + \
                            f"normalized_output: {colored(normalized_output, 'red')}")
            else:
                # NQ or NQSwap
                answers = metric_reference["answers"]
                normalized_answers = [normalize_text(an) for an in answers]
                output_answer = postprocess_text(output_text, data_args.data_path)
                normalized_output = normalize_text(output_answer)
                logger.info(f"normalized_answers: {colored(normalized_answers, 'blue')}\n" + \
                            f"normalized_output: {colored(normalized_output, 'red')}")
            matched = any(normalized_answer in normalized_output for normalized_answer in normalized_answers)
            correct += matched
            logger.info(f"idx: {idx}, total {total}, correct {correct}")
            logger.info('Batch processed. Accuracy so far: %.2f%%', correct / total * 100)
        else:
            result_list.append({
                "input_index": idx,
                "assigned_model": training_args.model_name,   
                "string": [output_text]
            })
    # Return the result
    if training_args.task_type == 'qa':
        return correct / total 
    else:
        factkb, rouge, bert_p = evaluate_summary(final_data, result_list, tokenizer_factkb, factkb, bertscore, rouge)
        metric = {
            "factkb": factkb,
            "rouge-L": rouge,
            "bert_p": bert_p,
        }
        return metric
    


if __name__ == "__main__":
    # --------------------------------Arguments--------------------------------
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # --------------------------------Initialization--------------------------------
    if training_args.task_type == 'summary':
        # Need to export HF_ENDPOINT=https://hf-mirror.com 
        tokenizer_factkb = transformers.AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True)
        factkb = transformers.AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", num_labels = 2)

        bertscore = evaluate.load("../eval/evaluate/metrics/bertscore")
        rouge = evaluate.load('../eval/evaluate/metrics/rouge')

    # Set the search range
    lr_list = [1e-6, 3e-6, 1e-5, 3e-5]
    epoch_list = [10, 20, 50]
    kl_w_list = [0.01, 0.03, 0.1, 0.3, 1]

    matrix = []
    best_lr, best_epoch, best_kl = None, None, None


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    data_collator = DataCollatorForPretrainDataset(tokenizer=tokenizer)

    res = []
    max_acc = 0
    early_exit_layers = [16,18,20,22,24,26,28,30,32]
    print(f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")

    mature_layer = early_exit_layers[-1]
    premature_layer = None
    candidate_premature_layers = early_exit_layers[:-1]
    premature_layer_dist = {l:0 for l in candidate_premature_layers}
    repetition_penalty = 1.2
    generate_kwargs = dict(repetition_penalty=repetition_penalty, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers)

    # --------------------------------Data--------------------------------
    if training_args.task_type == 'qa':
        final_data = jload(data_args.data_path)
    else:
        final_data = []
        with open(data_args.data_path, 'r', encoding="utf-8") as f:
            for line in f:
                proc_line = line.strip()
                if proc_line:
                    data = json.loads(proc_line)
                    doc_tokens = tokenizer.encode(data["article"])
                    doc_tokens = doc_tokens[:data_args.doc_token]
                    context_doc = tokenizer.decode(doc_tokens)
                    final_data.append({
                        "gold_answers": data["summary"],
                        "article": context_doc,
                        "context_string": context_doc,
                    })
    os.makedirs(data_args.output_path, exist_ok=True)

    # --------------------------------Grid Search--------------------------------
    for idx_lr, lr in enumerate(lr_list):
        for idx_a, epoch in enumerate(epoch_list):
            for idx_k, kl in enumerate(kl_w_list):
                training_args.kl_coeff = kl
                training_args.learning_rate = lr
                training_args.num_train_epochs = epoch
                param_to_optimize = []
                if training_args.model_name == 'llama3':
                    model = LlamaModel_ours.from_pretrained(
                        model_args.model_name_or_path,
                        cache_dir=training_args.cache_dir,
                        device_map = 'auto'
                    )
                    if '8B' in model_args.model_name_or_path:
                        layer_num = 32
                    elif '70B'  in model_args.model_name_or_path:
                        layer_num = 80
                elif training_args.model_name == 'qwen':
                    model = Qwen2Model_ours.from_pretrained(
                        model_args.model_name_or_path,
                        cache_dir=training_args.cache_dir,
                        device_map = 'auto'
                    )
                    if '4B' in model_args.model_name_or_path:
                        layer_num = 40
                    elif '7B' in model_args.model_name_or_path:
                        layer_num = 32
                    elif '14b'  in model_args.model_name_or_path:
                        layer_num = 40
                model.set_any_layer_num(model_args.train_layer)
                last_layer_params = {}
                train_list = []
                for i in range(layer_num - model_args.train_layer, layer_num):
                    train_list.append("model.layers.{}.mlp.gate_proj.weight".format(i))
                    train_list.append("model.layers.{}.mlp.up_proj.weight".format(i))
                    train_list.append("model.layers.{}.mlp.down_proj.weight".format(i))
                with torch.no_grad():
                    for name, item in model.named_parameters():
                        if name in train_list:
                            last_layer_params[name] = item.data.detach().clone()
                            item.requires_grad = True
                            param_to_optimize.append(item)
                        else:
                            item.requires_grad = False

                

                logger.info(f"lr: {lr}, epoch: {epoch}, kl: {kl}")
                if training_args.task_type == 'qa':
                    result = grid_search(
                        final_data, training_args, data_args, model_args, model, last_layer_params, tokenizer, generate_kwargs=generate_kwargs, layer_num=layer_num, param_to_optimize=param_to_optimize, data_collator=data_collator
                        )
                else:
                    result = grid_search(
                        final_data, training_args, data_args, model_args, model, last_layer_params, tokenizer, factkb, bertscore, rouge, generate_kwargs, layer_num=layer_num, param_to_optimize=param_to_optimize, data_collator=data_collator
                        )
                if training_args.task_type == 'qa':
                    if  result > max_acc:
                        best_kl = kl
                        best_epoch = epoch
                        best_lr = lr
                        max_acc = result
                else:
                    # As for summary task, we need to calculate and save the evaluation metric
                    with open(data_args.output_path + "/res_search_{}_{}_{}.jsonl".format(training_args.task_type, training_args.model_name, training_args.dataset_name), "a+") as f:
                        f.write(json.dumps({
                            "lr": lr,
                            "epoch": epoch,
                            "kl": kl,
                            "result": result
                        }))
                        f.write("\n")
    # Print the best hyperparameters
    if training_args.task_type == 'qa':
        logger.info(f"Best best_lr: {best_lr}, best_kl: {best_kl}, best_epoch: {best_epoch}, best_acc: {max_acc}")