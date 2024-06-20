import json
import argparse
from tqdm import tqdm
from pathlib import Path
import statistics
import json
from collections import defaultdict
import os
import evaluate
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import tqdm

tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True)
factkb = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", num_labels = 2)

def evaluate_qa(index2ex, eval_file):
    """
        output_data:
        {
            tokens:[token_id,,,]
            string: [decode result]
            input_index: [index]
            output_index = input_index
        }
    """
    print(eval_file)
    all_gold = []
    all_pred = []
    all_doc = []
    all_fact_score = []

    if os.path.exists(eval_file) == False:
        return 0
    with open(eval_file, "r") as f:
        output_data = [json.loads(line) for line in f] 
    cov_em_all = []
    category2em = defaultdict(list)
    id2ex_output = {}
    print("Start evaluating")
    for i, output in tqdm.tqdm(enumerate(output_data)):
        index = output["input_index"]
        pred = output["string"][0]     
        gold = index2ex[index]["summary"] 
        if pred == None:
            continue
        all_gold.append(gold)
        all_pred.append(pred)

        article = index2ex[index]["article"]
        summary = pred
        inputs = [[summary, article]]
        tokens = tokenizer(inputs, return_tensors="pt", padding="max_length", truncation=True)
        result = torch.softmax(factkb(**tokens).logits, dim = 1)

        fact_score = result[0][1].item()

        all_fact_score.append(fact_score)
        all_doc.append(article)
        output_dict = index2ex[index].copy()
        output_dict["pred"] = pred
        id2ex_output[i] = output_dict

    print("fact_score: ", statistics.mean(all_fact_score))
    rouge = evaluate.load('./evaluate/metrics/rouge')
    results = rouge.compute(predictions=all_pred, references=all_gold)
    print("rouge results: ", results)

    bertscore = evaluate.load("./evaluate/metrics/bertscore")
    results = bertscore.compute(predictions=all_pred, references=all_doc, lang="en")
    print("bertscore: ")
    for k, v in results.items():
        if k in ["precision", "recall", "f1"]:
            print(f"{k}: {statistics.mean(v)}")

# read data
def entity_data(dataset_path):
    raw_data = []
    with open(dataset_path) as f:
        for line in f:
            ex = json.loads(line)
            raw_data.append(ex)
    return raw_data

    



if __name__ == "__main__":
    # Note: Please export HF_ENDPOINT=https://hf-mirror.com first.
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/Summary/xsum_sample_test_3000.jsonl")
    parser.add_argument("--pred_path", type=str, default="path to result")
    args = parser.parse_args()

    data_path = args.data_path
    pred_path = args.pred_path
    index2ex = entity_data(data_path)  
    evaluate_qa(index2ex, pred_path)
    
    

