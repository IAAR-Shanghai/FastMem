import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union
import regex as re


import tqdm
import copy

# Profile the training and generation process
def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Starting snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
   )

def stop_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Stopping snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot(r_='-1') -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not exporting memory snapshot")
       return

   # Prefix for file names.
   from datetime import datetime, timedelta
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"/mnt/data132/shuochen/code/stanford_alpaca/result/memory70b/{r_}_{host_name}_{timestamp}"

   try:
       logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
       torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
   except Exception as e:
       logger.error(f"Failed to capture memory snapshot {e}")
       return

def data_format(
        train_args,
        data
): 
    """
        The function formats the data for the model.
        Args:
            train_args: training arguments.
            data: input data.
        Returns:
            A dictionary containing the input data and metric reference.
    """
    if train_args.task_type == 'qa':
        if "squad" == train_args.dataset_name:
            input_dict = {
                "instruction": data["context"],
                "input" : "",
                "question": data["question"]
            }
            metric_reference = {
                "answers": data["answers"],
                "id": data["id"]
            }
        elif "nqswap" == train_args.dataset_name:
            input_dict = {
                "instruction": data["sub_context"],
                "input" : "",
                "question": data["input"] + "?"
            }
            metric_reference = {
                "answers": [data["sub_answer"]]
            }
        elif "nq" == train_args.dataset_name:
            input_dict = {
                "instruction": data["context"],
                "input" : "",
                "question": data["input"] + "?"
            }
            metric_reference = {
                "answers": [data["answer"]]
            }
        elif "nq_icl" == train_args.dataset_name:
            icl = '''Please extract from the context the span that best answers the question and provide the answer in the following format: "The answer is: ...".\n'''
            instruction = icl + '''Context: {context}'''.format_map(data)
            input_dict = {
                "instruction": instruction,
                "input" : "",
                "question": data["input"] + "?"
            }
            metric_reference = {
                "answers": [data["answer"]]
            }
        elif "nqswap_icl" == train_args.dataset_name:
            icl = '''Please extract from the context the span that best answers the question and provide the answer in the following format: "The answer is: ...".\n'''
            instruction = icl + '''Context: {sub_context}'''.format_map(data)
            input_dict = {
                "instruction": instruction,
                "input" : "",
                "question": data["input"] + "?"
            }
            metric_reference = {
                "answers": [data["sub_answer"]]
            }
        elif "memotrap" == train_args.dataset_name:
            split_prompt = data["prompt"].split(":")
            if len(split_prompt) == 2:
                instruction, sentence = split_prompt
            elif len(split_prompt) > 2:
                instruction = split_prompt[0]
                sentence = ":".join(split_prompt[1:])
            else:
                raise ValueError("Prompt is not in the correct format.")
            instruction = instruction.strip() + "."
            sentence = sentence.strip()
            classes = ast.literal_eval(data["classes"])
            classes = [c.strip().rstrip(string.punctuation) for c in classes]
            input_dict = {
                "instruction": instruction,
                "input" : "",
                "question": sentence + f'\nCandidates: "{classes[0]}" or "{classes[1]}"'
            }
            metric_reference = {
                "answers": classes[data["answer_index"]],
                "id": None
            }
        elif "hotpot" in train_args.dataset_name.lower():
            paragraphs = [" ".join(data["context"]["sentences"][i]) for i in range(len(data["context"]["sentences"]))]
            input_dict = {
                "instruction": "\n".join(paragraphs),
                "input" : "",
                "question": data["question"].rstrip(string.punctuation) + "?"
            }
            metric_reference = {
                "answers": data["answer"],
                "id": data["id"]
            }
        else:
            raise ValueError(f"Unsupported dataset name: {train_args.dataset_name}")
    else:
        input_dict = {
            "instruction": data["article"],
            "input": "",
            "question": data["context_string"]
        }
        metric_reference = {
            "answers": data["gold_answers"]
        }

    return input_dict, metric_reference


# Post-process the generated text
def normalize_text(text):
    # Convert to lowercase to make the comparison case-insensitive
    text = text.lower()
    # Remove articles: 'a', 'an', 'the'
    text = re.sub(r'\b(a|an|the)\b', '', text)
    # Normalize whitespace, collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[\p{P}\p{S}]', '', text)
    # Normalize numeric expressions (e.g., removing commas in numbers)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    # Strip leading/trailing whitespace that might be left after other replacements
    text = text.strip()
    return text


def postprocess_text(preds, data_name):
    
    pattern = r'(?i)(?:the answer is:\s*)'
    matches = re.split(pattern, preds)
    # Check if any matches were found and get the last one
    if len(matches) > 1:
        return matches[-1].strip()
    else:
        return "Unknown"


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

