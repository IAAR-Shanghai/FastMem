import ast
import copy
import json
import socket
import string
import sys
import time
import traceback

import regex as re
import torch
import torch.nn.functional as F
import torch.nn.parallel
import tqdm
import transformers
from dataclasses import dataclass, field
from termcolor import colored
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer import *
from typing import Dict, List, Optional, Sequence

from utils import *
from NewModel_Llama_any_layer import *
from NewModel_Qwen_any_layer import *
from Trainer import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S",)
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100 # Index for ignoring tokens
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000 # Maximum number of memory events per snapshot

# Prompts used for Llama and Qwen models across various datasets
PROMPT_DICT = {
    "prompt_input": (
        "{instruction}"
    ),
    # Pretrain-like phase
    "pretrain_prompt_pretrain_assistant_llama3": (
        '''<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n{instruction}<|eot_id|>'''
    ),
    "pretrain_prompt_pretrain_assistant_qwen": (
        '''<|im_start|>assistant\n{instruction}<|im_end|>'''
    ),
    # Inference phase: Q&A
    "inference_prompt_llama3_nq": (
        '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease extract from the context the span that best answers the question and provide the answer in the following format: "The answer is: ...".\nContext: {instruction}\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
    ),
    "inference_prompt_llama3_nqswap": (
        '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease extract from the context the span that best answers the question and provide the answer in the following format: "The answer is: ...".\nContext: {instruction}\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
    ),
    "inference_prompt_llama3_hotpot": ( 
        '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease extract from the context the span that best answers the question and provide the answer in the following format: "The answer is: ...".\nContext: {instruction}\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
    ),
    "inference_prompt_llama3_memotrap": ( 
        '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease choose the candidate that best fits the instructions and provide the answer in the following format: "The answer is: ...".\nInstruction: {instruction}\nSentence: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
    ),
    "inference_prompt_qwen_nq": (
        '''<|im_start|>user\nPlease extract from the context the span that best answers the question and provide the answer in the following format: "The answer is: ...".\nContext: {instruction}\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n'''
    ),
    "inference_prompt_qwen_nqswap": (
        '''<|im_start|>user\nPlease extract from the context the span that best answers the question and provide the answer in the following format: "The answer is: ...".\nContext: {instruction}\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n'''
    ),
    "inference_prompt_qwen_nq_icl": (
        '''<|im_start|>user\n{instruction}\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n'''
    ),
    "inference_prompt_qwen_nqswap_icl": (
        '''<|im_start|>user\n{instruction}\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n'''
    ),
    "inference_prompt_qwen_hotpot": ( 
        '''<|im_start|>user\nPlease extract from the context the span that best answers the question and provide the answer in the following format: "The answer is: ...".\nContext: {instruction}\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n'''
    ),
    "inference_prompt_qwen_memotrap": ( 
        '''<|im_start|>user\nPlease choose the candidate that best fits the instructions and provide the answer in the following format: "The answer is: ...".\nInstruction: {instruction}\nSentence: {question}<|im_end|>\n<|im_start|>assistant\n'''
    ),
    # Inference phase: Summarization
    "inference_prompt_llama3_xsum": (
        '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nArticle: {instruction}\nSummarize the above article in 1 sentence.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
    ),
    "inference_prompt_llama3_cnndm": (
        '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nArticle: {instruction}\nSummarize the above article in 3 sentences.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
    ),
    "inference_prompt_llama3_wikihow": (
        '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nArticle: {instruction}\nSummarize the above article in few steps using concise verb-object phrases directly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
    ), 
}

# Arguments: Algorithm Hyperparameter Settings
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Meta-Llama-3-8B-Instruct", metadata={"help": "The path to the model."})
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
    kl_coeff: Optional[float] = field(default=0.1, metadata={"help": "The KL coefficient."})
    num_train_epochs: Optional[int] = field(default=10, metadata={"help": "Total number of training epochs to perform."})
    model_name: Optional[str] = field(default="llama3")
    dataset_name: Optional[str] = field(default="nqswap")
    is_adaptive_kl: Optional[bool] = field(default=False, metadata={"help": "Use adaptive KL."})
    target: Optional[float] = field(default=5.0, metadata={"help": "The target KL value."})
    task_type: Optional[str] = field(default="qa", metadata={"help": "The type of task. qa or summary"})
    max_new_tokens: Optional[int] = field(default=400, metadata={"help": "The number of new tokens."})
    choose_cd: Optional[bool] = field(default=False, metadata={"help": "Choose generate strategy."})
    choose_dola: Optional[bool] = field(default=False, metadata={"help": "Choose generate strategy."})
    profile: Optional[bool] = field(default=False, metadata={"help": "Profile the training and inference process."})  


# The following functions are used for the pretrain-like phase.
# Pretrain-like phase: Tokenize reference data
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """
        The function tokenizes the reference data.
        Args:
            strings: reference data.
            tokenizer: tokenizer of specific model.
        Returns:
            A dictionary containing the tokenized input_ids, labels, input_ids_lens, and labels_lens.
            e.g., {'input_ids': [tensor([128000, 128006,  78191, 128007,    271,    791,  11999,   3280,    315,                                                                                                                                                                                                                
                    10780,   6785,     11,    459,   3778,  20156,  12707,   4101,    449,                                                                                                                                                                                                                               
                    11145,  17276,  23373,  26296,     11,    323,  24190,  43423,  14433,                                                                                                                                                                                                                               
                    300,     11,   8096,  16835,     83,     11,    323,  13678,  12225,                                                                                                                                                                                                                               
                        76,  43280,     11,    574,  11713,    389,   7552,    220,     20,                                                                                                                                                                                                                               
                        11,    220,    679,     20,     11,    555,  24426,     11,    323,                                                                                                                                                                                                                               
                    85170,    389,   6664,    220,   1032,     11,    220,    679,     20,                                                                                                                                                                                                                               
                    323,  20536,    389,   3297,    220,   1114,     11,    220,    679,                                                                                                                                                                                                                               
                        21,     13,    578,   3280,  13282,    220,   1419,  18243,     13,                                                                                                                                                                                                                               
                    128009])], 
                    'labels': [tensor([128000, 128006,  78191, 128007,    271,    791,  11999,   3280,    315,                                                                                                                                                                                                 
                    10780,   6785,     11,    459,   3778,  20156,  12707,   4101,    449,                                                                                                                                                                                                                               
                    11145,  17276,  23373,  26296,     11,    323,  24190,  43423,  14433,                                                                                                                                                                                                                               
                    300,     11,   8096,  16835,     83,     11,    323,  13678,  12225,                                                                                                                                                                                                                               
                        76,  43280,     11,    574,  11713,    389,   7552,    220,     20,
                        11,    220,    679,     20,     11,    555,  24426,     11,    323,
                    85170,    389,   6664,    220,   1032,     11,    220,    679,     20,
                    323,  20536,    389,   3297,    220,   1114,     11,    220,    679,
                        21,     13,    578,   3280,  13282,    220,   1419,  18243,     13,
                    128009])], 
                    'input_ids_lens': [82], 
                    'labels_lens': [82]}
    """
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.shape[-1] for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

# Pretrain-like phase: Preprocess the data
def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    name
) -> Dict: 
    """
        Preprocess the data by formatting the inputs.
    """
    if name == 'llama3':
        sources = [PROMPT_DICT['pretrain_prompt_pretrain_assistant_llama3'].format_map({'instruction': sources[0]})]
        prompt_length = len(tokenizer("<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n")['input_ids'])
    elif name == 'qwen':
        sources = [PROMPT_DICT['pretrain_prompt_pretrain_assistant_qwen'].format_map({'instruction': sources[0]})]
        prompt_length = len(tokenizer("<|im_start|>assistant\n")['input_ids'])
    else:
        assert False, f"Unknown model name: {name}"
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = sources_tokenized["input_ids"]
    labels = input_ids.copy()
    return dict(input_ids=input_ids, labels=labels, context_len=[[prompt_length, sources_tokenized["input_ids_lens"][0]]])

# Pretrain-like phase: Dataset for tuning
class PretrainDataset(Dataset):
    """
        Args:
            data_dict: e.g., {"instruction": "instruction", "question": "question"}
            tokenizer: tokenizer of specific model.
            model_name: either llama3 or qwen.
        Returns:
            A dataset for tuning.
    """
    def __init__(self, data_dict: dict, tokenizer: transformers.PreTrainedTokenizer, model_name):
        super(PretrainDataset, self).__init__()
        logging.warning("Loading data...")
        logging.warning("Formatting inputs...")

        sources = [
            PROMPT_DICT["prompt_input"].format_map(data_dict),
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, tokenizer, model_name)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.context_len = data_dict["context_len"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], context_len=self.context_len[i])


# Pretrain-like phase: Data collator for tuning
@dataclass
class DataCollatorForPretrainDataset(object):
    """
        Collate examples for tuning.
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, context_len = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "context_len"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=IGNORE_INDEX
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=torch.ones_like(input_ids),
            context_len = context_len
        )

# Pretrain-like phase: Make dataset and collator
def make_Pretrain_data_module(tokenizer: transformers.PreTrainedTokenizer, data_dict, model_name) -> Dict:
    """
        Make dataset and collator
        Args:
            tokenizer: tokenizer of specific model.
            data_dict: e.g., {"instruction": "instruction", "question": "question"}
            model_name: either llama3 or qwen.
        Returns:
            A dictionary containing the PretrainDataset.
    """
    train_dataset = PretrainDataset(tokenizer=tokenizer, data_dict=data_dict, model_name=model_name)
    return dict(train_dataset=train_dataset)

"""
    The following functions are used for the inference phase.
"""
# Inference phase: Sampling for generate
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

# Inference phase: Next token prediction
@torch.inference_mode()
def generate(
            training_args,
            model_args,
            layer_num,
            last_layer_params,
            model,
            tokenizer,
            inputs_with_contexts,
            alpha: float = 0.5,
            temperature=0,
            top_p=0.9,
            max_length: int = 400,
            mature_layer=None, premature_layer=None, candidate_premature_layers=[],
            **kwargs,
            ) -> List[List[int]]:
    """
        Args:
            training_args: Training arguments.
            model_args: Model arguments.
            layer_num: The number of transformer blocks in the model.
            last_layer_params: Saved training parameters.
            inputs_with_contexts: A list of reference strings for training.
            max_length: Maximum truncation length for generated text.
            For CD:
                alpha: Coefficient for CD.
            For DoLa:
                mature_layer: The last Transformer layer.
                premature_layer: Preceding layer.
                candidate_premature_layers: Candidate premature layers.
                kwargs: Other arguments.
        Returns:
            response: The results of decoding the model outputs.
    """
    # Set the prompt
    dataset_name = training_args.dataset_name
    model_name = training_args.model_name
    prompt = "inference_prompt_{}_{}".format(model_name, dataset_name)
    if training_args.choose_cd:
        past_key_values_without_context = None
        last_layer_params_cad = {}
        train_list = []
        for i in range(layer_num - model_args.train_layer, layer_num):
            train_list.append("model.layers.{}.mlp.gate_proj.weight".format(i))
            train_list.append("model.layers.{}.mlp.up_proj.weight".format(i))
            train_list.append("model.layers.{}.mlp.down_proj.weight".format(i))
        with torch.no_grad():
            for name, item in model.named_parameters():
                if name in train_list:
                    last_layer_params_cad[name] = item.data.detach().clone()
    # Tokenize the input
    input_ids_with_context = torch.LongTensor(tokenizer.encode(PROMPT_DICT[prompt].format_map(inputs_with_contexts), add_special_tokens=True, max_length=tokenizer.model_max_length - max_length, truncation=True)).unsqueeze(0).to(model.device)
    history_decode_ids = None
    past_key_values_with_context = None
    # Generate
    with torch.no_grad():
        if training_args.choose_dola:
            outputs = model.generate(input_ids_with_context, training_args, max_new_tokens=max_length, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, dola_decoding=True,
                                    top_p=top_p, top_k=0, temperature=temperature, stopping_criteria=None, relative_top=0.1, 
                                    mature_layer=mature_layer, premature_layer=None, candidate_premature_layers=candidate_premature_layers, **kwargs,)
            sequence = outputs.sequences
            gen_squence = sequence[:, input_ids_with_context.shape[-1]:][0, :]
            sampled_sequences = tokenizer.decode(gen_squence, skip_special_tokens=True)
            response = sampled_sequences
            torch.cuda.empty_cache()
        else:
            for r_ in  range(max_length):
                if training_args.profile:
                    start_time = time.time()
                    start_record_memory_history()
                model_inputs_with_contexts = model.prepare_inputs_for_generation(input_ids_with_context, past_key_values=past_key_values_with_context)
                outputs_with_contexts = model(**model_inputs_with_contexts, output_hidden_states=False, epoch=0)
                next_token_logits_with_contexts = outputs_with_contexts.logits[:, -1, :]
                if training_args.profile:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    export_memory_snapshot(r_)

                    # Stop recording memory snapshot history
                    stop_record_memory_history()
                    if r_ == 0:
                        logger.info(f"prefix time: {execution_time} seconds")
                    else:
                        logger.info(f"step {r_} time: {execution_time} seconds")

                if training_args.choose_cd:
                    with torch.no_grad():
                        for name, item in model.named_parameters():
                            if name in train_list:
                                item.copy_(last_layer_params[name].clone())
                    model_inputs_without_contexts = model.prepare_inputs_for_generation(input_ids_with_context, past_key_values=past_key_values_without_context)
                    outputs_without_contexts = model(**model_inputs_without_contexts, output_hidden_states=False, epoch=0)
                    next_token_logits_without_contexts = outputs_without_contexts.logits[:, -1, :]

                    next_token_logits_with_contexts = (1 + alpha) * next_token_logits_with_contexts - alpha * next_token_logits_without_contexts
                    past_key_values_without_context = outputs_without_contexts.past_key_values
                    with torch.no_grad():
                        for name, item in model.named_parameters():
                            if name in train_list:
                                item.copy_(last_layer_params_cad[name].clone())
                # Next token prediction
                if temperature > 0:
                    probs = torch.softmax(next_token_logits_with_contexts / temperature, dim=-1)
                    real_token_ids_list = sample_top_p(probs, top_p)
                else:
                    real_token_ids_list = torch.argmax(next_token_logits_with_contexts, dim=-1).view(1, 1)


                # Concatenate the generated token to the input
                input_ids_with_context = torch.cat((input_ids_with_context, real_token_ids_list), dim=1)
                past_key_values_with_context = outputs_with_contexts.past_key_values
                if history_decode_ids is None:
                    history_decode_ids = real_token_ids_list
                else:
                    history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)

                if real_token_ids_list[0][-1].item() in model.generation_config.eos_token_id:
                    break

            sampled_sequences = tokenizer.batch_decode(history_decode_ids.clone().detach().to('cpu'), skip_special_tokens=True)
            response = sampled_sequences[0]

    return response


# Train and inference phase
def train_and_inference(model, model_args, trainer, tokenizer, data_dict, last_layer_params, training_args, generate_kwargs=None, layer_num=None):
    """
        Args:
            model_args: model arguments.
            trainer: trainer initialization.
            data_dict: dictionary of processed data.
            last_layer_params: Saved training parameters.
            training_args: training arguments.
            generate_kwargs: DoLa generate arguments.
            layer_num: the number of training layers.
        Returns:
            response: The results of decoding the model outputs.
    """
    data_module = make_Pretrain_data_module(tokenizer=tokenizer, data_dict=data_dict, model_name=training_args.model_name)
    # When model size is 70B, we need to use trainer to set multi-GPU execution
    if training_args.num_train_epochs > 0 or '70B' in model_args.model_name_or_path:
        trainer.train_dataset = data_module['train_dataset']
        # Flush the trainer
        trainer.set_ref()
        # Set the training parameters
        train_list = []
        for i in range(layer_num - model_args.train_layer, layer_num):
            train_list.append("model.layers.{}.mlp.gate_proj.weight".format(i))
            train_list.append("model.layers.{}.mlp.up_proj.weight".format(i))
            train_list.append("model.layers.{}.mlp.down_proj.weight".format(i))
        with torch.no_grad():
            for name, item in model.named_parameters():
                if name in train_list:
                    item.copy_(last_layer_params[name])
                    
        # Train
        logger.info("Start training...")
        trainer.ref = None
        model.train()
        model.mode = 'train'
        trainer.train()
    else:
        model.cuda()
    model.mode = 'test'
    # Generate
    logger.info("Start generating...")
    model.eval()
    with torch.no_grad():
        inputs_with_contexts = {
            "instruction": data_dict["instruction"],
            "question": data_dict["question"]
        }
        responses = generate(training_args, model_args, layer_num, last_layer_params, model, tokenizer, inputs_with_contexts, temperature=0, max_length=training_args.max_new_tokens, **generate_kwargs)
    torch.cuda.empty_cache()
    return responses



if __name__ == "__main__":
    # --------------------------------Arguments--------------------------------
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    early_exit_layers = [16,18,20,22,24,26,28,30,32]
    if training_args.choose_dola:
        logger.info(f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")

    mature_layer = early_exit_layers[-1]
    premature_layer = None
    candidate_premature_layers = early_exit_layers[:-1]
    premature_layer_dist = {l:0 for l in candidate_premature_layers}
    repetition_penalty = 1.2
    generate_kwargs = dict(repetition_penalty=repetition_penalty, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers)
    # --------------------------------Initialization--------------------------------
    total = 0
    unknown_count = 0
    correct = 0
    currect_context_mean_len = 0
    miss_context_mean_len = 0
    param_to_optimize = []
    trainer = None
    optimizer = None
    last_layer_params = {}
    train_list = []
    # --------------------------------Model and Tokenizer--------------------------------
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
        elif '14B'  in model_args.model_name_or_path:
            layer_num = 40
    else:
        assert False, f"Unknown model name: {training_args.model_name}"
    # Set the number of layers to train
    model.set_any_layer_num(model_args.train_layer)
    if training_args.num_train_epochs > 0 or '70B' in model_args.model_name_or_path:
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
    # Set the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    # --------------------------------Data--------------------------------
    data_collator = DataCollatorForPretrainDataset(tokenizer=tokenizer)
    if training_args.task_type == 'qa':
        final_data = jload(data_args.data_path)
    elif training_args.task_type == 'summary':
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
    else:
        assert False, f"Unknown task type: {training_args.task_type}"
    logger.info(f"Data size: {len(final_data)}")
    # --------------------------------Train and Evaluate--------------------------------
    flag = 0
    for idx, data_idx in tqdm.tqdm(enumerate((list(range(len(final_data)))[:data_args.num_data]))):
        # Set the optimizer and trainer
        if training_args.num_train_epochs > 0 or '70B' in model_args.model_name_or_path:
            optimizer = torch.optim.AdamW(param_to_optimize, lr=training_args.learning_rate) # flush the optimizer
            optimizer.zero_grad(set_to_none=True)
            trainer = NewTrainer(model=model,
                tokenizer=tokenizer,
                args=training_args,
                data_collator=data_collator,
                optimizers=(optimizer, None),
            )
            if training_args.is_adaptive_kl:
                trainer.set_adaptive_kl()
            is_init = True if data_idx == 0 else False
            trainer.is_init = is_init
            trainer.set_profile(training_args)
            model = trainer.model
        # Format the data
        input_dict, metric_reference = data_format(training_args, final_data[data_idx])
        doc_tokens = tokenizer.encode(input_dict["instruction"])
        if training_args.profile:  # if profile, only process the first data which is long enough to profile
            if len(doc_tokens) < 2500 and flag == 0:
                continue
            elif flag == 1:
                break
            else:
                doc_tokens = doc_tokens[:2500]
                flag = 1
        doc_tokens = doc_tokens[:data_args.doc_token]
        context_doc = tokenizer.decode(doc_tokens)
        input_dict["instruction"] = context_doc
        # Train and inference
        output_text = train_and_inference(model, model_args, trainer, tokenizer, input_dict, last_layer_params, training_args, generate_kwargs, layer_num=layer_num)
        total += 1
        # Post-process the generated text
        model.model.last_input_dic = {}
        logger.info(f"output_text: {colored(output_text, 'green')}")
        os.makedirs(data_args.output_path, exist_ok=True)
        if training_args.task_type == 'qa':
            if 'memotrap' == training_args.dataset_name or 'hotpot' in training_args.dataset_name:
                answers = [metric_reference["answers"]]
                normalized_answers = [normalize_text(an) for an in answers]
                output_answer = postprocess_text(output_text, data_args.data_path)
                normalized_output = normalize_text(output_answer)
                logger.info(f"normalized_answers: {colored(normalized_answers, 'blue')}\n" + \
                            f"normalized_output: {colored(normalized_output, 'red')}")
            else:  # NQ or NQSwap
                answers = metric_reference["answers"]
                normalized_answers = [normalize_text(an) for an in answers]
                output_answer = postprocess_text(output_text, data_args.data_path)
                if output_answer.lower() == "unknown": 
                    unknown_count += 1
                normalized_output = normalize_text(output_answer)
                logger.info(f"normalized_answers: {colored(normalized_answers, 'blue')}\n" + \
                            f"normalized_output: {colored(normalized_output, 'red')}")
            matched = any(normalized_answer in normalized_output for normalized_answer in normalized_answers)
            correct += matched
            logger.info(f"idx: {idx}, total {total}, correct {correct}")
            logger.info('Batch processed. Accuracy so far: %.2f%%', correct / total * 100)
            logger.info('Unknown so far: %.2f%%', unknown_count / total * 100) 
        else:
            if training_args.num_train_epochs > 0:
                mode = 'ours'
                if training_args.choose_cd:
                    mode = 'ours_cad'
                elif training_args.choose_dola:
                    mode = 'ours_dola'
            else:
                mode = 'baseline'
            with open(data_args.output_path + "/{}_{}_{}.jsonl".format(training_args.model_name, training_args.dataset_name, mode), 'a+', encoding='utf-8') as f:
                f.write(json.dumps({
                    "input_index": idx,
                    "assigned_model": "{}".format(training_args.model_name),   
                    "string": [output_text]
                }))
                f.write("\n")
            
