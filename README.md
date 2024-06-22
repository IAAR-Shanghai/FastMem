<h1 align="center">
   <img src="./assets/memorization.png" alt="" width="4%"> FastMem: Fast Memorization of Prompt Improves Context Awareness of Large Language Models
</h1>
<p align="center">
    <a href="https://opensource.org/license/apache-2-0/">
        <img alt="License: Apache" src="https://img.shields.io/badge/License-Apache2.0-yellow.svg">
    </a>
    <a href="">
        <img alt="arXiv Paper" src="https://img.shields.io/badge/Paper-arXiv-red.svg">
    </a>
</p>

This is the official code of the paper **FastMem: Fast Memorization of Prompt Improves Context Awareness of Large Language Models**
 by *Junyi Zhu, *Shuochen Liu, Yuyu, Tangbo, Yibo Yan, Zhiyu Li, Feiyu Xiong, Tong Xu, Matthew B. Blaschko
 
If you find this repository or paper useful, you can cite
```
Will be uploaded to arXiv soon.
```
## Overview

TL;DR: **FastMem** maximizes the likelihood of the prompt before inference by fine-tuning only the last Feed-Forward Network (FFN) module. This targeted approach ensures efficient optimization without overfitting, significantly improving the model's ability to comprehend and accurately follow the context.

<p align="center"><img src="./assets/framework.png" alt="" width="80%"></p>

## Dependencies

Ensure you have Python 3.11.0+. 

The required dependencies and their versions can be found in the [`requirements.txt`](requirements.txt). The main packages are `pytorch`, `transformers` and `accelerate`.

To install all packages in this codebase along with their dependencies, run
```sh
pip install -e .
```

## Run

We evaluate **FastMem** on Q&A and summarization tasks that require LLMs to respond based on the given context. Our experiments include evaluations of popular open-source instruction fine-tuned LLMs from the Llama 3 series (8B and 70B) and the Qwen 1.5 series (4B, 7B, and 14B).


### Input format
The complete test dataset has been uploaded to the repository at the path [`./eval/data/`](./eval/data/). Below is an example input to our method (from NQ-Swap for Q&A and CNN-DM for summarization). 

**NQ-SWAP**
```json
{
     "input": "who sings love will keep us alive by the eagles",
     "context": "`` Love Will Keep Us Alive '' is a song written by Jim Capaldi , Paul Carrack , and Peter Vale , and produced by the Eagles , Elliot Scheiner , and Rob Jacobs . It was first performed by the Eagles in 1994 , during their `` Hell Freezes Over '' reunion tour , with lead vocals by bassist Timothy B. Schmit .",
     "answer": "Timothy B. Schmit",
     "sub_context": "`` Love Will Keep Us Alive '' is a song written by Jim Capaldi , Paul Carrack , and Peter Vale , and produced by the Eagles , Elliot Scheiner , and Rob Jacobs . It was first performed by the Eagles in 1994 , during their `` Hell Freezes Over '' reunion tour , with lead vocals by bassist Yuliya Snigir .",
     "sub_answer": "Yuliya Snigir"
},
```
**CNN-DM**
```json
{
     "article": "Fabio Borini may not have had much success climbing up the pecking order in the Liverpool attack but the Italian striker had no problems scaling the heights at Delamere Forest Park on Tuesday. The former Swansea striker made the most of the warm weather as he spent the day at adventure park Go Ape at the Cheshire forest. Borini appeared as a second-half substitute in Liverpool's 2-0 win against Newcastle at Anfield on Monday as clearly still had plenty of energy left as he was pictured taking part in a climbing exercise...",
     "summary": "Fabio Borini visited Go Ape adventure park in Delamere Forest on Tuesay . The Liverpool striker shared Instagram pictures from his day out . Borini came on as a substitute for Liverpool against Newcastle on Monday ."
}
```
### Running FastMem on Q&A and summarization tasks

All scripts are saved in [`./scripts`](./scripts). Run `bash run_qa.sh` or `bash run_summary.sh` to reproduce our results. When you need to test different models and tasks, you need to modify parameters such as `model_name`, `model_name_or_path`, `task_type`, `dataset_name` and `data_path` accordingly to conduct the tests. 

Given the characteristics of different models and datasets, testing under the best hyperparameters is needed to achieve optimal results. 
For example, the default parameters for the Llama3-8B-Instruct model are the optimal parameters for the Q&A task. As for summarization, you need to modify the path of results in `evaluate_summary.py` and run the code to calculate the metrics.


> [!Important]
> 1. If you want to use Contrastive Decoding(CD), set `choose_cd` to True.
> 2. If you want to use DOLA, set `choose_dola` to True and need to use `../src/transformers_generation/utils.py` to replace the file in the transformers library(`python3.11/site-packages/transformers/generation/utils.py`).
> 3. If you need to test our method on a new dataset or other scale models, you need to first obtain the corresponding optimal hyperparameters in the `search_qa.sh` or `search_summary.sh` script. The modifications regarding the model(e.g. `model_name`) and dataset(e.g. `dataset_name`) in the bash file are the same as mentioned above.

## Results for Experiment

<p align="center"><img src="./assets/experiment/qa.png" width="50%"></p>
<p align="center"><img src="./assets/experiment/different_model.png" width="50%"></p>
<p align="center"><img src="./assets/experiment/small_model.png" width="50%"></p>
<p align="center"><img src="./assets/experiment/summarization.png" alt=""></p>

More details and analyses about experimental results can be found in [`paper`]("").

## Acknowledgement
Our code have been developed based on [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca). We thank this valuable work.


## Contact Us

* Junyi Zhu: junyizhu.ai@gmail.com
* Shuochen Liu: liusc@iaar.ac.cn
* Bo Tang: tangb@iaar.ac.cn
