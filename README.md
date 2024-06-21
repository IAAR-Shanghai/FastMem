<h1 align="center">
   <img src="./assets/memorization.png" alt="" width="4%"> FastMem: Fast Memorization of Prompt Improves Context Awareness of Large Language Models
</h1>
This is the official cod of the paper [`FastMem: Fast Memorization of Prompt Improves Context Awareness of Large Language Models`]()
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

The required dependencies and their versions can be found in the [`requirements.txt`](requirements.txt).

To install all packages in this codebase along with their dependencies, run
```sh
pip install -e .
```

## Run



## Acknowledgement
Our code have been developed based on [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca). We thank this valuable work.


## Contact Us

* Junyi Zhu: junyizhu.ai@gmail.com
* Shuochen Liu: liusc@iaar.ac.cn
* Bo Tang: tangb@iaar.ac.cn
