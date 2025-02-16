# Trustworthy ML (ECE 696B) Project 1 Fine-Tuning LLMs


This project includes evaluation and LoRA fine-tuning of [OPT](https://huggingface.co/facebook/opt-1.3b) models on [wikitext2](https://huggingface.co/datasets/Salesforce/wikitext).

---
# Requirements
- Torch
- cuda=12.1
- **Transformers >= 4.46.0**

# Clone and install the dependencies
```
git clone https://github.com/SqueezeAILab/SqueezeLLM
cd TRUSTWORTHYAI_P1
pip install -r requirements.txt
```
# How to use
## 1. Downloading [OPT](https://huggingface.co/facebook/opt-1.3b) models and set up the folder path.

## 2. OPT Perplexity Evaluation
```
python evaluate.py opt <your_model_path> wikitext2 --torch_profile
```
## 3. OPT sample outputs
You can type your own context to obtain model outputs. For example
```
python instruction_output.py <your_model_path> --context "What are we having for dinner?"
```
## 4. LoRA fine-tuning
First, you can run `lora_finetuning.py` after setting your model path and output path
```
python lora_finetuning.py <your_model_path> --output_dir <your_ouput_path> --dataset wikitext2
```
Then, you can merge LoRA weights to obtain the final model, just run
```
python merge_lora.py <your_model_path> --loraweights <your_lora_weights_path> --output_dir <your_final_model_path>
```
Finally, you can evaluate the finetuned model by running `evaluate.py`
```
python evaluate.py opt <your_final_model_path> wikitext2 --torch_profile
```

The code was tested on two RTX4090 GPUs with Cuda 12.1.

---
## Acknowledgement

This code reuses components from several libraries including [GPTQ](https://github.com/IST-DASLab/gptq) as well as [GPTQ-For-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa/).
