import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM,LlamaForCausalLM, Trainer,TrainingArguments, AutoTokenizer
import os
from utils.model_parse import (load_model,parse_model)
from utils.datautils import (get_loaders)

def train_model(initial_model_path,data_loader,lora_config, epochs=3,output_dir = ''):
    original_model = load_model(initial_model_path, model_type="opt")
    lora_model = get_peft_model(model=original_model,peft_config=lora_config)
    lora_model.bfloat16()

    training_args = TrainingArguments(
                output_dir=os.path.join(output_dir, f"epoch{epochs}"),
                evaluation_strategy="no",
                learning_rate=5e-6,
                per_device_train_batch_size=args.batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                deepspeed=None,
                local_rank=-1,
                save_strategy="epoch")

    trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=data_loader,
    data_collator=data_collator
    )

    trainer.train()
    lora_model.save_pretrained(os.path.join(output_dir,f"epoch{epochs}lora_weights"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="opt model to load")
    parser.add_argument("--model_type", type=str, default="opt", help="model type")

    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )

    parser.add_argument("--output_dir", type=str, default="", help="Trainer output dir")

    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs.",
    )

    parser.add_argument(
        "--dataset",
        default = "wikitext2",
        type = str,
        help="load which dataset",
    )
    parser.add_argument(
        "--finetuning_seqlen", type=int, default=1024, help="Seqlen of callibration data when finetuning."
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size of callibration data when finetuning."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=8, help="rank of lora layers"
    )
    parser.add_argument('--layers',
        default = 'all',
        choices=["all", "self_attn", "mlp"],
        type = str,
        help="Add LORA to specific layers")
    args = parser.parse_args()
    DEV = torch.device("cuda:0")
    # Build lora config
    if args.layers == 'all':
        target_modules = ["q_proj","k_proj","v_proj", "out_proj","fc1","fc2"]
    elif args.layers == 'self_attn':
        target_modules = ["q_proj","k_proj","v_proj", "out_proj"]
    else:
        target_modules = ["fc1","fc2"]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # load dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=2048)
    
    from datasets import load_dataset
    from transformers import DataCollatorForLanguageModeling

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_daset = tokenized_datasets["train"].shuffle(seed=0).select(range(128))
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False)
    
    train_model(initial_model_path=args.model,data_loader=train_daset,lora_config=lora_config,epochs=args.num_epochs,output_dir=args.output_dir)