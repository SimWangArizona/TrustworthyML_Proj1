from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM,LlamaForCausalLM, Trainer,TrainingArguments,OPTForCausalLM,AutoModel
import torch

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM, AutoModel, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model,torch_dtype=torch.float16)
    model.seqlen = model.config.max_position_embeddings
    return model



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="opt model to load")
    parser.add_argument("--loraweights", type=str, default="",help="lora weights to load")
    parser.add_argument("--output_dir", type=str, default="", help="final model dir")
    args = parser.parse_args()

    original_model = get_opt(model=args.model)
    lora_model = PeftModel.from_pretrained(original_model, args.loraweights)
    pretrained = lora_model.merge_and_unload()
    pretrained.save_pretrained(args.output_dir)

