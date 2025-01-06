from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

default_system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information."

def initializer(model_name_or_path, model_kwargs, padding_side = "right"):

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, add_eos_token=False, add_bos_token=False)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = padding_side

    return model, tokenizer