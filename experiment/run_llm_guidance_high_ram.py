import json
import os
import logging
import torch
import pandas as pd

from argparse import ArgumentParser
from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from llm_guidance_high_ramDTO import ScriptArguments
from models import get_model
from src import evaluator
from huggingface_hub import login

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    
    login(os.environ.get("HF_READ_TOKEN"))

    model_info = {
        "model_name_or_path": "gpt2",
        "model_family": "gpt2",
        "torch_dtype": "bfloat16",
        "max_new_tokens": 512,
        "do_sample": True,
        "top_p": 0.6,
        "temperature": 0.9,
        "use_cache": True,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "guidance_scale": 1.05,
        "save_path": "results/",
        "eval_template": "plain",
        "safety_bench": "test",
        "num_of_reps": 1,
    }

    parser = ArgumentParser((ScriptArguments, ModelConfig))
    args, model_config = parser.parse_dict(model_info)

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    
    print(f"torch_dtype: {torch_dtype}")
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model, tokenizer = get_model.get_model(model_config.model_name_or_path, model_kwargs, model_family=args.model_family, padding_side="left")
    model.eval()
    
    eval_template = evaluator.common_eval_template[args.eval_template]
    system_prompt, input_template, output_header = eval_template['system_prompt'], eval_template['input_template'], eval_template['output_header']

    if args.prefill_prefix is not None and args.num_perfix_tokens > 0:
        raise ValueError("prefill_prefix and num_perfix_tokens should not be used together")

    if args.prefill_prefix is not None:
        output_header = args.prefill_prefix
    
    if args.num_perfix_tokens > 0 and (args.safety_bench not in ["hex-phi_with_refusal_prefix", 'hex-phi_with_harmful_prefix']):
        raise ValueError("num_perfix_tokens should only be used with hex-phi_with_refusal_prefix or hex-phi_with_harmful_prefix")

    
    log = evaluator.eval_safety_in_batch(model, args.prompt_style, tokenizer, num_prefix_tokens = args.num_perfix_tokens, 
                save_path = args.save_path, batch_size_per_device = args.batch_size_per_device,
                bench = args.safety_bench, evaluator = args.evaluator,
                system_prompt = system_prompt, input_template = input_template, output_header = output_header,
                max_new_tokens = args.max_new_tokens, 
                do_sample = args.do_sample, top_p = args.top_p, temperature = args.temperature, use_cache = args.use_cache, top_k = args.top_k,
                repetition_penalty = args.repetition_penalty, length_penalty = args.length_penalty, guidance_scale= args.guidance_scale,
                num_of_reps = args.num_of_reps)
    
    
    save_path = f"{args.save_path}{args.model_family}/{args.safety_bench}/{'guidance_scale-' + str(args.guidance_scale) if args.guidance_scale > 1.0 else 'no_guidance'}/"

    log['hyperparameters'] = model_info
    results = log.pop('results')


    os.makedirs(os.path.dirname(save_path + 'log.json'), exist_ok=True)
    with open(save_path + 'log.json', 'w') as f:
        json.dump(log, f)

    answer_index = -1
    prompt_index = 1 if results[0][0]['role'] == 'system' else 0

    rows = []
    for result in results:
        Q, A = result[prompt_index], result[answer_index]
        rows.append({'input': Q['content'], 
                     'output': A['content'], 
                     'user_role': Q['role'], 
                     'model_role': A['role'], 
                     'category': Q['category'], 
                     'score': None},
                     )
    
    result_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path + 'results.csv'), exist_ok=True)
    result_df.to_csv(save_path + 'results.csv', index=False)