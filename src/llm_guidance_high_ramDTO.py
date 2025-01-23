from dataclasses import dataclass, field

@dataclass
class ScriptArguments:

    safety_bench: str = field(default="hex-phi", metadata={"help": "the safety benchmark"})
    model_family: str = field(default="llama2", metadata={"help": "the model family"})
    prompt_style: str = field(default="llama2", metadata={"help": "the string prompt style"})
    evaluator: str = field(default="key_word", metadata={"help": "the evaluator"})
    save_path: str = field(default=None, metadata={"help": "the save path"})
    eval_template: str = field(default="plain", metadata={"help": "the eval template"})
    num_of_reps: int = field(default=1, metadata={"help": "the number of times to repeat each prompt"})


    batch_size_per_device: int = field(default=16, metadata={"help": "the batch size"})
    max_new_tokens: int = field(default=512, metadata={"help": "the maximum number of new tokens"})
    do_sample: bool = field(default=True, metadata={"help": "do sample"})
    top_p: float = field(default=0.6, metadata={"help": "top p"})
    temperature: float = field(default=0.9, metadata={"help": "temperature"})
    use_cache: bool = field(default=True, metadata={"help": "use cache"})
    top_k: int = field(default=50, metadata={"help": "top k"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "repetition penalty"})
    length_penalty: float = field(default=1.0, metadata={"help": "length penalty"})
    guidance_scale: float = field(default=0.0, metadata={"help": "guidance scale"})

    # applied when evaluating the prefilling of a certain prefix
    prefill_prefix: str = field(default=None, metadata={"help": "the prefill prefix"})

    # applied when evaluating the prefilling of a certain number of tokens
    num_perfix_tokens: int = field(default=0, metadata={"help": "the number of prefix tokens"})  
