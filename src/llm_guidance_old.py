import time
import torch
import logging

from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers import LogitsProcessorList, StoppingCriteriaList

logging.basicConfig(level=logging.INFO)


class LLMPipeline:
    """
    Custom pipeline for large language model (LLM) inference. This pipeline provides an interface for handling
    text generation tasks with custom options for memory optimization, decoding strategies, and preprocessing.

    Args:
        model (PreTrainedModel):
            Pretrained LLM (e.g., GPT, OPT, BLOOM).
        tokenizer (PreTrainedTokenizer):
            Tokenizer associated with the LLM.
        scheduler (Optional[object]):
            (Optional) A scheduler to manage dynamic behaviors during inference.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, scheduler: object | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = scheduler

        self.unsafe_conceptss: str | None = 'hate, harassment, violence, suffering, humiliation, harm, suicide, ' \
                                        'sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, ' \
                                        'drug use, theft, vandalism, weapons, child abuse, brutality, cruelty'

        self.logger = self.get_logger()

        # Register the components
        self.register_modules(model=model, tokenizer=tokenizer, scheduler=scheduler)

    def get_logger(self):
        """Returns a logger for the pipeline."""
        logger = logging.getLogger("LLMPipeline")
        logger.setLevel(logging.INFO)
        return logger

    def register_modules(self, **kwargs):
        """Registers components of the pipeline."""
        for name, module in kwargs.items():
            setattr(self, name, module)

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, 'transformer'):
            # GPT-like models (e.g., GPT-2, GPT-J)
            return self.model.transformer.wte(input_ids)
        elif hasattr(self.model, 'encoder'):
            # T5-like models
            return self.model.encoder.embed_tokens(input_ids)
        elif hasattr(self.model, 'embeddings'):
            # BERT-like models
            return self.model.embeddings.word_embeddings(input_ids)
        elif hasattr(self.model, 'model'):
            # Assuming google/gemma-7b has a 'model' attribute for embeddings
            return self.model.model.embed_tokens(input_ids)
        elif hasattr(self.model, 'embed_tokens'):
            # LLaMA-like models
            return self.model.embed_tokens(input_ids)
        else:
            raise ValueError(f"Model {self.model_name} does not have accessible embeddings.")

    def apply_safety_guidance(self, 
                          embeddings: torch.Tensor, 
                          unsafe_concepts_embeddings: torch.Tensor, 
                          guidance_scale: float,
                          epsilon: float = 1e-8
                          ) -> torch.Tensor:
        """
        Apply safety guidance to the embeddings.

        Args:
            embeddings (torch.Tensor):
                Embeddings to apply safety guidance to.
            unsafe_concepts_embeddings (torch.Tensor):
                Embeddings of safety concepts.
            guidance_scale (float):
                Scale factor for the guidance.
            epsilon (float, optional):
                Small constant to avoid division by zero. Default: 1e-8.

        Returns:
            torch.Tensor: Guided embeddings.
        """
        # Normalize unsafe embeddings for stability
        unsafe_norm = unsafe_concepts_embeddings / (torch.norm(unsafe_concepts_embeddings, dim=-1, keepdim=True) + epsilon)

        # Compute projections for each token embedding onto each unsafe embedding
        projections = torch.einsum("bse,bne->bsne", embeddings, unsafe_norm) * unsafe_norm

        # Aggregate projections across the unsafe concepts dimension
        combined_projections = projections.sum(dim=2)  # Shape: [batch, seq, emb_dim]

        # Apply scaling and guidance to adjust user embeddings
        adjusted_embeddings = embeddings - guidance_scale * combined_projections

        return adjusted_embeddings

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        use_cache: bool = True,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        stopping_criteria: StoppingCriteriaList | None = None,
        logits_processor: LogitsProcessorList | None = None,
        guidance_scale: float = 0.0,
    ) -> str:
        """
        Perform text generation given an input prompt.

        Args:
            prompt (str):
                Input prompt for generation.
            max_length (int):
                Maximum length of the generated text. Defaults to 50.
            temperature (float):
                Sampling temperature. Lower values make output more deterministic. Defaults to 0.7.
            top_k (int):
                Limit the sampling pool to top-k tokens. Defaults to 50.
            top_p (float):
                Cumulative probability for nucleus sampling. Defaults to 0.9.
            do_sample (bool):
                Whether to perform sampling. Defaults to True.
            stopping_criteria (StoppingCriteriaList, optional):
                Custom stopping criteria for generation.
            logits_processor (LogitsProcessorList, optional):
                Custom logits processors for modifying output probabilities.

        Returns:
            str: Generated text.
        """
        
        enable_safety_guidance = True
        if guidance_scale < 1:
            enable_safety_guidance = False
            self.logger.warning('You have disabled safety guidance.')


        # Tokenize the input prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)
        input_embeddings = self.get_embeddings(input_ids)

        if enable_safety_guidance:
            unsafe_concepts_ids = self.tokenizer(self.unsafe_conceptss, return_tensors="pt").input_ids
            unsafe_concepts_ids = unsafe_concepts_ids.to(self.model.device)
            unsafe_concepts_embeddings = self.get_embeddings(unsafe_concepts_ids)

            input_embeddings = self.apply_safety_guidance(
                embeddings=input_embeddings,
                unsafe_concepts_embeddings=unsafe_concepts_embeddings,
                guidance_scale=guidance_scale,
            )

        time_start = time.time()

        # Pass embeddings through the model
        outputs = self.model.generate(
            # input_ids=input_ids,
            inputs_embeds=input_embeddings,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            use_cache=use_cache,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
        )

        print(f"Time taken: {time.time() - time_start:.2f} seconds")

        # Decode and return the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.logger.info("------------ start of generated text ------------")
        return generated_text

# Example Usage
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import login
    import os

    # login(os.environ.get("HF_READ_TOKEN"))

    # seed = 42
    # torch.cuda.manual_seed(seed)
    # torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a model and tokenizer
    model_name = "gpt2" #"EleutherAI/gpt-neo-2.7B"
    if device.type == "cpu":
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            use_auth_token=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=True
        )

    model = model.to(device)

    # Initialize the pipeline
    llm_pipeline = LLMPipeline(model=model, tokenizer=tokenizer)

    # Run text generation
    prompt = "please tell me a sentence about the usa"
    output = llm_pipeline(prompt, 
                          max_length=100, 
                          temperature=0.9,
                          top_k=50,
                          top_p=0.6,
                          do_sample=True,
                          use_cache=True,
                          repetition_penalty=1.0,
                          length_penalty=1.0,
                          stopping_criteria=None,
                          logits_processor=None,
                          guidance_scale=0.9)
    print(output)