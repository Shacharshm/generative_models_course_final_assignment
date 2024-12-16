from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers import LogitsProcessorList, StoppingCriteriaList
from typing import Optional, Union
import torch
import logging
import torch.nn.functional as F

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

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, scheduler: Optional[object] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = scheduler

        self.safety_concepts: Optional[str] = 'hate, harassment, violence, suffering, humiliation, harm, suicide, ' \
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
                              safety_concept_embeddings: torch.Tensor, 
                              guidance_scale: float,
                              safety_momentum: Optional[torch.Tensor] = None, 
                              guidance_threshold: float = 0.0,
                              guidance_momentum_scale: float = 0.0,
                              guidance_mom_beta: float = 0.0,
                              ) -> torch.Tensor:
        """
        Apply safety guidance to the embeddings.

        Args:
            embeddings (torch.Tensor):
                Embeddings to apply safety guidance to.
            safety_concept_embeddings (torch.Tensor):
                Embeddings of safety concepts.
            guidance_scale (float):
                Scale factor for the guidance.
            safety_momentum (torch.Tensor):
                Momentum from the previous step.
            guidance_threshold (float):
                Threshold for the guidance.
            guidance_momentum_scale (float):
                Scale factor for the guidance momentum.
            guidance_mom_beta (float):
                Beta factor for the guidance momentum.

        Returns:
            torch.Tensor: Guided embeddings.
        """
        if safety_momentum is None:
            safety_momentum = torch.zeros_like(embeddings)

        # Equation 6
        scale = torch.clamp(
            torch.abs((embeddings - safety_concept_embeddings)) * guidance_scale, max=1.)

        # Equation 6
        safety_concept_scale = torch.where(
            (embeddings - safety_concept_embeddings) >= guidance_threshold,
            torch.zeros_like(scale), scale)

        # Equation 4
        guidance_safety_emb = torch.mul(embeddings, safety_concept_scale)

        # Equation 7
        noise_guidance_safety = guidance_safety_emb + guidance_momentum_scale * safety_momentum

        # Equation 8
        safety_momentum = guidance_mom_beta * safety_momentum + (1 - guidance_mom_beta) * noise_guidance_safety

        return safety_momentum
        


    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        guidance_scale: float = 0.0,
        verbose: bool = False
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

        if enable_safety_guidance:
            safety_concept_ids = self.tokenizer(self.safety_concepts, return_tensors="pt").input_ids
            safety_concept_embeddings = self.get_embeddings(safety_concept_ids)
            safety_concept_embeddings = torch.mean(safety_concept_embeddings, dim=1, keepdim=True)
            safety_concept_embeddings = safety_concept_embeddings.to(self.model.device)
        
        # Tokenize the input prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_embeddings = self.get_embeddings(input_ids)


        current_input = input_embeddings.to(self.model.device)
        generated_tokens = input_ids.to(self.model.device)

        for i, _ in enumerate(range(max_length - 1)):

            if verbose:
                self.logger.info(f"Step {i+1}/{max_length}")

            # Forward pass with custom embeddings
            outputs = model(inputs_embeds=current_input)
            logits = outputs.logits

            # Get the next token from the logits
            next_token = logits[:, -1, :]
            next_token_probs = F.softmax(next_token, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)

            # Append the next token to the generated sequence
            generated_tokens = torch.cat((generated_tokens, next_token), dim=-1)

            # Get the embedding for the new token and use it as input for the next step
            new_embedding = self.get_embeddings(next_token)
            if enable_safety_guidance:
                new_embedding = self.apply_safety_guidance(new_embedding, safety_concept_embeddings, guidance_scale)
                
            current_input = torch.cat((current_input, new_embedding), dim=1)
                

        # Decode and return the generated text
        generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        self.logger.info("------------ start of generated text ------------")
        return generated_text

# Example Usage
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import login
    import os

    login(os.environ.get("HF_READ_TOKEN"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a model and tokenizer
    model_name = "gpt2"
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

    # Initialize the pipeline
    llm_pipeline = LLMPipeline(model=model, tokenizer=tokenizer)

    # Run text generation
    prompt = "please tell me a sentence about the usa"
    output = llm_pipeline(prompt, max_length=100, verbose=True, guidance_scale=1.1)
    print(output)