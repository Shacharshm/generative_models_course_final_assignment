import os
import torch

from huggingface_hub import login
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessorList, StoppingCriteriaList
from typing import Optional, Union, Dict, Any


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

        # Register the components
        self.register_modules(model=model, tokenizer=tokenizer, scheduler=scheduler)

    def register_modules(self, **kwargs):
        """Registers components of the pipeline."""
        for name, module in kwargs.items():
            setattr(self, name, module)

    def enable_sequential_cpu_offload(self):
        """
        Offloads the model to CPU and loads it to GPU only during forward pass to save memory.
        Requires `accelerate`.
        """
        if not hasattr(self.model, 'cpu_offload'):
            raise ValueError("Sequential CPU offloading is not supported for this model.")
        self.model.cpu_offload()

    def disable_sequential_cpu_offload(self):
        """
        Disables sequential CPU offload.
        """
        if hasattr(self.model, 'cpu_offload'):
            self.model.cpu_offload(False)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        """
        Enable sliced attention computation to save memory.

        Args:
            slice_size (str or int): Size of the slices to compute attention. Defaults to "auto".
        """
        if hasattr(self.model, 'set_attention_slice'):
            self.model.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        """Disables attention slicing."""
        if hasattr(self.model, 'set_attention_slice'):
            self.model.set_attention_slice(None)

    @torch.no_grad()
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request and generate a response.

        Args:
            request (Dict[str, Any]):
                A dictionary containing the input prompt and optional generation parameters.
                Expected keys:
                - "prompt" (str): The input text prompt.
                - Other optional generation parameters (e.g., "max_length", "temperature").

        Returns:
            Dict[str, Any]:
                A dictionary containing the generated text and any additional metadata.
        """
        prompt = request.get("prompt", "")
        if not prompt:
            return {"error": "Prompt is missing or empty."}

        # Extract generation parameters
        max_length = request.get("max_length", 50)
        temperature = request.get("temperature", 0.7)
        top_k = request.get("top_k", 50)
        top_p = request.get("top_p", 0.9)
        do_sample = request.get("do_sample", True)
        stopping_criteria = request.get("stopping_criteria", None)
        logits_processor = request.get("logits_processor", None)

        try:
            # Tokenize the input prompt
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            input_ids = input_ids.to(self.model.device)

            # # Generate embeddings
            # embeddings = self.model.get_input_embeddings()(input_ids)

            # # Perform custom operations on embeddings (if needed)
            # transformed_embeddings = embeddings  # Modify this step if custom processing is required

            # Pass embeddings through the model
            outputs = self.model.generate(
                input_ids=input_ids,
                # inputs_embeds=transformed_embeddings,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
            )

            # Decode and return the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"generated_text": generated_text}

        except Exception as e:
            return {"error": str(e)}

# Example Usage
if __name__ == "__main__":
    login(os.environ.get("HF_READ_TOKEN"))
    # Load Meta-Llama-3-8B model and tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B"  # Replace with actual Meta-Llama model path or name
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

    # Example request
    request = {
        "prompt": "What is the capital of France?",
        "max_length": 100,
        "temperature": 0.8,
        "top_k": 40
    }

    # Process request and get response
    response = llm_pipeline.process_request(request)
    print(response)
