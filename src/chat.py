import re
import torch

class Chat:
    # by default, will maintain all conversations in OpenAI chat format

    def __init__(self, model, prompt_style, tokenizer, max_length=2048, 
                 init_conversation = None, init_system_prompt = None):
        
        if init_conversation is not None and init_system_prompt is not None:
            raise ValueError("init_conversation and init_system_prompt cannot be provided at the same time")
       
        self.model = model
        self.prompt_style = prompt_style
        self.tokenizer = tokenizer
        self.max_length = max_length # limit the length of the whole conversation
        

        # formatter will be used to convert openai chat format to string
        if prompt_style == 'llama2':
            from src.models.model_families.llama2 import LlamaStringConverter, default_system_prompt
            self.string_formatter = LlamaStringConverter.string_formatter_completion_only 
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = None
        elif prompt_style == 'gemma':
            from src.models.model_families.gemma import GemmaStringConverter, default_system_prompt
            self.string_formatter = GemmaStringConverter.string_formatter_completion_only 
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = None
        elif prompt_style == 'llama2_base':
            from src.models.model_families.llama2_base import LlamaStringConverter, default_system_prompt, base_stopping_criteria
            self.string_formatter = LlamaStringConverter.string_formatter_completion_only 
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = base_stopping_criteria
        elif prompt_style == 'gemma_base':
            from src.models.model_families.gemma_base import GemmaStringConverter, default_system_prompt, base_stopping_criteria
            self.string_formatter = GemmaStringConverter.string_formatter_completion_only
            self.default_system_prompt = default_system_prompt
            self.stopping_criteria = base_stopping_criteria
        else:
            raise ValueError(f"Prompt style {prompt_style} not supported")


        if init_conversation is not None:
            self.validate_conversation(init_conversation)
            if isinstance(init_conversation, dict):
                init_conversation = init_conversation['messages']

            if init_conversation[-1]['role'] == 'user':
                raise ValueError("the last message of init_conversation should be assistant message or system prompt, not user message")

            if init_conversation[0]['role'] != 'system':
                self.system_prompt = self.default_system_prompt
                self.converstaion = self.init_conversation() + init_conversation
            else:
                self.system_prompt = init_conversation[0]['content']
                self.converstaion = init_conversation
        else:

            if init_system_prompt is not None:
                self.system_prompt = init_system_prompt
            else:
                self.system_prompt = self.default_system_prompt
            
            self.converstaion = self.init_conversation()

        self.unsafe_concepts: str | None = 'hate, harassment, violence, suffering, humiliation, harm, suicide, ' \
                                        'sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, ' \
                                        'drug use, theft, vandalism, weapons, child abuse, brutality, cruelty'

        

    def __call__(self, text, max_new_tokens = 1024, 
                 do_sample = True, top_p = 0.9, temperature = 0.6, use_cache = True, top_k = 50,
                 repetition_penalty = 1.0, length_penalty = 1.0, **kwargs):
        
        self.update_conversation(user_message=text)
        _, tokens_input = self.prepare_model_input(conversation=self.conversation, max_new_tokens=max_new_tokens)

        tokens_input = tokens_input.unsqueeze(0).to(self.model.device)
        input_length = tokens_input.shape[1]
        outputs = self.model.generate(
                input_ids = tokens_input,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                stopping_criteria = self.stopping_criteria,
                **kwargs
            )

        output_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        self.update_conversation(assistant_message=output_text)

        return output_text
    

    
    def generate_one_shot(self, input, max_new_tokens = 1024, 
                 do_sample = True, top_p = 0.9, temperature = 0.6, use_cache = True, top_k = 50,
                 repetition_penalty = 1.0, length_penalty = 1.0, **kwargs):
        # a one-shot conversation, input can be a string, a list of messages, or a dictionary with 'messages' key
        # no history will be maintained for one-shot conversation
        
        if isinstance(input, dict) or isinstance(input, list):
            input = self.validate_conversation(input)
        elif isinstance(input, str):
            input = self.init_conversation() + [{'role': 'user', 'content': input}, {'role': 'assistant', 'content': ''}]
        else:
            raise ValueError(f"input {input} is not a valid conversation input")

        

        _, tokens_input = self.prepare_model_input(input, max_new_tokens)
        tokens_input = tokens_input.to(self.model.device)
        input_length = tokens_input.shape[1]


        outputs = self.model.generate(
                input_ids = tokens_input,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                stopping_criteria = self.stopping_criteria,
                **kwargs
            )

        output_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True) # the model output part
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True) # the whole conversation

        return output_text, full_text
    

    def generate_one_shot_in_batch(self, inputs, accelerator, max_new_tokens = 1024, 
                 do_sample = True, top_p = 0.9, temperature = 0.6, use_cache = True, top_k = 50,
                 repetition_penalty = 1.0, length_penalty = 1.0, guidance_scale=0.0, **kwargs):
        # a one-shot conversation, input can be a string, a list of messages, or a dictionary with 'messages' key
        # no history will be maintained for one-shot conversation
        # this function is for batch inference to accelerate the evaluation
        
        user_input_processed = []
        system_input_processed = []
        assistant_prompt_processed = []

        enable_safety_guidance = True
        if guidance_scale < 1:
            enable_safety_guidance = False
            print('You have disabled safety guidance.')

        for item in inputs:

            if isinstance(item, dict) or isinstance(item, list):
                item_processed = self.validate_conversation(item)
            elif isinstance(item, str):
                item_processed = self.init_conversation() + [{'role': 'user', 'content': input}, {'role': 'assistant', 'content': ''}]
            else:
                raise ValueError(f"input {item} is not a valid conversation input")
            
            
            for role in item_processed:
                if role['role'] == 'user':
                    user_input_processed.append(role['content'])
                elif role['role'] == 'system':
                    system_input_processed.append(role['content'])
                elif role['role'] == 'assistant':
                    assistant_prompt_processed.append(role['content'])

        if len(user_input_processed) != 0:
            user_prompt_inputs = self.tokenizer(user_input_processed, padding = True, return_tensors="pt").to(self.model.device)
            user_prompt_embeddings = self.get_embeddings(user_prompt_inputs["input_ids"])

            if enable_safety_guidance:
                unsafe_concepts_ids = self.tokenizer(self.unsafe_concepts, return_tensors="pt").input_ids
                unsafe_concepts_ids = unsafe_concepts_ids.to(self.model.device)
                unsafe_concepts_embeddings = self.get_embeddings(unsafe_concepts_ids)
                
                user_prompt_embeddings = self.apply_safety_guidance(
                    embeddings=user_prompt_embeddings,
                    unsafe_concepts_embeddings=unsafe_concepts_embeddings,
                    guidance_scale=guidance_scale,
                )
        
        if len(system_input_processed) != 0:
            system_prompt_inputs = self.tokenizer(system_input_processed, padding = True, return_tensors="pt").to(self.model.device)
            system_prompt_embeddings = self.get_embeddings(system_prompt_inputs["input_ids"])
        else:
            system_prompt_embeddings = None
        
        if len(assistant_prompt_processed) != 0 and assistant_prompt_processed[0] != '':
            assistant_prompt_inputs = self.tokenizer(assistant_prompt_processed, padding = True, return_tensors="pt").to(self.model.device)
            assistant_prompt_embeddings = self.get_embeddings(assistant_prompt_inputs["input_ids"])
        else:
            assistant_prompt_embeddings = None
            
        emb_list = [emb for emb in [system_prompt_embeddings, user_prompt_embeddings, assistant_prompt_embeddings] if emb is not None]
        input_embeddings = torch.cat(emb_list, dim=1)

        outputs = self.model.generate(
                # input_ids = model_inputs['input_ids'],
                inputs_embeds=input_embeddings,
                # attention_mask = model_inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                stopping_criteria = self.stopping_criteria,
                **kwargs
            )


        output_texts = []

        for i, item in enumerate(outputs):

            output_text = self.tokenizer.decode(item, skip_special_tokens=True)

            output_texts.append(output_text)

        
        return output_texts
            

    def validate_conversation(self, conversation=None):
        # validate the conversation format, return the conversation in OpenAI chat format

        if conversation is None:
            conversation = self.conversation

        if isinstance(conversation, dict):
            if 'messages' not in conversation:
                raise ValueError(f"conversation {conversation} does not have 'messages' key")
            convs = conversation['messages']

        else: 
            convs = conversation
        
        if not isinstance(convs, list):
            raise ValueError(f"conversation {conversation} is not a valid list of messages")

        if len(convs) == 0:
            raise ValueError(f"the conversation {conversation} is empty")
        
        for conv in convs:
            if 'role' not in conv or 'content' not in conv:
                raise ValueError(f"the message {conv} does not have 'role' or 'content' key")

        
        if convs[0]['role'] != 'system':
            convs = self.init_conversation() + convs

        pt = 1
        
        while pt < len(convs):
            if convs[pt]['role'] != 'user':
                raise ValueError(f"the message should be user - assistant alternation, but the {pt}th message is {convs[pt]['role']}")
            pt += 1
            if pt >= len(convs):
                break
            if convs[pt]['role'] != 'assistant':
                raise ValueError(f"the message should be user - assistant alternation, but the {pt}th message is {convs[pt]['role']}")
            pt += 1
        return convs
    
    def init_conversation(self, system_prompt=None):
        if system_prompt is None:
            system_prompt = self.system_prompt
        return [{'role': 'system', 'content': system_prompt}]
    
    def refresh_conversation(self):
        self.conversation = self.init_conversation()
    
    def update_conversation(self, conversation = None, user_message=None, assistant_message=None):
        if conversation is None:
            conversation = self.conversation
        
        if user_message is None and assistant_message is None:
            raise ValueError("user_message or assistant_message should be provided")
        
        if user_message is not None:
            if conversation[-1]['role'] == 'user':
                raise ValueError("the message should be user - assistant alternation")
            conversation.append({'role': 'user', 'content': user_message})
        
        if assistant_message is not None:
            if conversation[-1]['role'] == 'assistant' or conversation[-1]['role'] == 'system':
                raise ValueError("the message should be user - assistant alternation")
            conversation.append({'role': 'assistant', 'content': assistant_message})
    
    def prepare_model_input(self, conversation=None, max_new_tokens=512):
        if conversation is None:
            conversation = self.conversation
        string_input = self.string_formatter({'messages': conversation})['text']
        
        tokens_input = self.tokenizer.encode(string_input, return_tensors="pt", 
                                             max_length=self.max_length - max_new_tokens, truncation=True)

        return string_input, tokens_input
    
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
        
            # Another option I saw
            # return model.get_input_embeddings()(input_ids)
        elif hasattr(self.model, 'embed_tokens'):
            # LLaMA-like models
            return self.model.embed_tokens(input_ids)
        
            # Another option I saw
            # return model.model.embed_tokens(input_ids)
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
