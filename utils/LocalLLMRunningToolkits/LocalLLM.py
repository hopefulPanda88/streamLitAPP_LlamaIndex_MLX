"""
A wrapper used to run a local llm for the RAG jobs.
Please refer to https://medium.com/@bSharpML/use-llamaindex-and-a-local-llm-to-summarize-youtube-videos-29817440e671
Created on Mon Jan 8th 2024
@author: Yang Nie
"""
import os.path
import warnings
import mlx.core as mx
from llama_index.core.base.llms.generic_utils import messages_to_prompt as generic_messages_to_prompt
from llama_index.core.base.llms.types import LLMMetadata, ChatMessage, CompletionResponse, CompletionResponseGen
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.types import PydanticProgramMode, BaseOutputParser
from mlx_lm import generate
from typing import Any, List, Optional, Callable, Sequence, Union

from utils.models.generate_other_types import generate as special_generate

from llama_index.core import PromptTemplate

from mlx_lm import load
from mlx_lm.utils import generate_step
from pydantic import Field, PrivateAttr

from utils.tools.file import read_json_file


class LocalLLMOnMLX(CustomLLM):
    """
    A custom class to load any local LLM using MLX and conduct predication.
    """
    model_name: str = Field(
        # default = ""
        description=(
            "The local model path of the llm"
            "Must be specified in advance by user"
        ),
    )
    tokenizer_name: str = Field(
        # default="",
        description=(
            "The name of the tokenizer to use from local llm. MUST be specified"
            "Unused if `tokenizer` is passed in directly."
        ),
    )
    context_window: int = Field(
        default=3900,
        description="The maximum number of tokens available for input",
        gt=0,
    )
    max_new_tokens: int = Field(
        default=256,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    temperature: float = Field(
        default=0.0,
        description="model temperature",
        ge=0.0,
        le=1.0
    )
    system_prompt: str = Field(
        default="",
        description=(
            "The system prompt, containing any extra instructions or context. "
            "The model card on HuggingFace should specify if this is needed."
        ),
    )
    query_wrapper_prompt: PromptTemplate = Field(
        default=PromptTemplate("{query_str}"),
        description=(
            "The query wrapper prompt, containing the query placeholder. "
            "The model card on HuggingFace should specify if this is needed. "
            "Should contain a `{query_str}` placeholder."
        ),
    )

    device_map: str = Field(
        default="auto",
        description="The device_map to use. Defaults to 'auto'.Here, on MacOS, MLX will be added and implemented."
    )
    stopping_ids: List[int] = Field(
        default_factory=list,
        description=(
            "The stopping ids to use. "
            "Generation stops when these token IDs are predicted."
        ),
    )
    tokenizer_outputs_to_remove: list = Field(
        default_factory=list,
        description=(
            "The outputs to remove from the tokenizer. "
            "Sometimes huggingface tokenizers return extra inputs that cause errors."
        ),
    )
    tokenizer_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the tokenizer."
    )
    model_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during initialization.",
    )
    generate_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during generation.",
    )
    is_streaming: bool = Field(
        default=False,
        description="whether run the model under streaming mode or not"
    )
    is_chat_model: bool = Field(
        default=False,
        description=(
                LLMMetadata.__fields__["is_chat_model"].field_info.description
                + " Be sure to verify that you either pass an appropriate tokenizer "
                  "that can convert prompts to properly formatted chat messages or a "
                  "`messages_to_prompt` that does so."
        ),
    )
    dummy_response = "I am a dummy response"
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _stopping_criteria: Any = PrivateAttr()

    def __init__(
            self,
            model_name: str,
            tokenizer_name: str,
            context_window: int = 3900,
            max_new_tokens: int = 256,
            query_wrapper_prompt: Union[str, PromptTemplate] = "{query_str}",
            model: Optional[Any] = None,
            tokenizer: Optional[Any] = None,
            chat_template: Optional[str] = None,
            device_map: Optional[str] = "auto",
            stopping_ids: Optional[List[int]] = None,
            tokenizer_kwargs: Optional[dict] = None,
            tokenizer_outputs_to_remove: Optional[list] = None,
            model_kwargs: Optional[dict] = None,
            generate_kwargs: Optional[dict] = None,
            is_chat_model: Optional[bool] = False,
            callback_manager: Optional[CallbackManager] = None,
            system_prompt: str = "",
            messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
            completion_to_prompt: Optional[Callable[[str], str]] = None,
            pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
            output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        """
        class initiation
        Args:
            model_name:
            tokenizer_name:
            context_window:
            max_new_tokens:
            query_wrapper_prompt:
            model:
            tokenizer:
            chat_template: self-defined chat template for generation
            device_map:
            stopping_ids:
            tokenizer_kwargs:
            tokenizer_outputs_to_remove:
            model_kwargs:
            generate_kwargs:
            is_chat_model:
            callback_manager:
            system_prompt:
            messages_to_prompt:
            completion_to_prompt:
            pydantic_program_mode:
            output_parser:
        """
        # chat mode
        # self.is_streaming = is_streaming
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
        except ImportError as err:
            raise ImportError(
                f"{type(self).__name__} requires torch and transformers packages.\n"
                "Please install both with `pip install transformers[torch]`."
            ) from err

        """Begin to load model and tokenizer using MLX. You can directly pass a mlx model instance"""

        if model or tokenizer is None:
            if tokenizer_kwargs is not None:
                self._model, self._tokenizer = load(model_name, tokenizer_config=tokenizer_kwargs)
            else:
                self._model, self._tokenizer = load(model_name)
        else:
            self._model, self._tokenizer = model, tokenizer

        # set my own chat template for testing on mistral AI model
        if chat_template is not None:
            self._tokenizer.chat_template = chat_template

        # check context_window and make sure its value is not missing and valid.
        config_dict = read_json_file(os.path.join(model_name, "config.json"))
        # config_dict = self._model.to_dict()
        model_context_window = int(
            config_dict.get("max_position_embeddings", context_window)
        )
        if model_context_window and model_context_window < context_window:
            warnings.warn(f"Supplied context window: {context_window} exceeds the maximum length of the model, "
                          f"so it will be reset to {model_context_window}")
            context_window = model_context_window

        """model kwargs definitions"""

        # self.temperature = 0.0
        # if self.model_kwargs is not None:
        #     if self.model_kwargs.get("temp", False) is not False:
        #         self.temperature = self.model_kwargs.get("temp")

        """setup stopping criteria"""
        stopping_ids_list = stopping_ids or []

        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                for stop_id in stopping_ids_list:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        self._stopping_criteria = StoppingCriteriaList([StopOnTokens])

        if isinstance(query_wrapper_prompt, str):
            query_wrapper_prompt = PromptTemplate(query_wrapper_prompt)

        messages_to_prompt = messages_to_prompt or self._tokenizer_messages_to_prompt

        super().__init__(
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            device_map=device_map,
            stopping_ids=stopping_ids or [],
            tokenizer_kwargs=tokenizer_kwargs or {},
            tokenizer_outputs_to_remove=tokenizer_outputs_to_remove or [],
            model_kwargs=model_kwargs or {},
            generate_kwargs=generate_kwargs or {},
            is_chat_model=is_chat_model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "LocalLLMonMLX"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata"""
        return LLMMetadata(
            context_window=self.context_window,
            max_new_tokens=self.max_new_tokens,
            model_name=self.model_name
        )

    def _tokenizer_messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """Use the tokenizer to convert messages to prompt. Fallback to generic."""
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages_dict = [
                {"role": message.role.value, "content": message.content}
                for message in messages
            ]
            tokens = self._tokenizer.apply_chat_template(messages_dict)
            return self._tokenizer.decode(tokens)

        return generic_messages_to_prompt(messages)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete the generation"""
        full_prompt = prompt
        is_formatted = kwargs.pop("is_formatted", False)

        if not is_formatted:
            if self.query_wrapper_prompt:
                full_prompt = self.query_wrapper_prompt.format(query_str=prompt)
            if self.system_prompt:
                full_prompt = f"{self.system_prompt} {full_prompt}"

        completion = generate(self._model, self._tokenizer,
                              full_prompt,
                              temp=self.temperature,
                              max_tokens=self.max_new_tokens,
                              verbose=False)
        return CompletionResponse(text=completion)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        full_prompt = prompt
        is_formatted = kwargs.pop("is_formatted", False)

        if not is_formatted:
            if self.query_wrapper_prompt:
                full_prompt = self.query_wrapper_prompt.format(query_str=prompt)
            if self.system_prompt:
                full_prompt = f"{self.system_prompt} {full_prompt}"

        encoded_prompt = mx.array(self._tokenizer.encode(full_prompt))
        tokens = []
        skip = 0
        for (token, prob), _ in zip(generate_step(encoded_prompt, self._model, temp=self.temperature),
                                    range(self.max_new_tokens)):
            if token == self._tokenizer.eos_token_id:
                break

            tokens.append(token.item())
            s = self._tokenizer.decode(tokens)
            yield CompletionResponse(text=s[:skip], delta=s[skip:])
            # print(s[skip:], end="", flush=True)
            skip = len(s)
