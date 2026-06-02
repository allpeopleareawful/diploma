from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from dec_unimumer.prompts import RECOGNITION_PROMPT


DEFAULT_PROMPT = RECOGNITION_PROMPT


@dataclass(frozen=True)
class GenerationResult:
    text: str
    avg_logprob: float | None = None
    confidence: float | None = None


class InferenceBackend(Protocol):
    model_name: str

    def predict(self, image: Image.Image, prompt: str = DEFAULT_PROMPT) -> str:
        ...

    def predict_with_metadata(
        self,
        image: Image.Image,
        prompt: str = DEFAULT_PROMPT,
    ) -> GenerationResult:
        ...

    def predict_batch_with_metadata(
        self,
        images: list[Image.Image],
        prompts: list[str],
    ) -> list[GenerationResult]:
        ...


@dataclass
class MockBackend:
    model_name: str = "mock-model"

    def predict(self, image: Image.Image, prompt: str = DEFAULT_PROMPT) -> str:
        return r"x = 1"

    def predict_with_metadata(
        self,
        image: Image.Image,
        prompt: str = DEFAULT_PROMPT,
    ) -> GenerationResult:
        return GenerationResult(text=self.predict(image, prompt), avg_logprob=0.0, confidence=1.0)

    def predict_batch_with_metadata(
        self,
        images: list[Image.Image],
        prompts: list[str],
    ) -> list[GenerationResult]:
        if len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length.")
        return [
            self.predict_with_metadata(image, prompt)
            for image, prompt in zip(images, prompts, strict=True)
        ]


@dataclass
class VLLMBackend:
    model_name: str = "phxember/Uni-MuMER-Qwen2.5-VL-3B"
    temperature: float = 0.2
    top_p: float = 0.8
    max_tokens: int = 2048
    max_model_len: int = 2048
    trust_remote_code: bool = True
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90

    def __post_init__(self) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is required for Uni-MuMER inference. "
                "Install it in a CUDA Linux/WSL environment, e.g. pip install vllm."
            ) from exc

        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        self._llm = LLM(
            model=self.model_name,
            trust_remote_code=self.trust_remote_code,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )

    def predict(self, image: Image.Image, prompt: str = DEFAULT_PROMPT) -> str:
        return self.predict_with_metadata(image, prompt).text

    def predict_with_metadata(
        self,
        image: Image.Image,
        prompt: str = DEFAULT_PROMPT,
    ) -> GenerationResult:
        return self.predict_batch_with_metadata([image], [prompt])[0]

    def predict_batch_with_metadata(
        self,
        images: list[Image.Image],
        prompts: list[str],
    ) -> list[GenerationResult]:
        if len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length.")
        if not images:
            return []
        requests = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image.convert("RGB")},
            }
            for image, prompt in zip(images, prompts, strict=True)
        ]
        outputs = self._llm.generate(requests, self._sampling_params)
        return [self._prediction_from_output(output) for output in outputs]

    @staticmethod
    def _prediction_from_output(output: object) -> GenerationResult:
        completions = getattr(output, "outputs", None)
        if not completions:
            return GenerationResult(text="")
        completion = completions[0]
        token_ids = list(getattr(completion, "token_ids", []) or [])
        cumulative_logprob = getattr(completion, "cumulative_logprob", None)
        avg_logprob = None
        confidence = None
        if cumulative_logprob is not None and token_ids:
            avg_logprob = float(cumulative_logprob) / len(token_ids)
            confidence = math.exp(min(0.0, avg_logprob))
        return GenerationResult(
            text=completion.text.strip(),
            avg_logprob=avg_logprob,
            confidence=confidence,
        )


@dataclass
class TransformersBackend:
    model_name: str = "phxember/Uni-MuMER-Qwen2.5-VL-3B"
    adapter: str | None = None
    max_new_tokens: int = 2048
    torch_dtype: str = "auto"
    device_map: str = "auto"
    attn_implementation: str | None = None

    def __post_init__(self) -> None:
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Transformers inference requires: transformers, torch, qwen-vl-utils. "
                "Install them in the environment where the model will run."
            ) from exc

        kwargs = {
            "torch_dtype": torch.bfloat16 if self.torch_dtype == "bf16" else self.torch_dtype,
            "device_map": self.device_map,
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        self._torch = torch
        self._process_vision_info = process_vision_info
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if hasattr(self._processor, "tokenizer"):
            self._processor.tokenizer.padding_side = "left"
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            **kwargs,
        )
        if self.adapter:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter)
        self._model.eval()

    def predict(self, image: Image.Image, prompt: str = DEFAULT_PROMPT) -> str:
        return self.predict_with_metadata(image, prompt).text

    def predict_with_metadata(
        self,
        image: Image.Image,
        prompt: str = DEFAULT_PROMPT,
    ) -> GenerationResult:
        return self.predict_batch_with_metadata([image], [prompt])[0]

    def predict_batch_with_metadata(
        self,
        images: list[Image.Image],
        prompts: list[str],
    ) -> list[GenerationResult]:
        if len(images) != len(prompts):
            raise ValueError("images and prompts must have the same length.")
        if not images:
            return []
        messages_batch = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image.convert("RGB")},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for image, prompt in zip(images, prompts, strict=True)
        ]
        texts = [
            self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_batch
        ]
        image_inputs, video_inputs = self._process_vision_info(messages_batch)
        inputs = self._processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        with self._torch.inference_mode():
            generation = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = generation.sequences
        prompt_width = inputs.input_ids.shape[-1]
        generated_trimmed = [output_ids[prompt_width:] for output_ids in generated_ids]
        decoded = self._processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        transition_scores = None
        if generation.scores:
            transition_scores = self._model.compute_transition_scores(
                generated_ids,
                generation.scores,
                getattr(generation, "beam_indices", None),
                normalize_logits=True,
            )

        predictions: list[GenerationResult] = []
        for row_index, text in enumerate(decoded):
            avg_logprob = None
            confidence = None
            if transition_scores is not None:
                token_scores = transition_scores[row_index]
                finite_scores = token_scores[
                    self._torch.isfinite(token_scores) & (token_scores != 0)
                ]
                if finite_scores.numel() > 0:
                    avg_logprob = float(finite_scores.mean().item())
                    confidence = math.exp(min(0.0, avg_logprob))
            predictions.append(
                GenerationResult(
                    text=text.strip(),
                    avg_logprob=avg_logprob,
                    confidence=confidence,
                )
            )
        return predictions


def build_backend(
    backend: str,
    *,
    model_name: str,
    adapter: str | None = None,
    temperature: float = 0.2,
    top_p: float = 0.8,
    max_tokens: int = 2048,
) -> InferenceBackend:
    if adapter and backend != "transformers":
        raise ValueError("Adapter inference is currently supported only with --backend transformers.")

    if backend == "vllm":
        return VLLMBackend(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    if backend == "transformers":
        return TransformersBackend(
            model_name=model_name,
            adapter=adapter,
            max_new_tokens=max_tokens,
        )

    if backend == "mock":
        return MockBackend(model_name=model_name)

    raise ValueError(f"Unknown inference backend: {backend}")
