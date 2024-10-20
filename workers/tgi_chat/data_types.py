import dataclasses
import random
import inspect
from typing import Dict, Any, List

from transformers import AutoTokenizer
import nltk

from lib.data_types import ApiPayload, JsonDataException

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()

tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")


@dataclasses.dataclass
class InputParameters:
    max_new_tokens: int = 256

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "InputParameters":
        errors = {}
        for param in inspect.signature(cls).parameters:
            if param not in json_msg:
                errors[param] = "missing parameter"
        if errors:
            raise JsonDataException(errors)
        return cls(
            **{
                k: v
                for k, v in json_msg.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclasses.dataclass
class InputData(ApiPayload):
    inputs: str
    parameters: InputParameters

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputData":
        return cls(
            inputs=data["inputs"], parameters=InputParameters(**data["parameters"])
        )

    @classmethod
    def for_test(cls) -> "InputData":
        prompt = " ".join(random.choices(WORD_LIST, k=int(250)))
        return cls(inputs=prompt, parameters=InputParameters())

    def generate_payload_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def count_workload(self) -> int:
        return self.parameters.max_new_tokens

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "InputData":
        errors = {}
        for param in inspect.signature(cls).parameters:
            if param not in json_msg:
                errors[param] = "missing parameter"
        if errors:
            raise JsonDataException(errors)
        try:
            parameters = InputParameters.from_json_msg(json_msg["parameters"])
            return cls(inputs=json_msg["inputs"], parameters=parameters)
        except JsonDataException as e:
            errors["parameters"] = e.message
            raise JsonDataException(errors)

@dataclasses.dataclass
class Message:
    role: str
    content: str

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "Message":
        errors = {}
        for param in inspect.signature(cls).parameters:
            if param not in json_msg:
                errors[param] = "missing parameter"
        if errors:
            raise JsonDataException(errors)
        return cls(
            role=json_msg["role"],
            content=json_msg["content"]
        )


@dataclasses.dataclass
class ChatInputData:
    model: str
    messages: List[Message]
    stream: bool
    max_tokens: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatInputData":
        messages = [Message.from_json_msg(msg) for msg in data["messages"]]
        return cls(
            model=data["model"],
            messages=messages,
            stream=data["stream"],
            max_tokens=data["max_tokens"]
        )


    @classmethod
    def for_test(cls) -> "InputData":
        messages = [{"role": "user", "content": " ".join(random.choices(WORD_LIST, k=int(250)))}]
        return cls(
            model="/models/otaku.gguf",
            messages=messages,
            stream=False,
            max_tokens=1024
        )

    def generate_payload_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def count_workload(self) -> int:
        return self.max_tokens

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "ChatInputData":
        errors = {}
        for param in inspect.signature(cls).parameters:
            if param not in json_msg:
                errors[param] = "missing parameter"
        if errors:
            raise JsonDataException(errors)
        try:
            messages = [Message.from_json_msg(msg) for msg in json_msg["messages"]]
            return cls(
                model=json_msg["model"],
                messages=messages,
                stream=json_msg["stream"],
                max_tokens=json_msg["max_tokens"]
            )
        except JsonDataException as e:
            errors["parameters"] = e.message
            raise JsonDataException(errors)

