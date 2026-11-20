import pytest

from jaxforge.codegen.parser import CodeParser
from jaxforge.exceptions import JaxForgeLLMError


class TestParser:
    def test_extract_code(self):
        response = {"code": "print('hello')", "intent": "build"}
        assert CodeParser.extract(response, "build") == "print('hello')"

    def test_missing_code_raises(self):
        response = {"intent": "build"}
        with pytest.raises(JaxForgeLLMError, match="missing 'code'"):
            CodeParser.extract(response, "build")

    def test_empty_code_raises(self):
        response = {"code": "   ", "intent": "build"}
        with pytest.raises(JaxForgeLLMError, match="empty code block"):
            CodeParser.extract(response, "build")

    def test_non_string_code_raises(self):
        response = {"code": 123, "intent": "build"}
        with pytest.raises(JaxForgeLLMError, match="must be a string"):
            CodeParser.extract(response, "build")

    def test_mismatched_intent_raises(self):
        response = {"code": "def train_model(): pass", "intent": "train"}
        with pytest.raises(JaxForgeLLMError, match="does not match"):
            CodeParser.extract(response, "build")
