import pytest
from jax_gemini.codegen.parser import CodeParser
from jax_gemini.exceptions import JaxGeminiLLMError

class TestParser:

    def test_extract_code(self):
        response = {"code": "print('hello')", "intent": "build"}
        assert CodeParser.extract(response, "build") == "print('hello')"

    def test_missing_code_raises(self):
        response = {"intent": "build"}
        with pytest.raises(JaxGeminiLLMError, match="missing 'code'"):
            CodeParser.extract(response, "build")

    def test_empty_code_raises(self):
        response = {"code": "   ", "intent": "build"}
        with pytest.raises(JaxGeminiLLMError, match="empty code block"):
            CodeParser.extract(response, "build")
