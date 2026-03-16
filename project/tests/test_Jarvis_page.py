import pytest
from unittest.mock import MagicMock, patch
from streamlit.testing.v1 import AppTest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# needed if import statement is not working. add project path to python module search

"""
tests b_Chat_with_Jarvis.py page. 

Just inputs a question
"""


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    """Patch OpenAI, Gemini, and get_llm_reply so tests run without real API keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    with patch("openai.OpenAI", return_value=MagicMock()), patch(
        "google.genai.Client", return_value=MagicMock()
    ), patch("app_utils.llm_utils.get_llm_reply", return_value="Mocked reply"):
        yield


@pytest.fixture
def app():
    """Fixture to initialize the app for each test"""
    return AppTest.from_file("../pages/b_Chat_with_Jarvis.py")


def test_app_loads_without_errors(app):
    """Test that the app loads successfully"""
    at = app.run(timeout=10)
    assert not at.exception


def test_selectbox_contains_selected_prompt(app):
    """Test that 'Programming expert' is in selectbox options"""
    at = app.run()
    # radio[0] = Provider, selectbox[0] = Model, selectbox[1] = System prompt preset
    assert "Programming expert" in at.selectbox[1].options


def test_selectbox_has_multiple_options(app):
    """Test that selectbox has multiple system prompt options"""
    at = app.run()
    # radio[0] = Provider, selectbox[0] = Model, selectbox[1] = System prompt preset
    assert len(at.selectbox[1].options) >= 3


def test_user_question_input(app):
    """Test inputting a question via chat input"""
    at = app.run()
    at.chat_input[0].set_value("hi").run()
    assert not at.exception


def test_provider_radio_options(app):
    """Test that provider radio has GPT and Gemini options"""
    at = app.run()
    # radio[0] = Provider
    assert "GPT" in at.radio[0].options
    assert "Gemini" in at.radio[0].options


def test_provider_radio_default_is_gpt(app):
    """Test that the default provider is GPT"""
    at = app.run()
    assert at.radio[0].value == "GPT"


def test_provider_switch_to_gemini(app):
    """Test switching provider to Gemini updates model selectbox"""
    at = app.run()
    at.radio[0].set_value("Gemini").run()
    assert at.radio[0].value == "Gemini"
    assert not at.exception


def test_complete_interaction_flow(app):
    """Test complete user interaction: select prompt and input question"""
    at = app.run()
    at.selectbox[1].set_value("Programming expert").run()
    at.chat_input[0].set_value("What is Python?").run()

    assert at.selectbox[1].value == "Programming expert"
    assert not at.exception
