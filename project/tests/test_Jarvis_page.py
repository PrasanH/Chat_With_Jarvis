import pytest
from unittest.mock import MagicMock, patch
from streamlit.testing.v1 import AppTest

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# needed if import statement is not working. add project path to python module search

from pages import b_Chat_with_Jarvis

"""
tests b_Chat_with_Jarvis.py page. 

Just inputs a question
"""


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    """Patch OpenAI and get_llm_reply so tests run without a real API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("openai.OpenAI", return_value=MagicMock()), patch(
        "app_utils.llm_utils.get_llm_reply", return_value="Mocked reply"
    ):
        yield


@pytest.fixture
def app():
    """Fixture to initialize the app for each test"""
    return AppTest.from_file("../pages/b_Chat_with_Jarvis.py")


def test_app_loads_without_errors(app):
    """Test that the app loads successfully"""
    at = app.run()
    assert not at.exception


def test_selectbox_contains_selected_prompt(app):
    """Test that 'Programming expert' is in selectbox options"""
    at = app.run()
    # selectbox[0] = Model, selectbox[1] = System prompt preset
    assert "Programming expert" in at.selectbox[1].options


def test_selectbox_has_multiple_options(app):
    """Test that selectbox has multiple system prompt options"""
    at = app.run()
    # selectbox[1] is the system prompt preset
    assert len(at.selectbox[1].options) >= 3


def test_user_question_input(app):
    """Test inputting a question via chat input"""
    at = app.run()
    at.chat_input[0].set_value("hi").run()
    assert not at.exception


def test_complete_interaction_flow(app):
    """Test complete user interaction: select prompt and input question"""
    at = app.run()
    at.selectbox[1].set_value("Programming expert").run()
    at.chat_input[0].set_value("What is Python?").run()

    assert at.selectbox[1].value == "Programming expert"
    assert not at.exception
