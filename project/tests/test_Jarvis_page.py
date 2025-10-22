import pytest
from streamlit.testing.v1 import AppTest

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#needed if import statement is not working. add project path to python module search

from pages import b_Chat_with_Jarvis 

"""
tests b_Chat_with_Jarvis.py page. 

Just inputs a question
"""

@pytest.fixture
def app():
    """Fixture to initialize the app for each test"""
    return AppTest.from_file('../pages/b_Chat_with_Jarvis.py')


def test_app_loads_without_errors(app):
    """Test that the app loads successfully"""
    at = app.run()
    assert not at.exception


def test_selectbox_contains_selected_prompt(app):
    """Test that 'You are an expert in programming' is in selectbox options"""
    at = app.run()
    assert "You are an expert in programming" in at.selectbox[0].options


def test_selectbox_has_multiple_options(app):
    """Test that selectbox has multiple system prompt options"""
    at = app.run()
    assert len(at.selectbox[0].options) >= 3


def test_user_question_input(app):
    """Test inputting a question"""
    at = app.run()
    at.text_input[0].input('hi').run()
    assert at.text_input[0].value == 'hi'


def test_complete_interaction_flow(app):
    """Test complete user interaction: select prompt and input question"""
    at = app.run()
    at.selectbox[0].set_value("You are an expert in programming").run()
    at.text_input[0].input('What is Python?').run()
    
    assert at.selectbox[0].value == "You are an expert in programming"
    assert at.text_input[0].value == 'What is Python?'


    


