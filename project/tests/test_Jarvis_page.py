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


def test_chat_function():
    
    at = AppTest.from_file('../pages/b_Chat_with_Jarvis.py').run()
    try:
        #select a value from the first dropdown(selectbox)
        at.selectbox[0].set_value("You are an expert in programming").run()
        
        assert at.selectbox[0].options[2] == "You are an expert in programming"

        #input 'hi' in the question area
        at.text_input[0].input('hi').run()
        print(f'Question input: {at.text_input[0].value}')
        
        print(at.session_state.chat_history_jarvis)
    except Exception as e:
        pytest.fail(f"Skipping test due to no OpenAI API key: {str(e)}")
    


