import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
from src.mcqgenerator.logger import logging

from langchain.chains import SequentialChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

#load the envvar from .env file
load_dotenv()

#access the API key
key=os.getenv('GROQ_API_KEY')
print(key)

#LLM object
llm_groq=ChatGroq(temperature=1.5, api_key=key,model_name="llama3-8b-8192")

TEMPLATE='''
Text:{text}
you are expert MCQ maker. Given the above text, it is your job to\
create a quiz of {number} multiple choice questions for {subject} students\
in {tone} tone. \
Make sure the questions are not repeated and check all the questions to be confirming the text as well.\
Make sure to format your response as per below json format named as RESPONSE_JSON and use it as a guide.\
Ensure to make {number} MCQs\

### RESPONSE_JSON\
{response_json}\
'''

quiz_generation_prompt=PromptTemplate(input_variables=['text','number','subject','tone','response_json'],template=TEMPLATE)

quiz_chain=LLMChain(llm=llm_groq,prompt=quiz_generation_prompt,output_key='quiz',verbose=True)

TEMPLATE2="""You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. \
Only use at max 50 words for complexity analysis. \
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities\
Quiz_MCQs:\
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt=PromptTemplate(input_variables=['quiz','subject'],template=TEMPLATE2)

review_chain=LLMChain(llm=llm_groq,prompt=quiz_evaluation_prompt,output_key='review',verbose=True)

generate_evaluate_chain=SequentialChain(chains=['quiz_chain','review_chain'],input_variables=['text','number','subject','tone','response_json'],output_key=['quiz','review'],verbose=True)





