from setuptools import find_packages,setup

setup(name='mcqgenerator',
      version='0.0.1',
      author='jai panchal',
      author_email='jaipanchal97@gmail',
      include_dirs=['openai','langchain','streamlit','python-dotenv','PyPDF2'],
      packages=find_packages()
      )