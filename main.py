import streamlit as st
from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
from utils import load_pdf_content, split_text, summarize


# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Page title
st.set_page_config(page_title='PDFSummarizer')
st.title('PDFSummarizer')

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# File input
file = st.file_uploader(label='Upload PDF file',
                        type=['pdf', 'PDF'],
                        )

output = None

if file and openai_api_key.startswith('sk-'):
    with st.spinner('Calculating...'):
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        # llm = ChatOpenAI(temperature=0,
        #                  openai_api_key=OPENAI_API_KEY,
        #                  model='gpt-3.5-turbo')

        text = load_pdf_content(file)
        docs = split_text(llm, text)
        output = summarize(llm, docs)

if output:
    st.info(output)
