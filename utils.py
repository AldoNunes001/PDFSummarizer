from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
import os
import tempfile


def load_pdf_content(file) -> str:
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.getvalue())  # Write the uploaded content to the temp file
        file_path = temp_file.name  # Get the path to the temporary file

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text = "".join(page.page_content for page in pages)
    text = text.replace("\t", " ")

    if not text:
        raise Exception("Cannot convert PDF")

    os.remove(file_path)

    return text


def split_text(llm, text: str):
    num_tokens = llm.get_num_tokens(text)
    print(f"This document has {num_tokens} tokens in it")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=100
    )
    docs = text_splitter.create_documents([text])

    num_docs = len(docs)
    num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)
    print(
        f"Now we have {num_docs} documents and the first one has {num_tokens_first_doc}"
        f" tokens"
    )
    return docs


def summarize(llm, docs):
    map_prompt = """
Write a complete summary of the following:
"{text}"
COMPLETE SUMMARY:
"""
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    summary_chain = load_summarize_chain(llm=llm,
                                         chain_type="map_reduce",
                                         map_prompt=map_prompt_template,
                                         #  verbose=True
                                         )

    return summary_chain.run(docs)
