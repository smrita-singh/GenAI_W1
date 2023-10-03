import openai
import os
import streamlit as st

# OpenAI configuration
openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_key = "a5637d1b2ad34453807c8a71cecd9dfd"
openai.api_base ="https://jp-sandbox.openai.azure.com/"

# Setting up environment var's
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://jp-sandbox.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "a5637d1b2ad34453807c8a71cecd9dfd"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
deployment_name = 'gpt-4'

# Checking if API works
response = openai.ChatCompletion.create(
    engine="gpt-4", # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
    messages = [
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": "What is EY?"}
        ]
        )

#print(response)
#print(response['choices'][0]['message']['content'])


# Reading the patient data in csv format
import pandas as pd
raw_data = pd.read_csv("patient_notes.csv")
rand_sample = raw_data.sample(n=100, random_state=123)  # Generating 100 random rows
# rand_sample.shape  # printing the number of (row,column)

rand_sample.to_csv("random_100notes.csv", index=False)

from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(
    file_path="./random_100notes.csv",
    csv_args = {"fieldnames": ["pn_num", "case_num", "pn_history"],},
    source_column = "pn_num"
    )

data = loader.load()

from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    deployment = "text-embedding-ada-002",
    model = "gpt-4",
    openai_api_key = os.getenv("OPENAI_API_KEY"),
    openai_api_base = os.getenv("OPENAI_API_BASE"),
    openai_api_type = "azure",
    chunk_size = 16,
)

st.title("MedScribe AIssistant​")

user_input = st.text_input("Enter the patient number to get patient demogrphics/symptoms/medical history and other details: ")

from langchain.vectorstores import FAISS
full_index = FAISS.from_documents(data, embeddings)

# Retrieve rows for this patient only, from the vector store
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
retriever = full_index.as_retriever(search_type="mmr", search_kwargs={'filter': {'source': user_input}}) 
#retriever = full_index.as_retriever() 
llm = AzureChatOpenAI(deployment_name="gpt-4", temperature=0)
query = RetrievalQA.from_chain_type(llm = llm, chain_type = "stuff", retriever = retriever, \
                                return_source_documents=True)

# Pass the query and the retrieved/relevant data to the LLM
q1_name = query(f"What is the age and gender of this patient {user_input}?\
                List any medications taken.\
                List any current symptoms mentioned by this patient.\
                List any family medical history hx.\
                Can you suggest tests or lab work that needs to be done?.\
                Provide your answers as a markdown")

st.write(q1_name["result"])
print(q1_name["result"])
q1_name["source_documents"]


