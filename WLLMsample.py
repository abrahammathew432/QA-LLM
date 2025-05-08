"""
This script creates a database of information gathered from a local PDF file.
"""

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import json 


# Load JSON data from a file
with open('input.json', 'r') as file:
    data = json.load(file)

# Extract the file path from the JSON data
file_path = data["queries"][0]["context"]

# interpret information in the PDF file
try:
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
except Exception as e:
    print(f"Error loading document: {e}")
    exit()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# create and save the local database
db = FAISS.from_documents(texts, embeddings)
db.save_local("faiss")


"""
This script reads the database of information from local text files
and uses a large language model to answer questions about their content.
"""

from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

# Load the JSON input file
with open('input.json', 'r') as f:
    input_data = json.load(f)

# Prepare the template we will use when prompting the AI
template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else. Also mention the page number the data was taken from 
Helpful answer:
"""

# Load the language model
llm = CTransformers(model='llama-2-7b-chat.ggmlv3.q4_1.bin',
                    model_type='llama',
                    config={'max_new_tokens': 256, 'temperature': 0.01})

# Load the interpreted information from the local database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})
db = FAISS.load_local("faiss", embeddings, allow_dangerous_deserialization=True)

# Prepare a version of the llm pre-loaded with the local content
retriever = db.as_retriever(search_kwargs={'k': 2})
prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question'])
qa_llm = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type='stuff',
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': prompt})

# Prepare the output data
output_data = {"results": []}

# Process each context and its associated questions from the JSON input
for query in input_data['queries']:
    context = query['context']
    for question in query['questions']:
        prompt = question  # Use the question as the prompt
        output = qa_llm({'query': prompt})
        result_entry = {
            "context": context,
            "question": question,
            "answer": output["result"]
        }
        output_data["results"].append(result_entry)

# Write the answers to output.json
with open('output.json', 'w') as f:
    json.dump(output_data, f, indent=4)
print("Answers have been written to output.json")
