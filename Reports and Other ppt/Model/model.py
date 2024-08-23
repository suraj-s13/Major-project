
# !pip install langchain
# !pip install faiss-cpu
# !pip install sentence-transformers

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from config import HUGGINGFACEHUB_API_TOKEN
import sentence_transformers
import textwrap
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

loader = TextLoader("SampleText.txt")

document = loader.load()

#print(document)

#pre-processing of the data

def wrap_text_preserve_newlines(text, width=110):
  #split the input text into lines based on newline characters
  lines = text.split('\n')

  #wrap each line individually
  wrapped_lines = [textwrap.fill(line, width = width) for line in lines]

  # Join the wrapped llines back together using newline characters
  wrapped_text = '\n'.join(wrapped_lines)

  return wrapped_text

wrap_text_preserve_newlines(str(document[0]))

#Text Splitting
text_splitter = CharacterTextSplitter(chunk_size =1000, chunk_overlap = 0)
docs = text_splitter.split_documents(document)

# print(docs[0])
# print(len(docs))

#Embedding

embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

query = " Who is suraj singh?"
doc = db.similarity_search(query)

wrap_text_preserve_newlines(str(doc[0].page_content))

#Q-A

llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.8, "max_length":512})

chain = load_qa_chain(llm, chain_type="stuff")

queryText = " Tell me about suraj Singh"
docsResult = db.similarity_search(queryText)
chain.run(input_documents = docsResult, question = queryText)

