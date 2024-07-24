from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Load and split the documents
documents = TextLoader(
    "RR1_i9_startup-config.txt",
).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
for idx, text in enumerate(texts):
    text.metadata["id"] = idx

# Define a function to pretty print documents
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# Create embeddings and retriever
embedding = OllamaEmbeddings(model="nomic-embed-text")
retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 10})

# Perform the query
query = "What can you tell me about this router and what protocols are running on it?"
docs = retriever.invoke(query)

# Pretty print the retrieved documents
pretty_print_docs(docs)

# Set up the contextual compression retriever
llm = Ollama(model="llama3")
compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# Perform the compressed retrieval query
compressed_docs = compression_retriever.invoke(
    "What can you tell me about this router and what protocols are running on it?"
)
print([doc.metadata["id"] for doc in compressed_docs])

# Create and execute the QA chain
chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever)
response = chain({"query": "What can you tell me about this router and what protocols are running on it?"})

# Print the response from the chain
print(response)

'''
Output (Cleaned):
{'query': 'What can you tell me about this router and what protocols are running on it?', 'result': 'Based on the provided configuration, I can tell you that:\n\n*
 The router is a Cisco router (likely an IOS-based router).\n* The router has multiple interfaces:
 \n\t+ Loopback0: This interface has an IP address of 1.0.0.5/32 and is configured for OSPF.
 \n\t+ FastEthernet0/0: This interface has an IP address of 10.0.0.5/24 and is also configured for OSPF.
 \n\t+ FastEthernet0/1: This interface is currently shut down, but it\'s connected to the CE1 Router (as indicated by the description).
 \n\t+ FastEthernet1/0, FastEthernet2/0: These interfaces are not enabled (shutdown) and have no IP address assigned.\n
 * The router is running OSPF (Open Shortest Path First) with a process ID of 1. It has two network statements:\n
 \t+ One for the 1.0.0.5/32 subnet, which is in area 0.\n
 \t+ One for the 10.0.0.0/25 subnet, which is also in area 0.\n
 * The router is running BGP (Border Gateway Protocol) with an autonomous system number of 5000. It has multiple neighbor statements:\n
 \t+ Neighbors 1.0.0.1, 1.0.0.2, 1.0.0.3, 1.0.0.4, and 1.0.0.6 are all connected to the router and running BGP.\n
 \t+ Neighbor 1.0.0.7 is also connected to the router but has not been explicitly configured (no update-source specified).\n
 * The router has some additional configuration:\n\t+ The username is set to "admin" with a password of 104F0D140C19 (hashed in Cisco\'s proprietary format).
 \n\t+ Archive logging is enabled, hiding sensitive information like passwords.
 \n\t+ The TCP SYN wait time is set to 5 seconds, and SSH version 2 is enabled.
 \n\nOverall, this router appears to be part of a larger network infrastructure, providing routing services for multiple networks using OSPF and BGP.'}

'''
