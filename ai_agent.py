import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Literal
from typing_extensions import TypedDict
import os
from langchain.schema import Document
from langchain.chains import RetrievalQA
import tempfile



os.environ["AZURE_OPENAI_API_KEY"] = "your API KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = "Your Endpoint"
os.environ["OPENAI_API_VERSION"] = "Your API Version"

st.title("🔍Multi Ai Agents RAG with LangGraph")
st.sidebar.header("Upload PDF and Configure")

embeddings = AzureOpenAIEmbeddings(
    model="Embedding-ada",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"]
)




# Upload file in sidebar
def read_files(uploaded_files):
    
    docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
            docs.extend(loader.load())
        
        os.remove(tmp_file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_db")
    st.success("Documents have been processed and stored in the FAISS vector database successfully!")

uploaded_files = st.sidebar.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    if st.sidebar.button("Submit"):
        retriever=read_files(uploaded_files)



GPT_DEPLOYMENT_NAME = "gpt-35-turbo-16k"
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=["OPENAI_API_VERSION"],
    azure_deployment=GPT_DEPLOYMENT_NAME, max_tokens=12000
)

# Define routing logic for the LLM
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search", "arxiv_search"] = Field(
        ..., description="Given a user question, route to Wikipedia, vectorstore, or arxiv."
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or wikipedia or arxiv.
The vectorstore contains documents related to patents related frequently asked question
Use the vectorstore for questions on these topics. Otherwise, use wiki_search or arxiv_search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

# Function for question routing
def route_question(state):
    st.write(("---ROUTE QUESTION---"))
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        st.write("---ROUTE QUESTION TO Wiki SEARCH---")
        return "wiki_search"
    elif source.datasource == "vectorstore":
        st.write("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    elif source.datasource == "arxiv_search":
        st.write("---ROUTE QUESTION TO ARXIV SEARCH---")
        return "arxiv_search"

# Define the StateGraph for LangGraph
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]



# Define retrieval functions for vectorstore, Wikipedia, and arXiv
def retrieve(state):
    question = state["question"]
    #preprocessed_question = preprocess_question(question)
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()

    qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever
    )
    documents = qa({"query": question})

    return {"documents": documents, "question": question}

def wiki_search(state):
    question = state["question"]
    docs = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)).invoke({"query": question})
    #return {"documents": page_content, "question": question}
    return {"documents": Document(page_content=docs), "question": question}

def arxiv_search(state):
    question = state["question"]
    docs = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000)).invoke({"query": question})
    #return {"documents": page_content, "question": question}
    return {"documents": Document(page_content=docs), "question": question}

# Build the workflow with nodes and edges
workflow = StateGraph(GraphState)
workflow.add_node("wiki_search", wiki_search)
workflow.add_node("arxiv_search", arxiv_search)
workflow.add_node("retrieve", retrieve)
workflow.add_conditional_edges(
    START, route_question, {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
        "arxiv_search": "arxiv_search"
    }
)
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)
workflow.add_edge("arxiv_search", END)
app = workflow.compile()

# Display LangGraph image
st.sidebar.header("Generate LangGraph Visualization")
if st.sidebar.button("Show LangGraph Image"):
    with st.spinner("Generating LangGraph visualization..."):
        
        st.image(app.get_graph().draw_mermaid_png())
        st.markdown("**Visual representation of the system's workflow, crafted with LangGraph for enhanced clarity.**")

st.sidebar.header("Ask a Question")
question = st.sidebar.text_input("Type your question here:")

if question:
    
    inputs = {
        "question": question
    }
    
    for output in app.stream(inputs):
        for key, value in output.items():

            st.write(f"Node '{key}':")
        st.write("ANSWER:")
    st.write(value['documents'])
            

