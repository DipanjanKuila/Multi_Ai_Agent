# AI Agent with LangGraph and Azure Integration

This project implements an AI-based question-answering agent using LangChain, LangGraph, and Azure OpenAI. It supports document uploads, advanced question routing, and integration with external sources like Wikipedia and Arxiv to enrich responses.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
- [License](#license)

## Overview

The `ai_agent.py` script is a Streamlit-based application for handling information retrieval and question-answering workflows. The agent processes uploaded documents, stores them in a FAISS vector database, and can route queries to different sources based on the topic. It uses LangGraph to structure and visualize the query-routing flow.

## Features

- **Document Upload and Processing**: Upload PDF documents to create embeddings stored in a FAISS vector database.
- **Azure OpenAI Integration**: Uses Azure OpenAI for generating embeddings and interacting with an AI language model.
- **Dynamic Query Routing**: Routes questions to the appropriate data source, including Wikipedia, Arxiv, or the FAISS database.
- **External Data Sources**: Fetches additional information from Wikipedia and Arxiv APIs for enhanced responses.
- **Workflow Visualization**: Generates a LangGraph workflow visualization to show the AI agent's routing logic.

## Usage

1. **Upload Documents**:
   - Navigate to the "Upload PDF and Configure" section on the sidebar.
   - Upload PDF or text files, which are processed and stored in the FAISS vector database.

2. **Ask Questions**:
   - Type a question into the input box on the sidebar.
   - The system determines whether to route the question to the vector store, Wikipedia, or Arxiv based on relevance.

3. **View LangGraph Visualization**:
   - Generate a graphical representation of the workflow by selecting "Show LangGraph Image" on the sidebar.

## Code Breakdown

### Key Components

1. **Environment Setup**:
   - Imports necessary modules, sets up environment variables for Azure integrations, and initializes Streamlit components.

2. **Embeddings Initialization**:
   - Sets up `AzureOpenAIEmbeddings` with a model to process document embeddings.

3. **Document Processing**:
   - Function `read_files(uploaded_files)`:
     - Accepts uploaded files, reads them, processes them using `RecursiveCharacterTextSplitter`, and saves document chunks in a FAISS vector database.
     - This function can process both PDF and text files.

4. **Language Model Setup**:
   - Uses Azure Chat OpenAI (`AzureChatOpenAI`) as the core language model with parameters such as model name and token limits.

5. **Question Routing**:
   - Function `route_question(state)`:
     - Routes questions to the appropriate data source based on relevance. Options include Wikipedia, the FAISS vector database, and Arxiv.
   - Class `RouteQuery`:
     - Defines routing options for query destinations, helping the structured model decide where to search for information.

6. **Data Retrieval Functions**:
   - `retrieve(state)`: Uses FAISS database for questions related to processed documents.
   - `wiki_search(state)`: Fetches information from Wikipedia based on the query.
   - `arxiv_search(state)`: Retrieves relevant articles from Arxiv.

7. **Workflow Definition with LangGraph**:
   - Defines and structures query routing logic using `StateGraph` with nodes for different data sources.
   - Nodes and edges are added to form a complete workflow that is visualized and executed in real-time.

8. **LangGraph Visualization**:
   - Generates a graphical view of the workflow, showing the question-routing structure and connections between nodes.

### Workflow Execution

- The system initiates a query flow based on user input, sending it through various nodes based on the routing decisions.
- Answers are displayed in the main interface, and LangGraph's graphical output shows the query processing stages.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
