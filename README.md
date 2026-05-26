# slm-qa-engine

A lightweight question-answering system powered by small language models (Gemma 2), semantic embeddings, and retrieval-augmented generation (RAG). Load documents, build vector indexes, and query with context-aware responses. Built for Google Colab and HuggingFace integration.

## Features

- **Document Processing** - Load and index PDF documents efficiently
- **Semantic Search** - Find relevant context using HuggingFace embeddings
- **Context-Aware Q&A** - Generate answers based only on provided context
- **Multi-turn Conversations** - Chat engine with conversation history

## Installation

### 1. Setup in Google Colab

Run the following in a Colab cell to install all dependencies:

```python
!pip install -q llama-index llama-index-llms-huggingface llama-index-embeddings-huggingface sentence-transformers transformers accelerate torch llama-index-readers-file pymupdf
```

### 2. Login to Hugging Face

```python
from huggingface_hub import notebook_login
notebook_login()
```

## Quick Start

### Basic Usage - Query Engine
```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Load and index a PDF
documents = SimpleDirectoryReader(input_files=["path/to/document.pdf"]).load_data()
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine()

# Ask questions
response = query_engine.query("What are the main topics?")
print(response)
```

### Chat Mode - Multi-turn Conversations
```python
# Create chat engine
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# First question
r1 = chat_engine.chat("Please list main AI applications")
print(r1)

# Follow-up question (maintains context)
r2 = chat_engine.chat("Can you elaborate on item 3?")
print(r2)
```

## Project Structure

```
slm-qa-engine/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── Data/                               # Document storage
│   └── Artificial intelligence - Wikipedia.pdf
├── SLM_Building_an_Intelligent_Q&A_System.ipynb
├── SLM_Fundamentals.ipynb
└── notebooks/                          # Additional notebooks
    └── examples/                       # Usage examples
```

## Notebooks

### 1. `SLM_Building_an_Intelligent_Q&A_System.ipynb`
Complete end-to-end pipeline:
- Environment setup and Gemma 2 loading
- Document indexing with semantic embeddings
- Query engine demonstration
- Chat engine with multi-turn conversations

### 2. `SLM_Fundamentals.ipynb`
Foundational concepts and model exploration

## Configuration

### Adjust Model Settings
Edit LLM parameters in the notebook:
```python
llm = HuggingFaceLLM(
    model=model_gemma,
    tokenizer=tokenizer_gemma,
    context_window=8192,           # Max context length
    max_new_tokens=500,            # Response length
    generate_kwargs={
        "temperature": 0.2,        # Lower = more deterministic
        "do_sample": True
    }
)
```

### Embedding Configuration
```python
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.chunk_size = 512          # Document chunk size
```

## Performance Tips

Optimization Strategies
- Use GPU with `device_map="auto"` for faster inference
- Reduce `max_new_tokens` for quicker responses
- Adjust `chunk_size` (256-1024) based on document type
- Lower `temperature` (0-0.5) for factual answers
- Use CPU offloading if VRAM is limited

## Limitations

Important Considerations
- Responses based only on provided context (no external knowledge)
- Gemma 2 2B is smaller than GPT-style models (quality trade-off)
- Requires HuggingFace token for model access
- No persistent storage between sessions
- Single-document focus (extend for multiple documents)

## Roadmap

- [ ] Multi-document handling
- [ ] Persistent vector database (Pinecone, Weaviate)
- [ ] Web UI with Streamlit/Gradio
- [ ] Support for additional document formats (DOCX, TXT, HTML)
- [ ] Model quantization for faster inference
- [ ] Batch processing capabilities

## Troubleshooting

### PDF not loading?
```python
# Verify file path is correct
import os
print(os.path.exists("Data/Artificial intelligence - Wikipedia.pdf"))
```

### Out of memory?
- Reduce `context_window` to 4096
- Use smaller embedding model
- Process documents in batches
- Enable CPU offloading in device_map

### Slow responses?
- Use GPU instead of CPU
- Reduce `max_new_tokens`
- Decrease `chunk_size` for faster retrieval
- Check system resource usage

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Acknowledgments

- [HuggingFace](https://huggingface.co) for Transformers and model hub
- [LlamaIndex](https://www.llamaindex.ai) for RAG framework
- [Google](https://www.google.com/intl/en/about/) for Gemma 2 model
- [Sentence Transformers](https://www.sbert.net) for embeddings


---

Built Q&A systems with small language models on HuggingFace and Google Colab
