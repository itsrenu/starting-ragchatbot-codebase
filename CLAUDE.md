# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- **Start the server**: `./run.sh` (recommended) or `cd backend && uv run uvicorn app:app --reload --port 8000`
- **Install dependencies**: `uv sync`
- **Make run script executable**: `chmod +x run.sh`

### Environment Setup
- Create `.env` file in root with: `ANTHROPIC_API_KEY=your_anthropic_api_key_here`
- Python 3.13+ required
- Uses `uv` package manager (not pip/conda)

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for course materials with the following key architectural patterns:

### Core Components Structure
- **RAGSystem** (`backend/rag_system.py`): Main orchestrator that coordinates all components
- **FastAPI App** (`backend/app.py`): REST API server serving both API endpoints and static frontend
- **Vector Store** (`backend/vector_store.py`): ChromaDB-based semantic search using sentence transformers
- **AI Generator** (`backend/ai_generator.py`): Anthropic Claude integration with tool-based search
- **Document Processor** (`backend/document_processor.py`): Chunks documents into course metadata and content

### Tool-Based Search Architecture
The system uses a **tool-based approach** where Claude can call search functions:
- `CourseSearchTool` in `search_tools.py` performs semantic searches
- `ToolManager` handles tool registration and execution
- AI responses are generated with access to search tools rather than pre-retrieved context

### Data Models
- **Course**: Represents course metadata (title, instructor, description)
- **CourseChunk**: Text chunks with course association for vector storage
- **Session Management**: Conversation history tracking per user session

### Key Design Patterns
- **Component Separation**: Each major function (embedding, search, AI generation) is in separate modules
- **Configuration-Driven**: All settings centralized in `config.py` using dataclasses
- **Tool Integration**: AI uses function calling to search rather than RAG retrieval patterns
- **Startup Document Loading**: Documents in `/docs` folder auto-loaded on server startup

### Frontend Architecture
- **Simple HTML/CSS/JS**: No framework, served as static files by FastAPI
- **Session-based**: Maintains conversation context across queries
- **Real-time Stats**: Shows course count and titles dynamically

## Development Notes

### Document Processing
- Supports PDF, DOCX, TXT files in the `docs/` folder  
- Documents are chunked with configurable size (800 chars) and overlap (100 chars)
- Course metadata extracted and stored separately from content chunks

### Vector Storage
- Uses ChromaDB with `all-MiniLM-L6-v2` sentence transformer
- Two collections: course metadata and course content chunks
- Persistent storage in `./chroma_db/` directory

### API Endpoints
- `POST /api/query`: Main chat endpoint with session support
- `GET /api/courses`: Returns course statistics and titles
- Static files served at root path for frontend

### Claude Configuration
- Model: `claude-sonnet-4-20250514`
- Tool-based architecture allows Claude to search course content
- System prompt optimized for educational responses with search capabilities