#!/bin/bash

# Code formatting script for RAG Chatbot project

set -e

echo "🔧 Running code formatting..."

echo "📝 Formatting code with Black..."
uv run black .

echo "📋 Sorting imports with isort..."
uv run isort .

echo "✅ Code formatting complete!"