#!/bin/bash

# Code linting script for RAG Chatbot project

set -e

echo "🔍 Running code quality checks..."

echo "📝 Checking code style with flake8..."
uv run flake8 backend/

echo "🔧 Checking formatting with Black..."
uv run black --check .

echo "📋 Checking import order with isort..."
uv run isort --check-only .

echo "✅ All code quality checks passed!"