#!/bin/bash

# Code linting script with mypy type checking for RAG Chatbot project
# Note: mypy may report type issues that require manual fixing

set -e

echo "🔍 Running code quality checks with type checking..."

echo "📝 Checking code style with flake8..."
uv run flake8 backend/

echo "🔧 Checking formatting with Black..."
uv run black --check .

echo "📋 Checking import order with isort..."
uv run isort --check-only .

echo "🔍 Type checking with mypy..."
uv run mypy backend/ --ignore-missing-imports --exclude "backend/tests/"

echo "✅ All code quality checks with type checking passed!"