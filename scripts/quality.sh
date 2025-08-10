#!/bin/bash

# Complete code quality workflow for RAG Chatbot project

set -e

echo "🚀 Running complete code quality workflow..."
echo

# Format code
echo "Step 1/3: Formatting code..."
./scripts/format.sh
echo

# Run linting checks
echo "Step 2/3: Running linting checks..."
./scripts/lint.sh
echo

# Run tests
echo "Step 3/3: Running tests..."
uv run pytest backend/tests/ -v
echo

echo "🎉 Code quality workflow completed successfully!"