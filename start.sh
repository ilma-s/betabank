#!/bin/bash

# Load environment variables from .env file
set -a  # automatically export all variables
source .env
set +a

# Start the FastAPI application
uvicorn api:app --reload --host 0.0.0.0 --port 8000 