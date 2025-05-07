#!/bin/bash

# Exit on error
set -e

# Wait for dependencies
echo "Waiting for PostgreSQL..."
while ! nc -z db 5432; do
  sleep 1
done

echo "Waiting for Redis..."
while ! nc -z redis 6379; do
  sleep 1
done

# Run migrations
alembic upgrade head

# Start the continuous operation
if [ "$RUNTIME_MODE" = "worker" ]; then
    python continuous_operation.py
else
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
fi