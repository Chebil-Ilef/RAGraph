#!/usr/bin/env bash
set -euo pipefail

# Pick compose command
if command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
elif docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
else
  echo "Error: docker compose/ docker-compose not found" >&2
  exit 2
fi

# Stable project name
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-hugo}"

# Clean up any orphaned containers from previous runs
"${COMPOSE[@]}" down --remove-orphans 2>/dev/null || true

# Build and start containers with proper dependency handling
echo "Building and starting containers..."
"${COMPOSE[@]}" up --build -d

# Wait for neo4j to be ready before starting the app
echo "Waiting for Neo4j to be ready..."
timeout 60s bash -c '
  while ! docker compose logs neo4j 2>&1 | grep -q "Bolt enabled"; do
    sleep 2
  done
' || echo "Neo4j may still be starting..."

echo "Containers are running:"
"${COMPOSE[@]}" ps