# Setup Guide

## 1. Prerequisites

* **Docker & Docker Compose** installed
* **Python 3.9+** installed (if you want to run locally without Docker)
* Create a `.env` file in the project root (you can copy from `.env.example`).
  This file provides environment variables used to configure **Neo4j** and the app.

Example values are included in `.env.example`.

## 2. Setup with Docker (**Recommended**)

### Method A: Automatic (using helper script)

Run everything with one command:

```bash
chmod +x ./scripts/docker-up.sh
./scripts/docker-up.sh
```

This script will:

* Start or reuse a **Neo4j** container
* Build the **app image**
* Start the app container

The API will be available at:
`http://localhost:8000`

### Method B: Manual (run containers step by step)

**1. Start Neo4j**

```bash
docker run -d --name neo4j-graphiti \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/testpass \
  -v $PWD/neo4j_data:/data \
  neo4j:5.26
```

If already created, just start it:

```bash
docker start neo4j-graphiti
```

**2. Build the app**

```bash
docker build -t hugo-app .
```

**3. Run the app**

```bash
docker run -d --name hugo-app -p 8000:8000 hugo-app:latest
```

The API will be available at:
`http://localhost:8000`

---

## 3. Setup without Docker (Manual Local Setup)

1. **Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Start Neo4j** (requires Docker installed)

```bash
docker run -d --name neo4j-graphiti \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/testpass \
  -v $PWD/neo4j_data:/data \
  neo4j:5.26
```

If already created:

```bash
docker start neo4j-graphiti
```

4. **Run the app**

```bash
nohup python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
```

The API will be available at:
`http://localhost:8000`

