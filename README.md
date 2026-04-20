# LabMate

LabMate is a CSV-first dataset coach for students created by Sofie Jensen and Annalise Jensen. Upload a CSV and a short description of what the dataset is about, then ask what you want to know.

It will:
- suggest realistic graph options based on your question and the dataset structure
- show how to make those graphs in Python
- answer questions with actual values from the uploaded data
- help interpret the likely result pattern

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000`.

## Accepted input

- CSV file only
- Short dataset description

No PDF or manual upload is supported in this reset package.

If port 8000 is binded to another project, then use port mapping in the docker-compose.yml file.
Make sure it looks like this

```services:
  web:
    build: .
    ports:
      - "7070:8000"
    volumes:
      - .:/app
```

You can either run this file by clicking "run services" or by creating a docker container with

```
docker compose up --build
```
