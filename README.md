# Academic Document Search Assistant

> A chat-first semantic search platform that lets students and staff query institutional documents using natural language — powered by OpenAI and real-time collaborative study groups.

---

## Overview

Academic Document Search Assistant is a full-stack AI application built for academic institutions. It enables students and staff to ask natural-language questions and receive contextual answers grounded in uploaded institutional content — syllabi, handouts, lecture slides, and policy documents. The platform also supports real-time study groups where multiple users collaboratively query a shared document corpus.

---

## Features

- **Semantic Q&A** — Ask natural-language questions; get answers grounded in uploaded institutional documents
- **Document Management** — Admins upload and manage PDFs, slides, handbooks, and policies
- **Role-Based Access Control** — Separate Admin and Student roles with distinct permissions
- **Real-Time Study Groups** — Multiple students can query the same document set collaboratively via live chat
- **Containerized Deployment** — Fully Dockerized setup via Docker Compose for consistent environments

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Flask (Python) |
| Database | MySQL |
| AI / Retrieval | OpenAI API (GPT-4), FAISS |
| Real-Time | SocketIO, WebSockets |
| Infrastructure | Docker, Docker Compose |

---

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose installed
- An active [OpenAI API key](https://platform.openai.com/account/api-keys)
- A reachable MySQL instance (can be run via Docker)

---

### Environment Setup

Navigate to the `backend/` directory and create a `.env` file from the provided example:

```bash
cd backend
cp .env.example .env
```

Open `.env` and fill in your credentials:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Flask
SECRET_KEY=your_secret_key
FLASK_ENV=production
PYTHONUNBUFFERED=1
PORT=5000

# MySQL
DB_HOST=your_db_host
DB_PORT=3306
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=your_db_name
DATABASE_URL=mysql+pymysql://your_db_user:your_db_password@your_db_host:3306/your_db_name?charset=utf8mb4

# Admin credentials
ADMIN_USERNAME=your_admin_username
ADMIN_PASSWORD=your_admin_password

# CORS
ALLOWED_ORIGINS=http://localhost,http://localhost:3000,http://localhost:8080
```

> The MySQL user requires `CREATE`, `SELECT`, `INSERT`, `UPDATE`, and `DELETE` privileges on the configured database.

---

### Running with Docker

From the project root:

```bash
docker-compose build
docker-compose up -d
```

To stream backend logs:

```bash
docker-compose logs -f backend
```

The application will be available at `http://localhost`.

---

## Usage

### Admin

1. Navigate to `http://localhost` and log in with Admin credentials
2. Upload institutional documents (PDFs, lecture slides, handbooks, policies)
3. Uploaded files are automatically processed and indexed for semantic search

### Student

1. Log in or register as a Student
2. Use the chat interface to ask questions about any uploaded document
3. Create or join a study group to query documents collaboratively in real time

---
## Screenshots

### Student — Chat Interface
<img width="1471" height="971" alt="Image" src="https://github.com/user-attachments/assets/2434ffad-abbe-4599-a731-3cd70afd4908" />

### Real-Time Study Group
<img width="1796" height="863" alt="Image" src="https://github.com/user-attachments/assets/08ddad53-73fb-452a-ac7c-091dc5b1ede8" />

> Additional screenshots and a full feature walkthrough are available on the [portfolio](#).
## Notes

- API usage costs are determined by OpenAI's pricing. 
