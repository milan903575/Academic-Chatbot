# AI-Powered Academic Document Search Assistant

A chat-first document search platform for academic institutions.
Students and staff ask natural-language questions and receive answers grounded in uploaded institutional documents such as syllabi, handouts, slides, and policy PDFs. The system also supports real-time study groups where multiple users collaborate and query the same shared corpus.

***

## Features

- Chat-based Q\&A over institutional content (syllabi, notes, policies, slides).
- Answer generation grounded in uploaded documents using OpenAI APIs and retrieval.
- Role-based access: admin can upload and manage documents; students can query and join groups.
- Real-time study groups so multiple students can chat with the same document set together.
- Containerized deployment with Docker and Docker Compose for easy setup.

***

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Flask (Python)
- **Database:** MySQL
- **AI:** OpenAI API (chat and retrieval)
- **Dev Tools:** Docker, VS Code

***

## Prerequisites

- Docker installed
- OpenAI API key
- MySQL instance or compatible MySQL service (can also run via Docker)

***

## Environment Configuration

1. In the `backend/` directory, copy the example environment file:
```bash
cd backend
cp .env.example .env
```

2. Open `.env` and replace the placeholders with your actual values:
```env
# OpenAI API Key
OPENAI_API_KEY=your-openai-api-key-here

# Flask Configuration
SECRET_KEY=your-super-secret-key-change-this-in-production
FLASK_ENV=production
PYTHONUNBUFFERED=1
PORT=5000

# MySQL Database Configuration
DB_HOST=your-database-host
DB_PORT=your-database-port
DB_USER=your-database-username
DB_PASSWORD=your-database-password
DB_NAME=your-database-name
DATABASE_URL=mysql+pymysql://your-database-username:your-database-password@your-database-host:your-database-port/your-database-name?charset=utf8mb4

# Admin Login Credentials (change in production)
ADMIN_USERNAME=your-admin-username
ADMIN_PASSWORD=your-admin-password

# CORS Configuration
ALLOWED_ORIGINS=http://localhost,http://localhost:3000,http://localhost:8080
```

Make sure the database credentials match a reachable MySQL instance and that the user has permission to create tables.

***

## Running with Docker

From the project root (where `docker-compose.yml` lives):

```bash
docker-compose build
docker-compose up -d
```

To follow backend logs:

```bash
docker-compose logs -f backend
```


***

## Accessing the Application

Once the containers are running:

- Open `http://localhost` or `http://localhost/index.html` in your browser.


### Admin flow

1. On the login screen, choose **Admin**.
2. Log in using the credentials configured in `.env` (`ADMIN_USERNAME` and `ADMIN_PASSWORD`).
3. Upload institutional documents (PDFs, slides, handouts, policies, etc.).
4. Watch `docker-compose logs -f backend` to verify that files are processed and indexed correctly.

### Student / user flow

1. Create a new **Student** user from the UI (if enabled) or use an existing account.
2. Start interacting with the chat assistant to ask questions about uploaded content.
3. Join or create a study group to collaborate with other users and query the same document set in real time.

***

## Notes

- Costs depend on OpenAI API usage; set appropriate rate limits and monitoring in production.

