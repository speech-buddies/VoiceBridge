# VoiceBridge Database (PostgreSQL)

Simple schema: **users** (customizations), **conversations** (user ↔ agent).

## Tables

| Table           | Purpose                                                                                                             |
| --------------- | ------------------------------------------------------------------------------------------------------------------- |
| `users`         | One row per person. `preferences` = theme, verbosity, ASR settings (JSONB). `consent` = consent flags (JSONB).      |
| `conversations` | One row per conversation. `messages` = JSONB array of `{ "role": "user"\|"agent", "content": "...", "at": "..." }`. |

## How to set up the database

### 1. Install PostgreSQL

- **Windows:** [Download](https://www.postgresql.org/download/windows/) and run the installer. Add the `bin` folder (e.g. `C:\Program Files\PostgreSQL\16\bin`) to your PATH so `psql` works.
- **macOS:** `brew install postgresql`
- **WSL/Linux:** `sudo apt install postgresql postgresql-client`

Start the PostgreSQL service if it isn’t running (e.g. Windows: Services → start “postgresql”, or from install directory: `pg_ctl start`).

### 2. Create a database

Open a terminal and connect as the postgres user, then create the database:

```bash
psql -U postgres -c "CREATE DATABASE voicebridge;"
```

(Use your actual PostgreSQL username if it’s not `postgres`. You’ll be prompted for the password.)

### 3. Set your connection string

Replace `YOUR_PASSWORD` and adjust host/port if needed:

```bash
# Windows (PowerShell)
$env:DATABASE_URL = "postgresql://postgres:YOUR_PASSWORD@localhost:5432/voicebridge"

# Windows (CMD)
set DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/voicebridge

# Linux / macOS / WSL
export DATABASE_URL="postgresql://postgres:YOUR_PASSWORD@localhost:5432/voicebridge"
```

For a permanent config, put the same line in a `.env` file in the project root (and add `.env` to `.gitignore`).

### 4. Run the schema

From the **project root** (`VoiceBridge`):

```bash
psql "$DATABASE_URL" -f src/database/schema.sql
```

Or with explicit options (you’ll be prompted for password):

```bash
psql -h localhost -U postgres -d voicebridge -f src/database/schema.sql
```

You should see `CREATE TABLE` for `users` and `conversations` with no errors. The database is ready to use.

### 5. Use it from your app

In your backend (e.g. `server.py`), read `DATABASE_URL` from the environment and use it with a PostgreSQL driver (e.g. `psycopg2`, `asyncpg`, or SQLAlchemy). Encrypt sensitive values in your app (M13) before writing.
