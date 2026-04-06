# Installing and running VoiceBridge

## Prerequisites

- **Git** — to clone the repository  
- **Python 3.10+** (3.11 recommended for PyTorch compatibility)  
- **Node.js** 18 LTS or newer and **npm** — for the React frontend  
- **Microphone access** — for speech capture (OS/browser permissions)

Optional, depending on features you enable:

- **API keys** — place or update `src/.env` (and `src/app/.env` if your team uses it) with the keys your project expects (for example browser automation or LLM providers). Do not commit real keys.

---

## 1. Clone from GitHub

Replace the URL with your team’s repository or fork:

```bash
git clone https://github.com/speech-buddies/voicebridge.git
cd voicebridge
```

---

## 2. Backend (FastAPI)

The API listens on **port 8000** by default (`http://localhost:8000`). Run commands from a terminal.

### Create a virtual environment (recommended)

**Windows (PowerShell or Command Prompt):**

```powershell
cd src
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
cd src
python3 -m venv .venv
source .venv/bin/activate
```

### Install Python dependencies

Still inside `src` with the virtual environment activated:

```bash
pip install -r requirements.txt
```

### Start the server

From the **`src`** directory (so imports such as `server:app` resolve correctly):

```bash
python server.py
```

Alternatively:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Leave this process running while you use the app. The frontend expects the backend at `http://localhost:8000`.

---

## 3. Frontend (React)

Open a **second** terminal. The UI lives under `src/presentation`.

### Install and run the development server

```bash
cd src/presentation
npm install
npm start
```

By default, Create React App serves the app at **http://localhost:3000**. The UI is configured to talk to the backend at `http://localhost:8000` (see `src/presentation/src/App.js`).

### Browser extension build (optional)

To build the packaged extension instead of using the dev server:

```bash
cd src/presentation
npm run build:extension
```

Then load the unpacked extension from the output folder your team uses with Chrome (see `src/presentation/README.md`).

---

## Quick checklist

1. Clone repo → `cd voicebridge`  
2. Backend: `cd src` → venv → `pip install -r requirements.txt` → `python server.py`  
3. Frontend: `cd src/presentation` → `npm install` → `npm start`  
4. Open the frontend URL and ensure the backend is still running on port **8000**
