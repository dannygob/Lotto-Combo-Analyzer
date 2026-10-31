# Quick Start

Follow these steps to run the Lotto-Combo-Analyzer API and, optionally, its web interface.

## What you'll need

- Docker Desktop installed and running.
- Python installed if you want to serve the web interface locally.

## 1. Open a terminal in the project folder

Run the commands below from the folder that contains `docker-compose.yml`.

If you still need to download the project:

```powershell
git clone https://github.com/dannygob/Lotto-Combo-Analyzer.git
cd Lotto-Combo-Analyzer
```

## 2. Start the API

```powershell
docker compose up -d
```

The first run may take a few minutes while Docker downloads the base image and installs dependencies.

## 3. Check that the API is running

Open the interactive API docs:

<http://localhost:8000/docs>

Or check the health endpoint in PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

You should see a response with `"status": "ok"`. To inspect the container logs:

```powershell
docker logs -f lotto-combo-analyzer
```

Press `Ctrl+C` to stop following the logs; this does not stop the container.

## 4. (Optional) Start the web interface

The frontend is a static site and is not started by Docker Compose. Open a **second terminal** in the project folder and run:

```powershell
python -m http.server 8080 --directory .\frontend
```

Keep that terminal open, then visit:

<http://localhost:8080/>

The interface calls the API at `http://localhost:8000`. The API is configured to allow the local frontend origins `http://localhost:8080` and `http://127.0.0.1:8080`. If the status still says the API is disconnected, make sure you opened the page at one of those addresses and refresh it.

## Suggested workflow

The dashboard shows an optional four-step guide: register a lottery, select a CSV and confirm the draw format, map the columns, import the history, then explore schedules and analyses. You can switch tabs freely at any time. A CSV can contain a draw number/ID, draw date, the drawn numbers, and other columns. The import screen suggests likely columns, but you choose each column's role: draw date, draw ID, drawn numbers, extra information to keep, or a column to ignore. If there is no date column, confirm that rows are ordered from oldest to newest.

The suggested number range and count come from values observed in the CSV, not from official lottery rules. Verify those values before saving the draw format. Extra columns can be kept as attributes or ignored; they are not treated as drawn numbers unless you assign them that role.

From the registered-lotteries list, you can edit a lottery's name or delete it. The app creates an internal ID from the name automatically. Deleting a lottery permanently removes its related rules, draws, imports, and models; the interface asks for confirmation first.

## Stop the API

From the project folder, run:

```powershell
docker compose down
```

To restart only the API container:

```powershell
docker restart lotto-combo-analyzer
```