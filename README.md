# Tower Hamlets GCSE Attainment Gaps

A data-driven application exploring gender and SEN (Special Educational Needs) attainment gaps across London boroughs and within Tower Hamlets schools.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

For a specific port (e.g. 8080):

```bash
streamlit run app.py --server.port 8080
```

## Data

Place CSV files in the `data/` directory:

- `GCSE results by sex - YYYY-YY.csv` — Borough-level gender data
- `GCSE results by SEN - YYYY-YY.csv` — Borough-level SEN data
- `TH results YYYY-YY.csv` — Tower Hamlets school-level data

## Configuration

- **DATA_DIR** (optional): Override the data directory. Default: `./data` relative to the app.
  ```bash
  set DATA_DIR=C:\path\to\data
  streamlit run app.py
  ```

## Deployment

### Docker

```bash
docker build -t tower-hamlets-gcse .
docker run -p 8080:8080 tower-hamlets-gcse
```

The app will be available at http://localhost:8080.

### Streamlit Cloud / Hugging Face Spaces

1. Push the repository to GitHub.
2. Connect to Streamlit Cloud or Hugging Face Spaces.
3. Set the run command: `streamlit run app.py --server.port 8080`

## Development

```bash
pip install -r requirements.txt
pytest tests/ -v
```
