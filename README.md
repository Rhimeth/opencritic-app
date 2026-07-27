```markdown
# GameScore AI

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GameScore AI** is a comprehensive gaming intelligence platform designed to aggregate industry news, predict critical reception, and drive game discovery. By combining dynamic data scraping with machine learning and NLP sentiment analysis, it offers an all-in-one dashboard for gaming analytics.

---

## Features

* ** Predictive Analytics:** Trains and utilizes a machine learning model to predict aggregate critic scores for upcoming releases based on historical data and sentiment trends.
* ** Smart News Aggregator:** Ingests live feeds from top gaming outlets (IGN, Polygon, Eurogamer, etc.) and utilizes transformer models to generate concise, abstractive summaries of breaking news.
* ** Developer & Game Profiling:** Provides deep-dive analytics into developer track records, franchise histories, and detailed game metadata.
* ** Discovery & Recommendations:** Leverages data from OpenCritic, RAWG, and other sources to recommend similar titles and highlight upcoming releases.

## Tech Stack

* **Language:** Python 3.10+
* **Data Sources:** OpenCritic API, RAWG API, RSS/Atom Feeds, Web Scraping (BeautifulSoup)
* **Machine Learning & NLP:** `transformers` (Summarization & Zero-Shot Classification), `sentence-transformers` (Deduplication)
* **Hosting:** Designed for seamless deployment on Hugging Face Spaces.

---

## Setup & Installation

### Prerequisites

* Python 3.10 or higher
* [Git LFS](https://git-lfs.github.com/) (required for pulling large model or data files)
* An OpenCritic API key (via RapidAPI) for live data refreshes

### Local Development

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Rhimeth/opencritic-app](https://github.com/Rhimeth/opencritic-app)
   cd opencritic-app

```

2. **Set up a virtual environment:**
```bash
python -m venv .venv

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Environment Variables:**
Create a `.env` file in the root directory and add your API keys:
```env
OPENCRITIC_API_KEY=your_rapidapi_key_here
RAWG_API_KEY=your_rawg_key_here

```


5. **Run the Application:**
```bash
# Modify "app.py" to match your specific entry script if using Streamlit, Gradio, or a basic Python script.
python app.py

```


## Deployment

This application is optimized for **Hugging Face Spaces**.

1. Create a new Space on Hugging Face.
2. Connect your GitHub repository or push directly via Git.
3. Ensure your API keys are added as **Repository Secrets** in the Hugging Face Space settings.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://www.google.com/search?q=https://github.com/Rhimeth/opencritic-app/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
