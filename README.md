---
title: GameScore AI
sdk: docker
pinned: false
---

# GameScore AI

GameScore AI is a comprehensive gaming intelligence platform that:

- Ingests game data from OpenCritic, RAWG, and other sources.
- Trains a machine learning model to predict critic scores for upcoming games.
- Gathers gaming news and builds short summaries.
- Recommends similar games to users.
- Provides detailed game and developer profiles with analytics.

## Setup

### Prerequisites

- Python 3.10+
- Git LFS (for large data files)
- OpenCritic API key (RapidAPI) for data refresh

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Rhimeth/opencritic-app
cd opencritic-app
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
