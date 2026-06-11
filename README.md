---
title: GameScore AI
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# OpenCritic ML App

Python application that:

- ingests OpenCritic game data incrementally,
- trains a machine learning model to predict OpenCritic scores,
- gathers gaming news and builds short summaries,
- recommends similar games to users.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
