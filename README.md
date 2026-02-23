---
title: Diagnostic Copilot API
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
python_version: "3.10"
---

# Multi-Modal Diagnostic Co-Pilot Model API

This repository contains the training scripts and the Gradio API endpoint for the Diagnostic Co-Pilot web application.

## Endpoints

- `/predict`: Accepts clinical history, examination text, and multimodal files (images) to generate a structured clinical Markdown report.
