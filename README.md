# Embeddings Research

A simple Python project for experimenting with embeddings using OpenAI's API.

## Overview

This repository contains a basic script that demonstrates how embeddings work by comparing similarities between different texts related to UAVs and autonomous devices. It's designed to understand the fundamentals of embeddings and semantic similarity.

## What's Included

- **`basic_embeddings.py`** - Main script that demonstrates embedding generation and similarity analysis
- **`requirements.txt`** - Python dependencies (numpy, openai, python-dotenv)
- **`.env.example`** - Template for setting up your OpenAI API key
- **`.gitignore`** - Excludes sensitive files and Python artifacts

## Features

The `basic_embeddings.py` script:

- Generates embeddings for 8 different texts related to autonomous devices
- Compares similarities between all pairs using cosine similarity
- Demonstrates how similar concepts have similar vector representations
- Shows embedding dimensions and properties
- Provides educational insights about how embeddings work

## Sample Texts Analyzed

- **UAV**: An unmanned aerial vehicle that can fly autonomously
- **drone**: A flying robot controlled remotely or by AI
- **autonomous_car**: A self-driving vehicle that uses sensors and AI
- **robot_arm**: A mechanical arm that can move and manipulate objects automatically
- **sensor**: A device that detects and measures environmental data
- **GPS**: A satellite navigation system for determining location
- **camera**: An optical device that captures visual information
- **lidar**: A laser-based sensor that measures distance using light detection

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Vect0rdecay/embeddings-research.git
   cd embeddings-research
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Run the script**:
   ```bash
   python basic_embeddings.py
   ```

## What it covered

- How embeddings convert text into numerical vectors
- How similar meanings result in similar vector representations
- How cosine similarity measures semantic similarity


## Requirements

- Python 3.7+
- OpenAI API key

## License

MIT License
