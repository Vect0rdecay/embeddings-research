# Embeddings Research

A Python project for learning and experimenting with embeddings using OpenAI's API.

## Overview

This repository contains two scripts that demonstrate how embeddings work by analyzing similarities between different texts related to UAVs, autonomous devices, and AI systems. It's designed to understand the fundamentals of embeddings, semantic similarity, and clustering visualization.

## What's Included

- **`basic_embeddings.py`** - Simple script demonstrating embedding generation and similarity analysis
- **`drone_embeddings_viz.py`** - Advanced visualization script with t-SNE clustering
- **`requirements.txt`** - Python dependencies (numpy, openai, python-dotenv, matplotlib, seaborn, scikit-learn)
- **`.env.example`** - Template for setting up your OpenAI API key
- **`.gitignore`** - Excludes sensitive files and Python artifacts

## Features

### Basic Embeddings Script (`basic_embeddings.py`):
- Generates embeddings for 8 different texts related to autonomous devices
- Compares similarities between all pairs using cosine similarity
- Demonstrates how similar concepts have similar vector representations
- Shows embedding dimensions and properties

### Drone Visualization Script (`drone_embeddings_viz.py`):
- Analyzes 24 drone and AI concepts across 4 categories (Hardware, Sensors, AI Systems, Drone Types)
- Creates 2D visualizations using t-SNE dimensionality reduction
- Implements cluster analysis and similarity verification
- Generates high-quality PNG visualizations

## Sample Concepts Analyzed

### Basic Script (8 concepts):
- **UAV**: An unmanned aerial vehicle that can fly autonomously
- **drone**: A flying robot controlled remotely or by AI
- **autonomous_car**: A self-driving vehicle that uses sensors and AI
- **robot_arm**: A mechanical arm that can move and manipulate objects automatically
- **sensor**: A device that detects and measures environmental data
- **GPS**: A satellite navigation system for determining location
- **camera**: An optical device that captures visual information
- **lidar**: A laser-based sensor that measures distance using light detection

### Visualization Script (24 concepts across 4 categories):

**Hardware Components:**
- propeller, motor, battery, frame, landing_gear, gimbal

**Sensors:**
- camera, lidar, gps, imu, altimeter, compass

**AI Systems:**
- flight_controller, path_planner, obstacle_detection, object_recognition, autonomous_navigation, mission_planner

**Drone Types:**
- quadcopter, fixed_wing, hexacopter, delivery_drone, surveillance_drone, agricultural_drone

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

5. **Run the scripts**:
   ```bash
   # Basic analysis
   python basic_embeddings.py
   
   # Advanced visualization
   python drone_embeddings_viz.py
   ```

## Requirements

- Python 3.7+
- OpenAI API key

## License

MIT License
