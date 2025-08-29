#!/usr/bin/env python3
"""
A simple script to understand how embeddings work
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

# Load environment variables (API key)
load_dotenv()

# Create OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-ada-002"):
    """Get embedding for a text using OpenAI API"""
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def main():
    print("=== EMBEDDINGS TESTS ===\n")
    
    # Sample texts to compare
    texts = {
        "UAV": "An unmanned aerial vehicle that can fly autonomously",
        "drone": "A flying robot controlled remotely or by AI", 
        "autonomous_car": "A self-driving vehicle that uses sensors and AI",
        "robot_arm": "A mechanical arm that can move and manipulate objects automatically",
        "sensor": "A device that detects and measures environmental data",
        "GPS": "A satellite navigation system for determining location",
        "camera": "An optical device that captures visual information",
        "lidar": "A laser-based sensor that measures distance using light detection"
    }
    
    print("Step 1: Getting embeddings for each text...")
    embeddings = {}
    
    for name, text in texts.items():
        print(f"  Getting embedding for '{name}': {text}")
        embedding = get_embedding(text)
        embeddings[name] = embedding
        print(f"    → Embedding has {len(embedding)} dimensions")
        print(f"    → First 5 values: {embedding[:5]}")
        print()
    
    print("Step 2: Comparing similarities...")
    print("(Higher values = more similar meanings)\n")
    
    # Compare all pairs
    items = list(texts.keys())
    for i, item1 in enumerate(items):
        for item2 in items[i+1:]:
            similarity = cosine_similarity(embeddings[item1], embeddings[item2])
            print(f"{item1} vs {item2}: {similarity:.4f}")
    
    print("\nTL;DR")
    print("• Embeddings convert text into numbers (vectors)")
    print("• Similar meanings have similar vectors")
    print("• Flying devices (UAV, drone) are more similar to each other")
    print("• Sensors (camera, lidar, GPS) are more similar to each other")
    print("• Autonomous vehicles (UAV, autonomous_car) are more similar to each other")
    print("• Different categories (sensors vs vehicles) are less similar to each other")
    print("• Cosine similarity measures how similar two vectors are")
    print("• Values range from -1 to 1, where 1 = identical")

if __name__ == "__main__":
    main()
