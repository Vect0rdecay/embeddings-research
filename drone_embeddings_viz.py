#!/usr/bin/env python3
"""
Drone Embeddings Visualization
Visualize embeddings for drone components and AI systems using t-SNE

This script helps me understand how AI "sees" relationships between different drone parts and systems.
It's like creating a map of how similar these concepts are to each other.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# Load environment variables
load_dotenv()

# Create OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-ada-002"):
    """Get embedding for a text using OpenAI API"""
    # This function takes text and converts it into a list of numbers (vector)
    # The AI model "understands" the text and represents it as coordinates in space
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

def create_drone_dataset():
    """Create a comprehensive dataset of drone and AI related concepts"""
    # Organizing drone concepts into 4 main categories to see how they relate
    
    # Drone hardware components - the physical parts that make up a drone
    # These should cluster together since they're all mechanical/physical components
    drone_hardware = {
        "propeller": "Mechanical rotating blade component that generates thrust for drone flight",
        "motor": "Mechanical electric motor component that spins the propeller to create lift",
        "battery": "Mechanical power source component that provides energy for drone operation",
        "frame": "Mechanical structural body component that holds all drone parts together",
        "landing_gear": "Mechanical support structure component for safe takeoff and landing",
        "gimbal": "Mechanical stabilized mount component for camera and sensor equipment"
    }
    
    # Sensors and perception - devices that gather information about the environment
    # These should form their own cluster since they all "sense" things
    sensors = {
        "camera": "Electronic sensor device for visual data collection and navigation",
        "lidar": "Electronic laser sensor device for precise distance measurement and mapping",
        "gps": "Electronic satellite navigation sensor device for determining location",
        "imu": "Electronic inertial measurement sensor device for orientation and acceleration",
        "altimeter": "Electronic sensor device that measures altitude above ground level",
        "compass": "Electronic magnetic sensor device for directional orientation"
    }
    
    # AI and autonomy systems - the "brain" of the drone that makes decisions
    # These should be closely related since they all involve AI/software
    ai_systems = {
        "flight_controller": "Software AI system that manages drone stability and movement",
        "path_planner": "Software algorithm that calculates optimal flight routes",
        "obstacle_detection": "Software AI system that identifies and avoids barriers",
        "object_recognition": "Software computer vision system for identifying objects",
        "autonomous_navigation": "Software AI system for self-directed flight without human input",
        "mission_planner": "Software system that defines and executes flight missions"
    }
    
    # Drone types and applications - different kinds of drones for different purposes
    # These should show relationships to their components and use cases
    drone_types = {
        "quadcopter": "Complete drone vehicle with four rotors for vertical takeoff and landing",
        "fixed_wing": "Complete drone vehicle with wings for efficient aircraft-style flight",
        "hexacopter": "Complete drone vehicle with six rotors for increased payload capacity",
        "delivery_drone": "Complete drone vehicle designed for autonomous package delivery",
        "surveillance_drone": "Complete drone vehicle equipped for monitoring and security",
        "agricultural_drone": "Complete drone vehicle specialized for farming and crop monitoring"
    }
    
    # Combine all categories into one big dictionary
    # The ** operator "unpacks" each dictionary and merges them together
    all_texts = {**drone_hardware, **sensors, **ai_systems, **drone_types}
    
    # Create category labels for coloring - this helps me see which group each item belongs to
    categories = {}
    for key in drone_hardware.keys():
        categories[key] = "Hardware"
    for key in sensors.keys():
        categories[key] = "Sensors"
    for key in ai_systems.keys():
        categories[key] = "AI Systems"
    for key in drone_types.keys():
        categories[key] = "Drone Types"
    
    return all_texts, categories

def visualize_embeddings(embeddings, labels, categories, title="Drone Embeddings Visualization"):
    """Create 2D visualization of embeddings using t-SNE"""
    # Try high-dimensional embeddings and make them 2D
    
    # Convert embeddings to numpy array
    embedding_matrix = np.array(list(embeddings.values()))
    
    # Normalize the embeddings to help with clustering
    # This ensures all embeddings are on the same scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    embedding_matrix = scaler.fit_transform(embedding_matrix)
    
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    print(f"Sample embedding values: {embedding_matrix[0][:5]}")  # Show first 5 values
    
    # Apply t-SNE for dimensionality reduction
    # t-SNE is like a magic algorithm that takes our 1536-dimensional vectors and squishes them into 2D
    # It tries to keep similar things close together and different things far apart
    print("Applying t-SNE dimensionality reduction...")
    # Using even smaller perplexity and more iterations for better separation
    # Perplexity controls how many neighbors each point considers - smaller = tighter clusters
    tsne = TSNE(n_components=2, random_state=42, perplexity=3, learning_rate=50, max_iter=2000, early_exaggeration=20)
    embeddings_2d = tsne.fit_transform(embedding_matrix)
    
    # Create color map for categories
    unique_categories = list(set(categories.values()))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
    color_map = dict(zip(unique_categories, colors))
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Plot each point with its category color
    # Each point represents one drone concept, positioned based on its embedding
    for i, (label, category) in enumerate(categories.items()):
        x, y = embeddings_2d[i]  # Get the 2D coordinates for this concept
        color = color_map[category]  # Get the color for this category
        plt.scatter(x, y, c=[color], s=100, alpha=0.7, edgecolors='black', linewidth=1)
        plt.annotate(label.replace('_', '\n'), (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, ha='left', va='bottom')
    
    # Add a crappy simlple legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color_map[cat], markersize=10, label=cat)
                      for cat in unique_categories]
    plt.legend(handles=legend_elements, loc='upper right', title="Categories")
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('drone_embeddings_visualization.png', dpi=300, bbox_inches='tight')
    plt.show() 
    
    return embeddings_2d

def analyze_clusters(embeddings_2d, labels, categories):
    """Analyze the clustering patterns in the visualization"""
    
    print("\n=== CLUSTER ANALYSIS ===")
    
    # Group by categories - organize everything by its category so I can see the patterns
    category_groups = {}
    for label, category in categories.items():
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(label)
    
    # Analyze each category
    for category, items in category_groups.items():
        print(f"\n{category} ({len(items)} items):")
        for item in items:
            print(f"  • {item.replace('_', ' ').title()}")
    
    print(f"\nTotal concepts analyzed: {len(labels)}")
    print("Visualization saved as 'drone_embeddings_visualization.png'")
    

def main():
    print("=== DRONE EMBEDDINGS VISUALIZATION ===\n")
       
    # Create dataset
    texts, categories = create_drone_dataset()
    print(f"Created dataset with {len(texts)} drone and AI concepts")
    
    # Get embeddings
    print("\nGenerating embeddings...")
    embeddings = {}
    for name, description in texts.items():
        print(f"  Processing: {name}")
        embedding = get_embedding(description)  # Convert this text to a vector
        embeddings[name] = embedding
    
    print(f"\nGenerated {len(embeddings)} embeddings with {len(list(embeddings.values())[0])} dimensions each")
    
    # Quick similarity check to verify embeddings are working
    print("\n=== QUICK SIMILARITY CHECK ===")
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Check similarity between similar concepts
    prop_embedding = embeddings['propeller']
    motor_embedding = embeddings['motor']
    camera_embedding = embeddings['camera']
    
    prop_motor_sim = cosine_similarity([prop_embedding], [motor_embedding])[0][0]
    prop_camera_sim = cosine_similarity([prop_embedding], [camera_embedding])[0][0]
    
    print(f"Propeller vs Motor similarity: {prop_motor_sim:.4f}")
    print(f"Propeller vs Camera similarity: {prop_camera_sim:.4f}")
    print(f"Expected: Hardware should be more similar than hardware vs sensor")
    
    # Create visualization
    print("\nCreating visualization...")
    embeddings_2d = visualize_embeddings(embeddings, list(texts.keys()), categories)
    
    # Analyze results
    analyze_clusters(embeddings_2d, list(texts.keys()), categories)
    
    print("\n=== INSIGHTS ===")
    print("• Hardware components should cluster together")
    print("• Sensors should form their own cluster")
    print("• AI systems should be closely related")
    print("• Drone types should show relationships to their components")
    print("• Similar functions should appear near each other")


if __name__ == "__main__":
    main()
