#!/usr/bin/env python3
"""
Script to create a JSONL dataset from the SROIE dataset structure.
Converts the entities directory into a structured JSONL format.
"""

import json
import os
from pathlib import Path


def parse_entities_file(entities_path: str) -> dict:
    """Parse an entities file and return a dictionary."""
    with open(entities_path, 'r') as f:
        return json.load(f)


def create_dataset_jsonl(input_dir: str, output_file: str):
    """Create a JSONL dataset from the SROIE dataset structure."""
    entities_dir = os.path.join(input_dir, 'entities')
    img_dir = os.path.join(input_dir, 'img')

    # Get all image filenames
    image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    with open(output_file, 'w') as outfile:
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            
            # Construct paths
            entities_path = os.path.join(entities_dir, f"{base_name}.txt")
            img_path = os.path.join(img_dir, img_file)
            
            # Parse files
            entities = parse_entities_file(entities_path)
            
            # Create dataset entry
            entry = {
                "input": {
                    "image_path": img_path
                },
                "result": entities,
                "id": base_name
            }
            
            # Write to JSONL
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    input_dir = "train"
    output_file = "train.jsonl"
    
    print(f"Creating dataset from {input_dir}...")
    create_dataset_jsonl(input_dir, output_file)
    print(f"Dataset created: {output_file}")
    print(f"Total entries: {len([f for f in os.listdir(os.path.join(input_dir, 'img')) if f.endswith('.jpg')])}")
