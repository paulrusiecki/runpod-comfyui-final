#!/usr/bin/env python3
"""
RunPod Client for DeepSeek Serverless Function
This script allows you to send requests to your deployed RunPod endpoint
"""

import requests
import json
import sys
import argparse
import os
from typing import Dict, Any

# Replace this with your actual RunPod endpoint URL
RUNPOD_ENDPOINT = "https://your-endpoint-id.runpod.net/run"
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")

def send_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a request to the RunPod endpoint
    """
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add authorization header if API key is provided
        if RUNPOD_API_KEY:
            headers["Authorization"] = f"Bearer {RUNPOD_API_KEY}"
        
        response = requests.post(
            RUNPOD_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=300  # 5 minute timeout for model loading
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        return {"status": "error", "message": str(e)}

def test_comfyui_simple(prompt: str, steps: int = 20, cfg_scale: float = 7.5, width: int = 512, height: int = 512, seed: int = None):
    """
    Test ComfyUI with a simple text-to-image workflow
    """
    payload = {
        "input": {
            "prompt": prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "seed": seed
        }
    }
    
    print(f"🎨 Generating image with ComfyUI - prompt: {prompt}")
    print(f"⚙️  Parameters: {steps} steps, {cfg_scale} CFG, {width}x{height}")
    print("⏳ Generating image...")
    
    result = send_request(payload)
    
    if result.get("status") == "success":
        print(f"✅ Image generated successfully!")
        print(f"📝 Prompt ID: {result.get('prompt_id', 'N/A')}")
        print(f"🖼️  Generated {len(result.get('images', []))} images")
        print(f"🔧 Output nodes: {result.get('outputs', [])}")
        
        # Display image info
        for i, image in enumerate(result.get('images', [])):
            print(f"🖼️  Image {i+1}: {image.get('filename', 'N/A')} (size: {len(image.get('data', ''))} bytes)")
            
    else:
        print(f"❌ Error: {result.get('message', 'Unknown error')}")

def test_comfyui_workflow(workflow_file: str):
    """
    Test ComfyUI with a custom workflow file
    """
    try:
        with open(workflow_file, 'r') as f:
            workflow = json.load(f)
        
        payload = {
            "input": {
                "workflow": workflow
            }
        }
        
        print(f"🎨 Running custom ComfyUI workflow from: {workflow_file}")
        print("⏳ Processing workflow...")
        
        result = send_request(payload)
        
        if result.get("status") == "success":
            print(f"✅ Workflow completed successfully!")
            print(f"📝 Prompt ID: {result.get('prompt_id', 'N/A')}")
            print(f"🖼️  Generated {len(result.get('images', []))} images")
            print(f"🔧 Output nodes: {result.get('outputs', [])}")
            
        else:
            print(f"❌ Error: {result.get('message', 'Unknown error')}")
            
    except FileNotFoundError:
        print(f"❌ Workflow file not found: {workflow_file}")
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in workflow file: {workflow_file}")

def test_text_processing(text: str):
    """
    Test text processing functionality
    """
    payload = {
        "input": {
            "text": text
        }
    }
    
    print(f"📝 Testing text processing: {text}")
    result = send_request(payload)
    
    if result.get("status") == "success":
        print(f"✅ Original: {result.get('original_text', 'N/A')}")
        print(f"✅ Processed: {result.get('processed_text', 'N/A')}")
    else:
        print(f"❌ Error: {result.get('message', 'Unknown error')}")

def test_default_handler():
    """
    Test default handler
    """
    payload = {
        "input": {
            "test_data": "Hello from client!"
        }
    }
    
    print("🔧 Testing default handler...")
    result = send_request(payload)
    
    if result.get("status") == "success":
        print(f"✅ Message: {result.get('message', 'N/A')}")
        print(f"✅ Input received: {result.get('input_received', 'N/A')}")
    else:
        print(f"❌ Error: {result.get('message', 'Unknown error')}")

def main():
    parser = argparse.ArgumentParser(description="RunPod ComfyUI Client")
    parser.add_argument("--endpoint", required=True, help="Your RunPod endpoint URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Simple ComfyUI generation command
    simple_parser = subparsers.add_parser("generate", help="Generate image with simple ComfyUI workflow")
    simple_parser.add_argument("prompt", help="Prompt for image generation")
    simple_parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    simple_parser.add_argument("--cfg-scale", type=float, default=7.5, help="CFG scale")
    simple_parser.add_argument("--width", type=int, default=512, help="Image width")
    simple_parser.add_argument("--height", type=int, default=512, help="Image height")
    simple_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    # Custom workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Run custom ComfyUI workflow")
    workflow_parser.add_argument("workflow_file", help="Path to workflow JSON file")
    
    # Text processing command
    text_parser = subparsers.add_parser("text", help="Test text processing")
    text_parser.add_argument("text", help="Text to process")
    
    # Default handler command
    subparsers.add_parser("default", help="Test default handler")
    
    args = parser.parse_args()
    
    # Set the endpoint URL
    global RUNPOD_ENDPOINT
    RUNPOD_ENDPOINT = args.endpoint
    
    if args.command == "generate":
        test_comfyui_simple(args.prompt, args.steps, args.cfg_scale, args.width, args.height, args.seed)
    elif args.command == "workflow":
        test_comfyui_workflow(args.workflow_file)
    elif args.command == "text":
        test_text_processing(args.text)
    elif args.command == "default":
        test_default_handler()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
