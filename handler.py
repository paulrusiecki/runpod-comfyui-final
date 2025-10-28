import runpod
import json
import base64
import io
import os
import logging
import requests
import time
import uuid
from typing import Dict, Any, Optional
from PIL import Image
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ComfyUI configuration
COMFYUI_SERVER_URL = os.getenv("COMFYUI_SERVER_URL", "http://127.0.0.1:8188")

def validate_input(input_data: Dict[str, Any]) -> Optional[str]:
    """Validate input data and return error message if invalid"""
    if not isinstance(input_data, dict):
        return "Input must be a dictionary"
    
    # Validate ComfyUI workflow input
    if 'workflow' in input_data:
        workflow = input_data['workflow']
        if not isinstance(workflow, dict):
            return "Workflow must be a dictionary"
        
        # Basic workflow validation
        if 'prompt' not in workflow:
            return "Workflow must contain 'prompt' section"
        
    # Validate simple text-to-image input
    elif 'prompt' in input_data:
        prompt = input_data['prompt']
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            return "Prompt must be a non-empty string"
        if len(prompt) > 1000:
            return "Prompt too long (max 1000 characters)"
        
        # Validate generation parameters
        steps = input_data.get('steps', 20)
        if not isinstance(steps, int) or steps < 1 or steps > 100:
            return "steps must be between 1 and 100"
        
        cfg_scale = input_data.get('cfg_scale', 7.5)
        if not isinstance(cfg_scale, (int, float)) or cfg_scale < 1 or cfg_scale > 20:
            return "cfg_scale must be between 1 and 20"
        
        width = input_data.get('width', 512)
        height = input_data.get('height', 512)
        if not isinstance(width, int) or width < 64 or width > 2048:
            return "width must be between 64 and 2048"
        if not isinstance(height, int) or height < 64 or height > 2048:
            return "height must be between 64 and 2048"
    
    # Validate image input for img2img
    if 'init_image' in input_data:
        image_data = input_data['init_image']
        if not isinstance(image_data, str):
            return "Image data must be base64 encoded string"
        try:
            if len(image_data) > 20 * 1024 * 1024:  # 20MB limit
                return "Image too large (max 20MB)"
            base64.b64decode(image_data[:100])  # Test decode first 100 chars
        except Exception:
            return "Invalid base64 image data"
    
    return None

def queue_prompt(workflow: Dict[str, Any]) -> str:
    """Queue a prompt to ComfyUI and return the prompt ID"""
    try:
        response = requests.post(
            f"{COMFYUI_SERVER_URL}/prompt",
            json={"prompt": workflow},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result['prompt_id']
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to queue prompt: {e}")
        raise

def get_prompt_status(prompt_id: str) -> Dict[str, Any]:
    """Get the status of a queued prompt"""
    try:
        response = requests.get(
            f"{COMFYUI_SERVER_URL}/history/{prompt_id}",
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get prompt status: {e}")
        raise

def wait_for_completion(prompt_id: str, timeout: int = 300) -> Dict[str, Any]:
    """Wait for prompt completion with timeout"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            status = get_prompt_status(prompt_id)
            if prompt_id in status:
                # Prompt completed
                return status[prompt_id]
            
            # Check if prompt is still running
            response = requests.get(f"{COMFYUI_SERVER_URL}/queue", timeout=30)
            queue_data = response.json()
            
            # Check if our prompt is still in the queue
            prompt_in_queue = any(
                item[1] == prompt_id 
                for item in queue_data.get('queue_running', []) + queue_data.get('queue_pending', [])
            )
            
            if not prompt_in_queue and prompt_id not in status:
                raise Exception("Prompt not found in queue or history")
                
            time.sleep(1)  # Wait before checking again
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error checking prompt status: {e}")
            time.sleep(2)
    
    raise TimeoutError(f"Prompt {prompt_id} timed out after {timeout} seconds")

def create_simple_workflow(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a simple text-to-image workflow for ComfyUI"""
    prompt = input_data['prompt']
    negative_prompt = input_data.get('negative_prompt', '')
    steps = input_data.get('steps', 20)
    cfg_scale = input_data.get('cfg_scale', 7.5)
    width = input_data.get('width', 512)
    height = input_data.get('height', 512)
    sampler_name = input_data.get('sampler_name', 'euler')
    scheduler = input_data.get('scheduler', 'normal')
    
    # This is a simplified workflow structure
    # In practice, you'd want to use a proper ComfyUI workflow JSON
    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": input_data.get('seed', int.from_bytes(os.urandom(4), 'big')),
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": input_data.get('model_name', 'v1-5-pruned-emaonly.ckpt')
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["4", 1]
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["4", 1]
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": ["8", 0]
            }
        }
    }
    
    return workflow

def extract_images_from_output(outputs: Dict[str, Any]) -> list:
    """Extract images from ComfyUI output"""
    images = []
    
    for node_id, node_output in outputs.items():
        if 'images' in node_output:
            for image_data in node_output['images']:
                # Download and process the image
                try:
                    image_url = f"{COMFYUI_SERVER_URL}/view?filename={image_data['filename']}&subfolder={image_data.get('subfolder', '')}&type={image_data.get('type', 'output')}"
                    response = requests.get(image_url, timeout=30)
                    response.raise_for_status()
                    
                    # Convert to base64
                    image_b64 = base64.b64encode(response.content).decode()
                    images.append({
                        "filename": image_data['filename'],
                        "subfolder": image_data.get('subfolder', ''),
                        "type": image_data.get('type', 'output'),
                        "data": image_b64
                    })
                except Exception as e:
                    logger.error(f"Failed to extract image: {e}")
    
    return images

def handler(event):
    """
    Main handler function for RunPod serverless endpoint with ComfyUI integration.
    """
    request_id = event.get('id', 'unknown')
    logger.info(f"Processing request {request_id}")
    
    try:
        # Parse input data
        input_data = event.get('input', {})
        
        # Log request details
        logger.info(f"Request {request_id} - Input keys: {list(input_data.keys())}")
        
        # Validate input
        validation_error = validate_input(input_data)
        if validation_error:
            logger.warning(f"Request {request_id} - Input validation failed: {validation_error}")
            return {
                "status": "error",
                "message": f"Input validation failed: {validation_error}",
                "request_id": request_id
            }
        
        # Check if ComfyUI server is accessible
        try:
            response = requests.get(f"{COMFYUI_SERVER_URL}/system_stats", timeout=10)
            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": "ComfyUI server is not accessible",
                    "request_id": request_id
                }
        except requests.exceptions.RequestException:
            return {
                "status": "error",
                "message": "Cannot connect to ComfyUI server",
                "request_id": request_id
            }
        
        # Process workflow or create simple workflow
        if 'workflow' in input_data:
            workflow = input_data['workflow']
            logger.info(f"Request {request_id} - Using custom workflow")
        else:
            workflow = create_simple_workflow(input_data)
            logger.info(f"Request {request_id} - Created simple workflow for prompt: {input_data['prompt'][:50]}...")
        
        # Queue the prompt
        prompt_id = queue_prompt(workflow)
        logger.info(f"Request {request_id} - Queued prompt with ID: {prompt_id}")
        
        # Wait for completion
        logger.info(f"Request {request_id} - Waiting for completion...")
        result = wait_for_completion(prompt_id, timeout=300)
        
        # Extract images from output
        outputs = result.get('outputs', {})
        images = extract_images_from_output(outputs)
        
        logger.info(f"Request {request_id} - Completed successfully - generated {len(images)} images")
        
        return {
            "status": "success",
            "prompt_id": prompt_id,
            "images": images,
            "outputs": list(outputs.keys()),
            "request_id": request_id
        }
            
    except TimeoutError as e:
        logger.error(f"Request {request_id} - Timeout: {str(e)}")
        return {
            "status": "error",
            "message": f"Generation timed out: {str(e)}",
            "request_id": request_id
        }
    except Exception as e:
        logger.error(f"Request {request_id} - Error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Internal server error: {str(e)}",
            "request_id": request_id
        }

# Start the RunPod serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
