"""
Telegram Image Editor Bot - Railway Job Processor

=============================================================================
ARCHITECTURE DIAGRAM - How the files connect:
=============================================================================

┌──────────────────────────────────────────────────────────────────────────┐
│                           USER FLOW                                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. User sends IMAGE to Telegram bot                                     │
│     └──> Cloudflare Worker (image_worker.js) receives file_id           │
│                                                                          │
│  2. User sends TEXT (e.g., "HUSTLE") to Telegram bot                    │
│     └──> Worker stores job in Redis: {file_id, text_overlay, chat_id}   │
│     └──> Worker wakes up Railway via GraphQL API                        │
│                                                                          │
│  3. THIS FILE (tele_image_editor.py) on Railway:                        │
│     └──> Fetches job from Redis queue                                   │
│     └──> Downloads image from Telegram using file_id                    │
│     └──> Calls main.py's ThumbnailGenerator.create_thumbnail()          │
│          ┌─────────────────────────────────────────────────────────┐    │
│          │  ThumbnailGenerator.create_thumbnail(                   │    │
│          │      image_path="downloads/job123.jpg",  # INPUT        │    │
│          │      text="HUSTLE",                      # INPUT        │    │
│          │      output_path="outputs/out_job123.jpg" # OUTPUT      │    │
│          │  )                                                       │    │
│          │                                                         │    │
│          │  This does ALL the image processing magic:              │    │
│          │  - Smart subject positioning (so text isn't covered)    │    │
│          │  - AI background removal (rembg)                        │    │
│          │  - Blurred + tinted background                         │    │
│          │  - Typography stack with fading text                   │    │
│          │  - Film grain overlay                                   │    │
│          │  - Cinematic camera filter                             │    │
│          │  - Final sharpening                                     │    │
│          └─────────────────────────────────────────────────────────┘    │
│     └──> Submits result (output.jpg) to Cloudflare Worker              │
│     └──> Self-terminates to save resources                             │
│                                                                          │
│  4. Cloudflare Worker sends final image to Telegram chat               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

FILES INVOLVED:
  - image_worker.js     : Cloudflare Worker (Telegram webhook, Redis, Railway wake)
  - tele_image_editor.py: THIS FILE (Railway job processor)
  - main.py             : Core image processing (ThumbnailGenerator class)

=============================================================================
"""

import os
import time
import requests
import json
import logging
import threading
from flask import Flask
from waitress import serve

# =============================================================================
# IMPORT THE IMAGE PROCESSOR FROM main.py
# =============================================================================
# This is the connection! ThumbnailGenerator does all the image magic.
# It's the same code we've been running locally, now running on Railway.
from main import ThumbnailGenerator, TARGET_SIZE

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# --- Constants and Configuration ---
BOT_TOKEN = os.environ.get("BOT_TOKEN")
WORKER_PUBLIC_URL = os.environ.get("WORKER_PUBLIC_URL")
RAILWAY_API_TOKEN = os.environ.get("RAILWAY_API_TOKEN")
RAILWAY_SERVICE_ID = os.environ.get("RAILWAY_SERVICE_ID")
UPSTASH_REDIS_REST_URL = os.environ.get("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN")

# Validate environment variables
required_env = [BOT_TOKEN, WORKER_PUBLIC_URL, RAILWAY_API_TOKEN, RAILWAY_SERVICE_ID, 
                UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN]
if not all(required_env):
    raise ValueError("All required environment variables must be set!")

# --- File Paths ---
DOWNLOAD_PATH = "downloads"
OUTPUT_PATH = "outputs"


# --- Helper Functions ---
def cleanup_files(file_list):
    """Safely delete a list of files."""
    for file_path in file_list:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Cleaned up file: {file_path}")
            except OSError as e:
                logging.error(f"Error deleting file {file_path}: {e}")


def create_directories():
    """Create necessary directories if they don't exist."""
    for path in [DOWNLOAD_PATH, OUTPUT_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)


# --- Railway API Functions ---
def stop_railway_deployment():
    """Stops the Railway deployment using the GraphQL API."""
    logging.info("Attempting to stop Railway deployment...")
    api_token = RAILWAY_API_TOKEN
    service_id = RAILWAY_SERVICE_ID
    
    if not api_token or not service_id:
        logging.warning("RAILWAY_API_TOKEN or RAILWAY_SERVICE_ID is not set. Skipping stop command.")
        return
    
    graphql_url = "https://backboard.railway.app/graphql/v2"
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    
    # Get latest deployment ID
    get_id_query = {
        "query": "query getLatestDeployment($serviceId: String!) { service(id: $serviceId) { deployments(first: 1) { edges { node { id } } } } }",
        "variables": {"serviceId": service_id}
    }
    
    try:
        response = requests.post(graphql_url, json=get_id_query, headers=headers, timeout=15)
        response.raise_for_status()
        deployment_id = response.json()['data']['service']['deployments']['edges'][0]['node']['id']
        logging.info(f"Successfully fetched latest deployment ID for shutdown: {deployment_id}")
    except (requests.exceptions.RequestException, KeyError, IndexError) as e:
        logging.error(f"Failed to get Railway deployment ID for shutdown: {e}")
        return
    
    # Stop deployment
    stop_mutation = {
        "query": "mutation deploymentStop($id: String!) { deploymentStop(id: $id) }",
        "variables": {"id": deployment_id}
    }
    
    try:
        response = requests.post(graphql_url, json=stop_mutation, headers=headers, timeout=15)
        response.raise_for_status()
        logging.info("Successfully sent stop command to Railway. Service will shut down.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send stop command to Railway: {e}")


# --- Redis Queue Functions ---
def fetch_job_from_redis():
    """Fetches a single job from the Upstash Redis queue."""
    url = f"{UPSTASH_REDIS_REST_URL}/rpop/image_job_queue"
    headers = {"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        result = data.get("result")
        
        if result:
            logging.info("Successfully fetched a new job from Redis.")
            return json.loads(result)
        else:
            logging.info("Job queue in Redis is empty.")
            return None
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        logging.error(f"Could not fetch or decode job from Redis: {e}")
        return None


# --- Telegram API Functions ---
def download_telegram_file(file_id, job_id):
    """Downloads a file from Telegram using a file_id."""
    try:
        # Get file path from Telegram
        file_info_url = f"https://api.telegram.org/bot{BOT_TOKEN}/getFile"
        params = {'file_id': file_id}
        response = requests.get(file_info_url, params=params, timeout=15)
        response.raise_for_status()
        file_path = response.json()['result']['file_path']
        
        # Download the file
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
        file_extension = os.path.splitext(file_path)[1] or '.jpg'
        save_path = os.path.join(DOWNLOAD_PATH, f"{job_id}{file_extension}")
        
        with requests.get(file_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        logging.info(f"Successfully downloaded image to {save_path}")
        return save_path
    except Exception as e:
        logging.error(f"Failed to download file_id {file_id}: {e}", exc_info=True)
        return None


# --- Worker Communication ---
def submit_result_to_worker(chat_id, image_path, messages_to_delete):
    """Uploads the final image to the Cloudflare worker."""
    url = f"{WORKER_PUBLIC_URL}/submit-image-result"
    logging.info(f"Submitting result for chat_id {chat_id} to worker...")
    
    try:
        with open(image_path, 'rb') as image_file:
            files = {
                'image': ('final_image.jpg', image_file, 'image/jpeg'),
                'chat_id': (None, str(chat_id)),
                'messages_to_delete': (None, json.dumps(messages_to_delete))
            }
            response = requests.post(url, files=files, timeout=60)
            response.raise_for_status()
        
        logging.info("Successfully submitted result to worker.")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Error uploading result to worker: {e}")
        return False


# --- Core Processing Logic ---
def process_image_job(job_data):
    """
    The main image processing logic for a single job.
    
    DATA FLOW:
    1. Telegram user sends IMAGE → Worker stores file_id in Redis
    2. Telegram user sends TEXT → Worker stores text_overlay in Redis
    3. This function fetches job from Redis containing:
       - job_data['file_id']      → Telegram file ID for the image
       - job_data['text_overlay'] → The text user wants on the cover (e.g., "HUSTLE")
       - job_data['chat_id']      → Where to send the result
    4. Download the image from Telegram using file_id
    5. Call main.py's ThumbnailGenerator.create_thumbnail():
       - INPUT:  image_path (downloaded file)
       - INPUT:  text_overlay (user's text)
       - OUTPUT: output_path (processed viral cover)
    6. Upload result to Cloudflare Worker → Worker sends to Telegram user
    """
    
    # Extract job data from Redis queue
    chat_id = job_data['chat_id']
    job_id = job_data['job_id']
    text_overlay = job_data.get('text_overlay', 'HUSTLE')  # Default if not provided
    messages_to_delete = job_data.get('messages_to_delete', [])
    
    logging.info("=" * 60)
    logging.info(f"JOB STARTED: {job_id}")
    logging.info(f"  Chat ID: {chat_id}")
    logging.info(f"  Text Overlay: '{text_overlay}'")
    logging.info(f"  File ID: {job_data.get('file_id', 'N/A')[:50]}...")
    logging.info("=" * 60)
    
    files_to_clean = []
    
    try:
        # ============================================================
        # STEP 1: Download image from Telegram using file_id
        # ============================================================
        # The file_id was captured when user sent the image to Telegram
        # We use Telegram's getFile API to download it to our server
        logging.info("[STEP 1/3] Downloading image from Telegram...")
        
        image_path = download_telegram_file(job_data['file_id'], job_id)
        if not image_path:
            raise ValueError("Image download failed - check file_id and BOT_TOKEN")
        files_to_clean.append(image_path)
        
        logging.info(f"  ✓ Downloaded to: {image_path}")
        
        # ============================================================
        # STEP 2: Process image using main.py's ThumbnailGenerator
        # ============================================================
        # This is where the MAGIC happens!
        # ThumbnailGenerator.create_thumbnail() does:
        #   - Smart subject positioning (so text isn't covered)
        #   - AI background removal (rembg)
        #   - Blurred + tinted background
        #   - Typography stack (repeated fading text)
        #   - Film grain
        #   - Camera filter (cinematic color grade)
        #   - Sharpening
        logging.info("[STEP 2/3] Processing image with ThumbnailGenerator...")
        logging.info(f"  Input:  {image_path}")
        logging.info(f"  Text:   '{text_overlay}'")
        
        output_path = os.path.join(OUTPUT_PATH, f"output_{job_id}.jpg")
        
        # THIS IS THE CONNECTION TO main.py!
        # ThumbnailGenerator is imported from main.py at the top of this file
        generator = ThumbnailGenerator()
        generator.create_thumbnail(
            image_path=image_path,      # Input: downloaded image file
            text=text_overlay,          # Input: user's text from Telegram
            output_path=output_path     # Output: where to save the result
        )
        
        if not os.path.exists(output_path):
            raise ValueError("Image processing failed - ThumbnailGenerator did not create output")
        files_to_clean.append(output_path)
        
        logging.info(f"  ✓ Output saved to: {output_path}")
        
        # ============================================================
        # STEP 3: Submit result to Cloudflare Worker
        # ============================================================
        # Worker will then send the image to Telegram chat
        logging.info("[STEP 3/3] Submitting result to Cloudflare Worker...")
        
        success = submit_result_to_worker(chat_id, output_path, messages_to_delete)
        
        if success:
            logging.info("  ✓ Result submitted successfully!")
        else:
            logging.error("  ✗ Failed to submit result to worker")
        
    except Exception as e:
        logging.error(f"JOB FAILED: {job_id}")
        logging.error(f"  Error: {str(e)[-500:]}", exc_info=True)
    finally:
        logging.info(f"Cleaning up {len(files_to_clean)} files for job {job_id}...")
        cleanup_files(files_to_clean)
        logging.info("=" * 60)


# --- Keep-Alive Web Server ---
app = Flask(__name__)

@app.route('/')
@app.route('/health')
def keep_alive():
    """Endpoint for Railway's health check."""
    return "Image Editor Bot is awake and healthy.", 200


def run_web_server():
    """Runs the Flask app on the port provided by Railway."""
    port = int(os.environ.get("PORT", 8080))
    serve(app, host='0.0.0.0', port=port)


# --- Main Entry Point ---
if __name__ == '__main__':
    # Step 1: Start the web server in a background thread
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    logging.info("Keep-alive web server started in a background thread.")
    
    # Step 2: Initialize and check for a job
    logging.info("Starting Image Editor Job Processor...")
    create_directories()
    initial_job = fetch_job_from_redis()
    
    # Step 3: Decide what to do
    if initial_job:
        # --- PATH A: REAL JOB ---
        logging.info("Hot Start: Job found immediately. Starting processing.")
        process_image_job(initial_job)
        
        # Process any remaining jobs in queue
        while True:
            job = fetch_job_from_redis()
            if job:
                process_image_job(job)
            else:
                logging.info("Job queue is empty.")
                break
        
        # When all jobs are done, shutdown
        logging.info("All tasks complete. Requesting shutdown in 70 seconds...")
        time.sleep(70)
        stop_railway_deployment()
    
    else:
        # --- PATH B: PING (Cold Start) ---
        logging.warning("Cold Start: No initial job found. Staying alive for 70 seconds for pinger.")
        time.sleep(70)
        
        # Shutdown after ping handled
        logging.info("Ping handled successfully. Requesting shutdown.")
        stop_railway_deployment()
    
    logging.info("Image Editor Processor has finished and is exiting.")
