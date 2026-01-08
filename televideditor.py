import os
import time
import requests
import json
import logging
import base64
import subprocess
import threading
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import textwrap
# --- New imports for the keep-alive web server ---
from flask import Flask
from waitress import serve

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# --- Constants and Configuration ---
BOT_TOKEN_2 = os.environ.get("BOT_TOKEN")
WORKER_PUBLIC_URL = os.environ.get("WORKER_PUBLIC_URL")
RAILWAY_API_TOKEN = os.environ.get("RAILWAY_API_TOKEN")
RAILWAY_SERVICE_ID = os.environ.get("RAILWAY_SERVICE_ID")
UPSTASH_REDIS_REST_URL = os.environ.get("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN")

if not all([BOT_TOKEN_2, WORKER_PUBLIC_URL, RAILWAY_API_TOKEN, RAILWAY_SERVICE_ID, UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN]):
    raise ValueError("All required environment variables must be set!")


# --- Video Processing Constants (Unchanged) ---
COMP_WIDTH = 1080
COMP_HEIGHT = 1920
COMP_SIZE_STR = f"{COMP_WIDTH}x{COMP_HEIGHT}"
BACKGROUND_COLOR = "black"
FPS = 30
IMAGE_DURATION = 7
MEDIA_FADE_DURATION = 10
CAPTION_FADE_DURATION = 4
MEDIA_Y_OFFSET = 0
CAPTION_V_PADDING = 37
CAPTION_FONT_SIZE = 40
CAPTION_TOP_PADDING_LINES = 0
CAPTION_LINE_SPACING = 5
CAPTION_FONT = "Poppins-Bold"
CAPTION_TEXT_COLOR = (255, 255, 255)
CAPTION_BG_COLOR = (255, 255, 255)
SHADOW_COLOR = (0, 0, 0)
SHADOW_OFFSET = (0, 0)
SHADOW_BLUR_RADIUS = 20

# --- File Paths (Unchanged) ---
DOWNLOAD_PATH = "downloads"
OUTPUT_PATH = "outputs"

# --- Helper Functions (Unchanged) ---
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

# --- Railway API Functions (Unchanged) ---
def stop_railway_deployment():
    """Stops the Railway deployment using the GraphQL API."""
    logging.info("Attempting to stop Railway deployment...")
    api_token = os.environ.get("RAILWAY_API_TOKEN")
    service_id = os.environ.get("RAILWAY_SERVICE_ID")
    if not api_token or not service_id:
        logging.warning("RAILWAY_API_TOKEN or RAILWAY_SERVICE_ID is not set. Skipping stop command.")
        return
    graphql_url = "https://backboard.railway.app/graphql/v2"
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
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
    stop_mutation = {"query": "mutation deploymentStop($id: String!) { deploymentStop(id: $id) }", "variables": {"id": deployment_id}}
    try:
        response = requests.post(graphql_url, json=stop_mutation, headers=headers, timeout=15)
        response.raise_for_status()
        logging.info("Successfully sent stop command to Railway. Service will shut down.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send stop command to Railway: {e}")

# --- Worker Communication Functions (Unchanged) ---
def fetch_job_from_redis():
    """Fetches a single job from the Upstash Redis queue endpoint."""
    url = f"{UPSTASH_REDIS_REST_URL}/rpop/job_queue"
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

def submit_result_to_worker(chat_id, video_path, frame_path, messages_to_delete):
    """Uploads the final video, a frame, and the list of message IDs to the worker."""
    url = f"{WORKER_PUBLIC_URL}/submit-result"
    logging.info(f"Submitting result for chat_id {chat_id} to worker...")
    try:
        with open(frame_path, "rb") as image_file, open(video_path, 'rb') as video_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            files = {
                'video': ('final_video.mp4', video_file, 'video/mp4'),
                'image_data': (None, image_data),
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

# --- Core Processing Logic (Unchanged) ---
def download_telegram_file(file_id, job_id):
    """Downloads a file from Telegram using a file_id."""
    try:
        file_info_url = f"https://api.telegram.org/bot{BOT_TOKEN_2}/getFile"
        params = {'file_id': file_id}
        response = requests.get(file_info_url, params=params, timeout=15)
        response.raise_for_status()
        file_path = response.json()['result']['file_path']
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN_2}/{file_path}"
        file_extension = os.path.splitext(file_path)[1]
        save_path = os.path.join(DOWNLOAD_PATH, f"{job_id}{file_extension}")
        with requests.get(file_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        logging.info(f"Successfully downloaded media to {save_path}")
        return save_path
    except Exception as e:
        logging.error(f"Failed to download file_id {file_id}: {e}", exc_info=True)
        return None

def get_media_dimensions(media_path, media_type):
    """Gets dimensions and duration for media."""
    if media_type == 'image':
        with Image.open(media_path) as img: return img.width, img.height, IMAGE_DURATION
    else: # video
        command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,duration', '-of', 'json', media_path]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
            data = json.loads(result.stdout)['streams'][0]
            return data['width'], data['height'], float(data['duration'])
        except Exception as e:
            logging.error(f"FFprobe failed: {e}")
            return None, None, None

def create_caption_image(text, job_id):
    """Creates a caption image with text and drop shadow."""
    padded_text = ("\n" * CAPTION_TOP_PADDING_LINES) + text
    font = ImageFont.truetype(f"{CAPTION_FONT}.ttf", CAPTION_FONT_SIZE)
    final_lines = [item for line in padded_text.split('\n') for item in textwrap.wrap(line, width=35, break_long_words=True) or ['']]
    wrapped_text = "\n".join(final_lines)
    dummy_draw = ImageDraw.Draw(Image.new('RGB', (0,0)))
    text_bbox = dummy_draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center", spacing=CAPTION_LINE_SPACING, stroke_width=1)
    text_width, text_height = int(text_bbox[2] - text_bbox[0]), int(text_bbox[3] - text_bbox[1])
    img_padding = SHADOW_BLUR_RADIUS * 4
    img_width, img_height = text_width + img_padding, text_height + img_padding
    shadow_img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_img)
    shadow_pos = (img_padding / 2 + SHADOW_OFFSET[0], img_padding / 2 + SHADOW_OFFSET[1])
    shadow_draw.multiline_text(shadow_pos, wrapped_text, font=font, fill=SHADOW_COLOR, anchor="la", align="center", spacing=CAPTION_LINE_SPACING, stroke_width=2, stroke_fill=SHADOW_COLOR)
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=SHADOW_BLUR_RADIUS))
    final_draw = ImageDraw.Draw(shadow_img)
    text_pos = (img_padding / 2, img_padding / 2)
    final_draw.multiline_text(text_pos, wrapped_text, font=font, fill=CAPTION_TEXT_COLOR, anchor="la", align="center", spacing=CAPTION_LINE_SPACING, stroke_width=2, stroke_fill=(0, 0, 0))
    caption_image_path = os.path.join(OUTPUT_PATH, f"caption_{job_id}.png")
    shadow_img.save(caption_image_path)
    return caption_image_path, text_height

def extract_frame_from_video(video_path, duration, job_id):
    """Extracts a frame from the video."""
    frame_path = os.path.join(OUTPUT_PATH, f"frame_{job_id}.jpg")
    midpoint = duration / 2
    command = ['ffmpeg', '-y', '-i', video_path, '-ss', str(midpoint), '-vframes', '1', frame_path]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        logging.info(f"Successfully extracted frame for job {job_id} to {frame_path}")
        return frame_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logging.error(f"FFmpeg frame extraction failed: {getattr(e, 'stderr', e)}")
        return None

def process_video_job(job_data):
    """The main video creation logic for a single job."""
    chat_id, job_id = job_data['chat_id'], job_data['job_id']
    messages_to_delete = job_data.get("messages_to_delete", [])
    logging.info(f"Starting processing for job_id: {job_id}")
    files_to_clean = []
    try:
        media_path = download_telegram_file(job_data['file_id'], job_id)
        if not media_path: raise ValueError("Media download failed.")
        files_to_clean.append(media_path)
        media_type = job_data['media_type']
        media_w, media_h, final_duration = get_media_dimensions(media_path, media_type)
        if not all([media_w, media_h, final_duration]): raise ValueError("Could not get media dimensions.")
        caption_image_path, _ = create_caption_image(job_data['caption_text'], job_id)
        files_to_clean.append(caption_image_path)
        output_filepath = os.path.join(OUTPUT_PATH, f"output_{job_id}.mp4")
        scaled_media_h = int(media_h * (COMP_WIDTH / media_w))
        media_y_pos = int((COMP_HEIGHT / 2 - scaled_media_h / 2) + MEDIA_Y_OFFSET)
        command = ['ffmpeg', '-y', '-f', 'lavfi', '-i', f'color=c={BACKGROUND_COLOR}:s={COMP_SIZE_STR}:d={final_duration}']
        if media_type == 'image': command.extend(['-loop', '1', '-t', str(final_duration)])
        command.extend(['-i', media_path, '-loop', '1', '-i', caption_image_path])
        filter_parts = []
        media_fade_filter = f",fade=t=in:st=0:d={MEDIA_FADE_DURATION}" if job_data['apply_fade'] else ""
        filter_parts.append(f"[1:v]scale={COMP_WIDTH}:-1,setpts=PTS-STARTPTS{media_fade_filter}[scaled_media]")
        caption_fade_filter = f",fade=t=in:st=0:d={CAPTION_FADE_DURATION}" if job_data['apply_fade'] else ""
        filter_parts.append(f"[2:v]format=rgba,trim=duration={final_duration}{caption_fade_filter}[faded_caption]")
        filter_parts.extend([f"[0:v][scaled_media]overlay=(W-w)/2:{media_y_pos}[base_scene]", f"[base_scene][faded_caption]overlay=(W-w)/2:(H-h)/2[final_v]"])
        filter_complex = ";".join(filter_parts)
        map_args = ['-map', '[final_v]']
        if media_type == 'video':
            filter_complex += ";[1:a]asetpts=PTS-STARTPTS[final_a]"
            map_args.extend(['-map', '[final_a]'])
        command.extend(['-filter_complex', filter_complex, *map_args, '-ss', '0.4', '-c:v', 'libx264', '-preset', 'superfast', '-tune', 'zerolatency', '-c:a', 'aac', '-b:a', '192k', '-r', str(FPS), '-pix_fmt', 'yuv420p', output_filepath])
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logging.error(f"FFMPEG STDERR: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, command, stderr=result.stderr)
        logging.info(f"FFmpeg processing finished for job {job_id}.")
        files_to_clean.append(output_filepath)
        trimmed_duration = final_duration - 0.4
        frame_path = extract_frame_from_video(output_filepath, trimmed_duration, job_id)
        if not frame_path: raise ValueError("Frame extraction failed.")
        files_to_clean.append(frame_path)
        submit_result_to_worker(chat_id, output_filepath, frame_path, messages_to_delete)
    except Exception as e:
        logging.error(f"Failed to process job {job_id}: {str(e)[-1000:]}", exc_info=True)
    finally:
        logging.info(f"Cleaning up files for job {job_id}.")
        cleanup_files(files_to_clean)


# --- NEW: Keep-Alive Web Server ("The Receptionist") ---
app = Flask(__name__)
@app.route('/')
@app.route('/health') # Added a dedicated health endpoint for best practice
def keep_alive():
    """Endpoint for both the pinger and Railway's health check."""
    return "Bot is awake and healthy.", 200

def run_web_server():
    """Runs the Flask app on the port provided by Railway."""
    port = int(os.environ.get("PORT", 8080))
    serve(app, host='0.0.0.0', port=port)

# --- FINAL VERSION: Main Bot Logic ---
if __name__ == '__main__':
    # Step 1: Start the web server in a background thread.
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    logging.info("Keep-alive web server started in a background thread.")

    # Step 2: Initialize and check for a job.
    logging.info("Starting Python Job Processor and checking for an initial job...")
    create_directories()
    initial_job = fetch_job_from_redis()

    # Step 3: Decide what to do.
    if initial_job:
        # --- PATH A: REAL JOB ---
        # The web server will continue running in the background while this processes.
        logging.info("Hot Start: Job found immediately. Starting processing.")
        process_video_job(initial_job)
        
        while True:
            job = fetch_job_from_redis()
            if job:
                process_video_job(job)
            else:
                logging.info("Job queue is empty.")
                break 
        
        # When all jobs are done, we proceed to shutdown.
        logging.info("All tasks complete. Requesting shutdown.")
        time.sleep(70)
        stop_railway_deployment()

    else:
        # --- PATH B: PING ---
        # We were woken by a ping. Our only job is to stay alive long enough
        # for the web thread to answer the ping successfully.
        # A simple sleep is the cleanest way to do this.
        logging.warning("Cold Start: No initial job found. Staying alive for 70 seconds for pinger.")
        time.sleep(70)
        
        # Now that the ping has been answered, shut down.
        logging.info("Ping handled successfully. Requesting shutdown.")
        stop_railway_deployment()

    logging.info("Processor has finished its work and is exiting.")
