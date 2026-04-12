from flask import (
    Flask,
    jsonify,
    Response,
    make_response,
    send_file,
    send_from_directory,
    request,
)
import json
import time
import pickle
import numpy as np
from pathlib import Path
import subprocess
import signal
import threading
try:
    import mido
    MIDO_AVAILABLE = True
    print("mido library loaded successfully")
except ImportError as e:
    MIDO_AVAILABLE = False
    print(f"Warning: mido not available. MIDI note API will not work. Error: {e}")

# File access locks for pickle files (per piece_id)
_fingering_locks = {}
_fingering_locks_lock = threading.Lock()

def get_fingering_lock(piece_id):
    """Get or create a lock for a specific piece's fingering file."""
    with _fingering_locks_lock:
        if piece_id not in _fingering_locks:
            _fingering_locks[piece_id] = threading.Lock()
        return _fingering_locks[piece_id]


def kill_process_on_port(port):
    """Kill the process using the specified port."""
    try:
        # Use lsof command to find PID using the port on Linux/Mac
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"Killing process using port {port} (PID: {pid})...")
                    try:
                        subprocess.run(["kill", "-9", pid], check=True)
                        print(f"Process {pid} terminated.")
                    except subprocess.CalledProcessError:
                        print(f"Failed to terminate process {pid}.")
            # Wait for process termination
            import time
            time.sleep(0.5)
            return True
        else:
            print(f"No process using port {port}.")
            return False
    except FileNotFoundError:
        # Try fuser if lsof is not available
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], check=True)
            import time
            time.sleep(0.5)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Cannot find or kill process using port {port}.")
            return False


def get_release_root_path():
    # server.py is in web/ folder, so return the parent folder
    return Path(__file__).resolve().parent.parent  # project root (PiAnnotate)

print("Loading data...")

# web/ folder path (where server.py is located)
web_dir = Path(__file__).resolve().parent
assets_dir = web_dir / "assets"
piece_id = 1
data_dir = get_release_root_path() / "for_elise" / "dataset" / f"{piece_id:03d}"

# Try multiple possible locations for mano_faces.pkl
possible_paths = [
    Path(__file__).parent / "static" / "resources" / "mano_faces.pkl",  # Generated file
    assets_dir / "mano_faces.pkl",  # web/assets/mano_faces.pkl
    get_release_root_path() / "fingering" / "assets" / "mano_faces.pkl",  # Fallback
]

faces = None
for path in possible_paths:
    if path.exists():
        try:
            # Read the file content first
            with open(path, "rb") as f:
                file_content = f.read()
            
            # Check if it's a Git LFS pointer file (text file starting with "version")
            if file_content.startswith(b"version https://git-lfs"):
                print(f"Warning: {path} is a Git LFS pointer file, not actual data. Skipping...")
                continue
            
            # Try to load as pickle from the content
            import io
            faces = pickle.load(io.BytesIO(file_content))
            print(f"Successfully loaded mano_faces.pkl from {path}")
            break
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

if faces is None:
    raise FileNotFoundError(
        f"Could not find valid mano_faces.pkl file. "
        f"Tried: {[str(p) for p in possible_paths]}. "
        f"Please ensure the file is downloaded from Git LFS or exists in one of these locations."
    )

print("Data loaded.")
app = Flask(__name__)

# Disable gzip compression to avoid Content-Length mismatch issues
# This prevents ERR_CONTENT_LENGTH_MISMATCH and ERR_INVALID_CHUNKED_ENCODING errors
app.config['COMPRESS_RESPONSE'] = False

# After request hook to handle compression and CORS
@app.after_request
def after_request_handler(response):
    # Remove Content-Encoding header if present (gzip/deflate)
    if 'Content-Encoding' in response.headers:
        del response.headers['Content-Encoding']
    
    # Add CORS headers for all responses (including OPTIONS preflight)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    
    return response


@app.route("/")
def index():
    # return the index.html in the current directory
    return send_file(Path(__file__).parent / "static" / "index.html")


@app.route("/resources/<path:filename>")
def resources(filename):
    # Try multiple possible locations for resources
    # visualizer/static/resources might have Git LFS pointer files
    possible_paths = [
        Path(__file__).parent / "static" / "resources" / filename,
        get_release_root_path() / "fingering" / "static" / "resources" / filename,
    ]
    
    for path in possible_paths:
        if path.exists():
            # Check if it's a Git LFS pointer file (for .hdr files)
            if filename.endswith('.hdr'):
                try:
                    with open(path, "rb") as f:
                        first_bytes = f.read(20)
                        if first_bytes.startswith(b"version https://git-lfs"):
                            print(f"Warning: {path} is a Git LFS pointer file, trying alternative...")
                            continue
                except:
                    pass
            
            return send_file(path)
    
    return jsonify({"error": f"Resource not found: {filename}"}), 404


@app.route("/js/<path:filename>")
def js(filename):
    # return the index.html in the current directory
    return send_file(Path(__file__).parent / "static" / "js" / filename)


@app.route("/css/<path:filename>")
def css(filename):
    # return the index.html in the current directory
    return send_file(Path(__file__).parent / "static" / "css" / filename)


@app.route("/vis")
def vis():
    return send_file(Path(__file__).parent / "static" / "vis.html")


@app.route("/pieces")
def pieces():
    # return the index.html in the current directory
    return send_file(Path(__file__).parent / "pieces.html")


@app.route("/pieces_metadata")
def pieces_metadata():
    # Try multiple possible locations for metadata.json
    possible_paths = [
        get_release_root_path() / "for_elise" / "merge" / "metadata.json",
        get_release_root_path() / "merge" / "metadata.json",
        get_release_root_path() / "metadata.json",
    ]
    
    # split.json path
    split_json_path = get_release_root_path() / "data" / "metadata" / "annotation" / "split.json"
    
    metadata = None
    for metadata_file in possible_paths:
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            break
    
    if metadata is None:
        return jsonify({
            "error": "Metadata file not found",
            "searched_paths": [str(p) for p in possible_paths]
        }), 404
    
    # Load split info from split.json and merge
    if split_json_path.exists():
        with open(split_json_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        
        piece_to_split = split_data.get("piece_to_split", {})
        piece_to_annotator = split_data.get("piece_to_annotator", {})
        piece_metadata_extra = split_data.get("piece_metadata", {})
        
        # Add split info to each piece
        for piece in metadata:
            piece_id_str = str(piece.get("piece_id"))
            # Add split info (train/valid/test)
            piece["split"] = piece_to_split.get(piece_id_str, "unknown")
            # Add annotator info
            piece["annotator"] = piece_to_annotator.get(piece_id_str)
            # Merge extra metadata (period, difficulty, etc.)
            if piece_id_str in piece_metadata_extra:
                extra = piece_metadata_extra[piece_id_str]
                piece["period"] = extra.get("period", "")
                piece["difficulty"] = extra.get("difficulty", 0)
            # Add dataset info (data loaded from for_elise folder)
            piece["dataset"] = "FurElise"
    
    return jsonify(metadata)


def get_data_path(piece_id):
    return get_release_root_path() / "for_elise" / "dataset" / f"{piece_id:03d}"


def get_fingering_path(piece_id):
    """Get path to fingering data file (./data/fingering/{id:03d}.pkl)"""
    return get_release_root_path() / "data" / "fingering" / f"{piece_id:03d}.pkl"




def get_fingering_edited_path(piece_id):
    """Get path to edited fingering data file (./data/fingering_edited/{id:03d}.pkl)"""
    return get_release_root_path() / "data" / "fingering_edited" / f"{piece_id:03d}.pkl"


def get_fingering_edited_ai_path(piece_id):
    """Get path to AI-edited fingering data file (./data/fingering_edited_ai/{id:03d}.pkl) - r0"""
    return get_release_root_path() / "data" / "fingering_edited_ai" / f"{piece_id:03d}.pkl"


@app.route("/audio/<int:piece_id>")
def get_audio(piece_id):
    audio_path = get_data_path(piece_id) / "audio.mp3"
    return send_file(audio_path)


@app.route("/metadata/<int:piece_id>")
def get_piece_metadata(piece_id):
    return send_file(
        get_data_path(piece_id) / "vis" / "metadata.json", mimetype="application/json"
    )


@app.route("/mano_faces_data")
def mano_faces_data():
    d = {
        "left_faces": faces["left_faces"].tolist(),
        "right_faces": faces["right_faces"].tolist(),
    }
    return jsonify(d)


# Cache for motion data to avoid repeated pickle loads
_motion_cache = {}

def get_motion_data(piece_id):
    """Get motion data with caching."""
    if piece_id not in _motion_cache:
        data_path = get_data_path(piece_id)
        with open(data_path / "motion.pkl", "rb") as f:
            motion_data = pickle.load(f)
        with open(data_path / "vis" / "pressed_keys.pkl", "rb") as f:
            pressed_keys = pickle.load(f)
        
        n_frames = min(len(motion_data["left"]["mano_params"]["verts"]), len(pressed_keys))
        _motion_cache[piece_id] = {
            "motion": motion_data,
            "pressed_keys": pressed_keys,
            "n_frames": n_frames
        }
        print(f"Cached motion data for piece {piece_id}: {n_frames} frames")
    
    return _motion_cache[piece_id]


def build_frame_data(motion_data, pressed_keys, frame_idx):
    """Build frame data for a single frame."""
    return {
        "left_vertices": np.round(
            motion_data["left"]["mano_params"]["verts"][frame_idx], 4
        ).tolist(),
        "right_vertices": np.round(
            motion_data["right"]["mano_params"]["verts"][frame_idx], 4
        ).tolist(),
        "left_joints": np.round(
            motion_data["left"]["joints"][frame_idx].flatten(), 4
        ).tolist(),
        "right_joints": np.round(
            motion_data["right"]["joints"][frame_idx].flatten(), 4
        ).tolist(),
        "pressed_keys": pressed_keys[frame_idx].tolist(),
    }


@app.route("/mano_vertices_data/<int:piece_id>/info")
def get_mesh_info(piece_id):
    """Return mesh data info (frame count)."""
    try:
        cached = get_motion_data(piece_id)
        response = jsonify({
            "piece_id": piece_id,
            "n_frames": cached["n_frames"]
        })
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
    except Exception as e:
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500




@app.route("/mano_vertices_data/<int:piece_id>/frames/<int:start>-<int:end>")
def get_frame_range(piece_id, start, end):
    """Return a range of frames as plain JSON."""
    try:
        cached = get_motion_data(piece_id)
        motion_data = cached["motion"]
        pressed_keys = cached["pressed_keys"]
        n_frames = cached["n_frames"]
        
        # Clamp range
        start = max(0, start)
        end = min(end, n_frames - 1)
        
        if start > end:
            response = jsonify({"error": "Invalid range"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 400
        
        # Build frames
        frames = []
        for i in range(start, end + 1):
            frame_data = build_frame_data(motion_data, pressed_keys, i)
            frame_data["frame_idx"] = i
            frames.append(frame_data)
        
        print(f"Serving frames {start}-{end} for piece {piece_id} ({len(frames)} frames)")
        
        # Use jsonify - Flask handles encoding correctly
        response = jsonify(frames)
        response.headers["Access-Control-Allow-Origin"] = "*"
        
        # Remove Content-Length and Content-Encoding to use chunked transfer
        # This prevents ERR_CONTENT_LENGTH_MISMATCH errors with large responses
        if 'Content-Length' in response.headers:
            del response.headers['Content-Length']
        if 'Content-Encoding' in response.headers:
            del response.headers['Content-Encoding']
        
        return response
        
    except Exception as e:
        print(f"Error getting frame range: {e}")
        import traceback
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500


@app.route("/mano_vertices_data/<int:piece_id>/frame/<int:frame_idx>")
def get_single_frame(piece_id, frame_idx):
    """Return single frame data (JSON format)."""
    try:
        cached = get_motion_data(piece_id)
        motion_data = cached["motion"]
        pressed_keys = cached["pressed_keys"]
        n_frames = cached["n_frames"]
        
        if frame_idx < 0 or frame_idx >= n_frames:
            response = jsonify({"error": "Frame index out of range"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 404
        
        frame_data = build_frame_data(motion_data, pressed_keys, frame_idx)
        
        response = jsonify(frame_data)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
        
    except Exception as e:
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500


@app.route("/fingering_data/<int:piece_id>")
def get_fingering_data(piece_id):
    """Return fingering data as JSON.
    
    Priority (highest to lowest):
    - r1: fingering_edited/ (human annotation) - editable
    - r0: fingering_edited_ai/ (AI annotation) - AI prior, editable by annotator
    - original: fingering/ (rule-based) - editable
    """
    try:
        original_path = get_fingering_path(piece_id)  # ./data/fingering/{id}.pkl
        edited_path = get_fingering_edited_path(piece_id)  # ./data/fingering_edited/{id}.pkl (r1)
        ai_edited_path = get_fingering_edited_ai_path(piece_id)  # ./data/fingering_edited_ai/{id}.pkl (r0)
        
        # Load original file (optional - may not exist for new annotations)
        original_data = None
        if original_path.exists():
            try:
                with open(original_path, "rb") as f:
                    original_data = pickle.load(f)
                print(f"Loaded original fingering data for piece {piece_id}: {len(original_data)} frames")
            except Exception as e:
                print(f"Warning: Failed to load original fingering data for piece {piece_id}: {e}")
                original_data = None
        
        # Load edited file if exists (r1 - human annotation)
        edited_data = None
        if edited_path.exists():
            try:
                with open(edited_path, "rb") as f:
                    edited_data = pickle.load(f)
                print(f"Loaded edited fingering data (r1) for piece {piece_id}: {len(edited_data)} frames")
            except Exception as e:
                print(f"Warning: Failed to load edited fingering data for piece {piece_id}: {e}")
                edited_data = None
        
        # Load AI-edited file if exists (r0 - AI annotation)
        ai_edited_data = None
        if ai_edited_path.exists():
            try:
                with open(ai_edited_path, "rb") as f:
                    ai_edited_data = pickle.load(f)
                print(f"Loaded AI-edited fingering data (r0) for piece {piece_id}: {len(ai_edited_data)} frames")
            except Exception as e:
                print(f"Warning: Failed to load AI-edited fingering data for piece {piece_id}: {e}")
                ai_edited_data = None
        
        # Check if we have any data
        if original_data is None and edited_data is None and ai_edited_data is None:
            response = jsonify({"error": "Fingering data not found"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 404
        
        # Determine max frames
        max_frames = 0
        if original_data:
            max_frames = len(original_data)
        if edited_data:
            max_frames = max(max_frames, len(edited_data))
        if ai_edited_data:
            max_frames = max(max_frames, len(ai_edited_data))
        
        # Determine annotation source
        # Priority: r1 (human) > r0 (AI) > original
        is_ai_annotation = False
        if edited_data is not None:
            # r1 exists - use human annotation (editable)
            active_data = edited_data
            is_ai_annotation = False
            print(f"Using r1 (human annotation) for piece {piece_id}")
        elif ai_edited_data is not None:
            # r0 exists but no r1 - use AI annotation as prior (editable by annotator)
            active_data = ai_edited_data
            is_ai_annotation = True
            print(f"Using r0 (AI annotation) for piece {piece_id} - AI prior, editable")
        else:
            # Only original exists
            active_data = original_data
            is_ai_annotation = False
            print(f"Using original (rule-based) for piece {piece_id}")
        
        frames = []
        for frame_idx in range(max_frames):
            # Original frame: empty if no original data
            original_frame = []
            if original_data and frame_idx < len(original_data):
                original_frame = original_data[frame_idx]
            
            # Active frame (based on priority)
            active_frame = []
            if active_data and frame_idx < len(active_data):
                active_frame = active_data[frame_idx]
            
            frame_data = {
                "frame_idx": frame_idx,
                "fingering": active_frame,
                "original_fingering": original_frame,
                "is_ai_annotation": is_ai_annotation,  # r0 flag for AI prior
            }
            
            frames.append(frame_data)
        
        # Use streaming response to avoid Content-Length mismatch issues with large data
        print(f"Successfully prepared fingering data response for piece {piece_id}: {len(frames)} frames")
        
        def generate():
            yield json.dumps(frames)
        
        return Response(
            generate(),
            mimetype='application/json',
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
            }
        )
        
    except Exception as e:
        print(f"Error in get_fingering_data for piece {piece_id}: {e}")
        import traceback
        traceback.print_exc()
        response = jsonify({"error": f"Internal server error: {str(e)}"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500


@app.route("/fingering_data/<int:piece_id>/edit", methods=["POST", "OPTIONS"])
def save_fingering_edit(piece_id):
    """Save fingering edits."""
    import platform
    
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    print(f"Received edit request for piece_id={piece_id}, platform: {platform.system()}")
    
    try:
        data = request.get_json()
        frame_idx = data.get("frame_idx")
        fingering = data.get("fingering")
        
        if frame_idx is None or fingering is None:
            return jsonify({"error": "frame_idx and fingering are required"}), 400
        
        # Type validation and conversion for frame_idx (platform-independent)
        if not isinstance(frame_idx, int):
            original_type = type(frame_idx).__name__
            print(f"Warning: frame_idx type mismatch: expected int, got {original_type} ({frame_idx}), platform: {platform.system()}")
            try:
                # Try to extract integer from dict if it's a dict
                if isinstance(frame_idx, dict):
                    frame_idx = frame_idx.get("frameIndex") or frame_idx.get("frame_idx") or frame_idx.get("index")
                    if frame_idx is None:
                        response = jsonify({"error": f"Cannot extract frame_idx from dict: {data.get('frame_idx')}"})
                        response.headers["Access-Control-Allow-Origin"] = "*"
                        return response, 400
                
                # Convert to int
                frame_idx = int(frame_idx)
                print(f"Converted frame_idx from {original_type} to int: {frame_idx}")
            except (ValueError, TypeError) as e:
                print(f"Error: Cannot convert frame_idx to int: {frame_idx}, type: {type(frame_idx)}, error: {e}, platform: {platform.system()}")
                response = jsonify({"error": f"Invalid frame_idx type: expected int, got {type(frame_idx).__name__}"})
                response.headers["Access-Control-Allow-Origin"] = "*"
                return response, 400
        
        original_path = get_fingering_path(piece_id)  # ./data/fingering/{id}.pkl
        edited_path = get_fingering_edited_path(piece_id)  # ./data/fingering_edited/{id}.pkl
        
        # Acquire lock for this piece to prevent concurrent file access
        lock = get_fingering_lock(piece_id)
        with lock:
            # Ensure edited directory exists
            edited_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy original if edited file doesn't exist
            if not edited_path.exists() and original_path.exists():
                import shutil
                shutil.copy2(original_path, edited_path)
                print(f"Copied original fingering file to edited file for piece {piece_id}")
            
            # Load edited file
            if edited_path.exists():
                with open(edited_path, "rb") as f:
                    all_fingering_data = pickle.load(f)
            elif original_path.exists():
                with open(original_path, "rb") as f:
                    all_fingering_data = pickle.load(f)
                # Create edited file by copying original
                import shutil
                shutil.copy2(original_path, edited_path)
                print(f"Created edited fingering file from original for piece {piece_id}")
            else:
                # Create new if original doesn't exist
                all_fingering_data = []
                print(f"Warning: No original fingering file found, creating new one for piece {piece_id}")
            
            # Ensure all_fingering_data is a list
            if not isinstance(all_fingering_data, list):
                print(f"Error: all_fingering_data is not a list, type: {type(all_fingering_data)}, platform: {platform.system()}")
                response = jsonify({"error": "Invalid data structure: expected list"})
                response.headers["Access-Control-Allow-Origin"] = "*"
                return response, 500
            
            # Extend array to match frame index
            while len(all_fingering_data) <= frame_idx:
                all_fingering_data.append([])
            
            # Update fingering for the frame
            all_fingering_data[frame_idx] = fingering
            
            # Save file
            with open(edited_path, "wb") as f:
                pickle.dump(all_fingering_data, f)
        
        print(f"Saved fingering edit for piece {piece_id}, frame {frame_idx}, platform: {platform.system()}")
        response = jsonify({"success": True, "frame_idx": frame_idx})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 200
        
    except Exception as e:
        print(f"Error saving fingering edit: {e}")
        import traceback
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500


@app.route("/fingering_data/<int:piece_id>/edit_block", methods=["POST", "OPTIONS"])
def save_fingering_edit_block(piece_id):
    """Save fingering block edits (save multiple frames at once)."""
    import platform
    
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    try:
        data = request.get_json()
        frames = data.get("frames")
        
        if not frames or not isinstance(frames, list):
            return jsonify({"error": "frames array is required"}), 400
        
        if len(frames) == 0:
            return jsonify({"success": True, "frames_updated": 0}), 200
        
        # Validate and normalize frame_idx values (platform-independent type checking)
        normalized_frames = []
        for i, frame_data in enumerate(frames):
            if not isinstance(frame_data, dict):
                print(f"Warning: Frame {i} is not a dict, type: {type(frame_data)}, platform: {platform.system()}")
                continue
            
            frame_idx = frame_data.get("frame_idx")
            fingering = frame_data.get("fingering")
            
            # Type validation and conversion for frame_idx
            if frame_idx is None:
                print(f"Warning: frame_idx is None for frame {i}, platform: {platform.system()}")
                continue
            
            # Convert to int if not already an integer
            if not isinstance(frame_idx, int):
                original_type = type(frame_idx).__name__
                print(f"Warning: frame_idx type mismatch for frame {i}: expected int, got {original_type} ({frame_idx}), platform: {platform.system()}")
                try:
                    # Try to extract integer from dict if it's a dict
                    if isinstance(frame_idx, dict):
                        # Try common dict keys that might contain the actual index
                        frame_idx = frame_idx.get("frameIndex") or frame_idx.get("frame_idx") or frame_idx.get("index")
                        if frame_idx is None:
                            print(f"Error: Could not extract frame_idx from dict: {frame_data.get('frame_idx')}")
                            continue
                    
                    # Convert to int
                    frame_idx = int(frame_idx)
                    print(f"Converted frame_idx from {original_type} to int: {frame_idx}")
                except (ValueError, TypeError) as e:
                    print(f"Error: Cannot convert frame_idx to int: {frame_idx}, type: {type(frame_idx)}, error: {e}, platform: {platform.system()}")
                    continue
            
            if fingering is None:
                print(f"Warning: fingering is None for frame {i}, platform: {platform.system()}")
                continue
            
            normalized_frames.append({
                "frame_idx": frame_idx,
                "fingering": fingering
            })
        
        if len(normalized_frames) == 0:
            return jsonify({"error": "No valid frames to save"}), 400
        
        # Calculate max_frame_idx from normalized frames (all are guaranteed to be int)
        max_frame_idx = max(f["frame_idx"] for f in normalized_frames)
        
        original_path = get_fingering_path(piece_id)  # ./data/fingering/{id}.pkl
        edited_path = get_fingering_edited_path(piece_id)  # ./data/fingering_edited/{id}.pkl
        
        # Acquire lock for this piece to prevent concurrent file access
        lock = get_fingering_lock(piece_id)
        with lock:
            # Ensure edited directory exists
            edited_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy original if edited file doesn't exist
            if not edited_path.exists() and original_path.exists():
                import shutil
                shutil.copy2(original_path, edited_path)
                print(f"Copied original fingering file to edited file for piece {piece_id}")
            
            # Load edited file
            if edited_path.exists():
                with open(edited_path, "rb") as f:
                    all_fingering_data = pickle.load(f)
            elif original_path.exists():
                with open(original_path, "rb") as f:
                    all_fingering_data = pickle.load(f)
                import shutil
                shutil.copy2(original_path, edited_path)
            else:
                all_fingering_data = []
            
            # Ensure all_fingering_data is a list
            if not isinstance(all_fingering_data, list):
                print(f"Error: all_fingering_data is not a list, type: {type(all_fingering_data)}, platform: {platform.system()}")
                response = jsonify({"error": "Invalid data structure: expected list"})
                response.headers["Access-Control-Allow-Origin"] = "*"
                return response, 500
            
            # Extend array to required size (max_frame_idx is guaranteed to be int)
            while len(all_fingering_data) <= max_frame_idx:
                all_fingering_data.append([])
            
            # Update fingering for each frame
            for frame_data in normalized_frames:
                frame_idx = frame_data["frame_idx"]
                fingering = frame_data["fingering"]
                all_fingering_data[frame_idx] = fingering
            
            # Save file
            with open(edited_path, "wb") as f:
                pickle.dump(all_fingering_data, f)
        
        frame_indices = [f["frame_idx"] for f in normalized_frames]
        start_frame = min(frame_indices)
        end_frame = max(frame_indices)
        print(f"Saved fingering block for piece {piece_id}: frames {start_frame}~{end_frame} ({len(normalized_frames)} frames), platform: {platform.system()}")
        
        response = jsonify({
            "success": True, 
            "frames_updated": len(normalized_frames),
            "start_frame": start_frame,
            "end_frame": end_frame
        })
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 200
        
    except Exception as e:
        print(f"Error saving fingering block: {e}")
        import traceback
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500


@app.route("/piano_mesh/<path:filename>")
def get_obj_file(filename):
    piano_mesh_dir = get_release_root_path() / "for_elise" / "piano_meshes"
    if (piano_mesh_dir / filename).exists():
        response = send_from_directory(piano_mesh_dir, filename)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
    else:
        return jsonify({"error": "File not found"}), 404


# ==========================================
# Annotation Status Management API (2-Stage Review)
# Stage 1: Annotator review
# Stage 2: Professional pianist review
# ==========================================

def get_annotation_status_dir():
    """Return annotation status directory path (per-piece files)."""
    return get_release_root_path() / "data" / "metadata" / "annotation" / "status"


def get_piece_status_path(piece_id):
    """Return status file path for a specific piece."""
    return get_annotation_status_dir() / f"{int(piece_id):03d}.json"


def load_piece_status(piece_id):
    """Load annotation status for a specific piece."""
    status_path = get_piece_status_path(piece_id)
    if status_path.exists():
        with open(status_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "piece_id": int(piece_id),
        "fingering_completed": None,
        "post_completed": None,
        "review1": None,
        "review2": None,
        "review3": None
    }


def save_piece_status(piece_id, status_data):
    """Save annotation status for a specific piece."""
    status_dir = get_annotation_status_dir()
    status_dir.mkdir(parents=True, exist_ok=True)
    
    status_path = get_piece_status_path(piece_id)
    status_data["piece_id"] = int(piece_id)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2, ensure_ascii=False)


def load_all_annotation_status():
    """Load annotation status for all pieces (aggregated view)."""
    status_dir = get_annotation_status_dir()
    result = {"review1": {}, "review2": {}, "review3": {}, "fingering_completed": {}, "post_completed": {}}
    
    if not status_dir.exists():
        return result
    
    for status_file in status_dir.glob("*.json"):
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                piece_data = json.load(f)
            
            piece_id_str = str(piece_data.get("piece_id", status_file.stem))
            
            if piece_data.get("review1"):
                result["review1"][piece_id_str] = piece_data["review1"]
            if piece_data.get("review2"):
                result["review2"][piece_id_str] = piece_data["review2"]
            if piece_data.get("review3"):
                result["review3"][piece_id_str] = piece_data["review3"]
            if piece_data.get("fingering_completed"):
                result["fingering_completed"][piece_id_str] = piece_data["fingering_completed"]
            if piece_data.get("post_completed"):
                result["post_completed"][piece_id_str] = piece_data["post_completed"]
        except Exception as e:
            print(f"Error loading {status_file}: {e}")
    
    return result


@app.route("/annotation_status")
def get_annotation_status():
    """Return annotation status for all pieces (both stages)."""
    status = load_all_annotation_status()
    response = jsonify(status)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/annotation_sources")
def get_annotation_sources():
    """Return annotation source (r0/r1) for each piece.
    
    r0: AI annotation (still in fingering_edited_ai/ - not yet human-reviewed)
    r1: Human annotation (in fingering_edited/ but NOT in fingering_edited_ai/)
    original: Rule-based only (fingering/)
    """
    fingering_edited_path = get_release_root_path() / "data" / "fingering_edited"
    fingering_edited_ai_path = get_release_root_path() / "data" / "fingering_edited_ai"
    fingering_path = get_release_root_path() / "data" / "fingering"
    
    sources = {}
    
    # Get AI-annotated piece IDs (still not human-reviewed)
    ai_piece_ids = set()
    if fingering_edited_ai_path.exists():
        for f in fingering_edited_ai_path.glob("*.pkl"):
            try:
                ai_piece_ids.add(int(f.stem))
            except ValueError:
                continue
    
    # Get all piece IDs from fingering folder
    if fingering_path.exists():
        for f in fingering_path.glob("*.pkl"):
            try:
                piece_id = int(f.stem)
                sources[str(piece_id)] = "original"
            except ValueError:
                continue
    
    # Classify based on fingering_edited_ai presence
    if fingering_edited_path.exists():
        for f in fingering_edited_path.glob("*.pkl"):
            try:
                piece_id = int(f.stem)
                if piece_id in ai_piece_ids:
                    # Still in AI folder = not yet human-reviewed
                    sources[str(piece_id)] = "r0"
                else:
                    # Not in AI folder = human has reviewed/edited
                    sources[str(piece_id)] = "r1"
            except ValueError:
                continue
    
    response = jsonify(sources)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/annotation_status/<int:piece_id>", methods=["GET"])
def get_piece_annotation_status(piece_id):
    """Return annotation status for a specific piece (both stages + subtasks)."""
    piece_status = load_piece_status(piece_id)
    
    review1_status = piece_status.get("review1")
    review2_status = piece_status.get("review2")
    review3_status = piece_status.get("review3")
    fingering_status = piece_status.get("fingering_completed")
    post_status = piece_status.get("post_completed")
    
    response = jsonify({
        "piece_id": piece_id,
        "review1_completed": review1_status is not None,
        "review1_details": review1_status,
        "review2_completed": review2_status is not None,
        "review2_details": review2_status,
        "review3_completed": review3_status is not None,
        "review3_details": review3_status,
        "fingering_completed": fingering_status is not None,
        "fingering_details": fingering_status,
        "post_completed": post_status is not None,
        "post_details": post_status
    })
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


def get_piece_annotator(piece_id):
    """Get annotator number for a piece from split.json"""
    try:
        split_path = get_release_root_path() / "data" / "metadata" / "annotation" / "split.json"
        if split_path.exists():
            with open(split_path, "r", encoding="utf-8") as f:
                split_data = json.load(f)
            piece_to_annotator = split_data.get("piece_to_annotator", {})
            return piece_to_annotator.get(str(piece_id))
    except Exception as e:
        print(f"Error getting annotator for piece {piece_id}: {e}")
    return None


@app.route("/annotation_status/<int:piece_id>", methods=["POST", "OPTIONS"])
def update_piece_annotation_status(piece_id):
    """Update annotation status for a specific piece.
    
    Request body:
        stage: 1, 2, or 3 (default: 1)
        completed: true/false
        notes: optional string
    
    Special rules:
        - A3 (annotator 3): R1 completion also completes R2
        - A4 (annotator 4): R1 completion also completes R3
    """
    from datetime import datetime
    
    # Handle CORS preflight request
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    try:
        data = request.get_json() or {}
        stage = data.get("stage", 1)
        completed = data.get("completed", True)
        notes = data.get("notes", "")
        
        if stage not in [1, 2, 3]:
            response = jsonify({"error": "stage must be 1, 2, or 3"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 400
        
        piece_status = load_piece_status(piece_id)
        review_key = f"review{stage}"
        auto_review_completed = None  # Track auto-completed review
        
        if completed:
            # Mark as completed
            piece_status[review_key] = {
                "completed_at": datetime.now().isoformat(),
                "notes": notes
            }
            print(f"Marked piece {piece_id} as review{stage} completed")
            
            # Special rule: A3/A4 auto-complete R2/R3 when R1 is completed
            if stage == 1:
                annotator = get_piece_annotator(piece_id)
                if annotator == 3 and piece_status.get("review2") is None:
                    # A3: R1 completion also completes R2
                    piece_status["review2"] = {
                        "completed_at": datetime.now().isoformat(),
                        "notes": "Auto-completed (A3 reviewer)"
                    }
                    auto_review_completed = 2
                    print(f"Auto-marked piece {piece_id} R2 as completed (A3 reviewer)")
                elif annotator == 4 and piece_status.get("review3") is None:
                    # A4: R1 completion also completes R3
                    piece_status["review3"] = {
                        "completed_at": datetime.now().isoformat(),
                        "notes": "Auto-completed (A4 reviewer)"
                    }
                    auto_review_completed = 3
                    print(f"Auto-marked piece {piece_id} R3 as completed (A4 reviewer)")
        else:
            # Unmark as completed
            piece_status[review_key] = None
            print(f"Unmarked piece {piece_id} as review{stage} completed")
            
            # If R1 is uncompleted, also uncomplete auto-completed reviews
            if stage == 1:
                annotator = get_piece_annotator(piece_id)
                if annotator == 3:
                    piece_status["review2"] = None
                    print(f"Auto-unmarked piece {piece_id} R2 (A3 reviewer)")
                elif annotator == 4:
                    piece_status["review3"] = None
                    print(f"Auto-unmarked piece {piece_id} R3 (A4 reviewer)")
        
        save_piece_status(piece_id, piece_status)
        
        response = jsonify({
            "success": True,
            "piece_id": piece_id,
            "stage": stage,
            "completed": completed,
            "auto_review_completed": auto_review_completed
        })
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
        
    except Exception as e:
        print(f"Error updating annotation status: {e}")
        import traceback
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500


@app.route("/annotation_status/<int:piece_id>/subtask", methods=["POST", "OPTIONS"])
def update_subtask_status(piece_id):
    """Update fingering/post completion status for a specific piece.
    
    Request body:
        subtask: 'fingering' or 'post'
        completed: true/false
    """
    from datetime import datetime
    
    # Handle CORS preflight request
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    try:
        data = request.get_json() or {}
        subtask = data.get("subtask", "")
        completed = data.get("completed", True)
        
        if subtask not in ["fingering", "post"]:
            response = jsonify({"error": "subtask must be 'fingering' or 'post'"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 400
        
        piece_status = load_piece_status(piece_id)
        status_key = f"{subtask}_completed"
        
        if completed:
            # Mark as completed
            piece_status[status_key] = {
                "completed_at": datetime.now().isoformat(),
                "notes": ""
            }
            print(f"Marked piece {piece_id} {subtask} as completed")
        else:
            # Unmark as completed
            piece_status[status_key] = None
            print(f"Unmarked piece {piece_id} {subtask} as completed")
        
        # Check if both fingering and post are completed -> auto-complete R1
        fingering_done = piece_status.get("fingering_completed") is not None
        post_done = piece_status.get("post_completed") is not None
        
        auto_r1_updated = False
        if fingering_done and post_done:
            # Auto-mark R1 as completed
            if piece_status.get("review1") is None:
                piece_status["review1"] = {
                    "completed_at": datetime.now().isoformat(),
                    "notes": "Auto-completed (fingering + post done)"
                }
                auto_r1_updated = True
                print(f"Auto-marked piece {piece_id} R1 as completed")
        
        save_piece_status(piece_id, piece_status)
        
        response = jsonify({
            "success": True,
            "piece_id": piece_id,
            "subtask": subtask,
            "completed": completed,
            "auto_r1_completed": auto_r1_updated
        })
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
        
    except Exception as e:
        print(f"Error updating subtask status: {e}")
        import traceback
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500


# ==========================================
# Motion Issues Management API
# Track problematic motion segments
# ==========================================

def get_motion_issues_path(piece_id):
    """Return motion issues file path for a specific piece (./data/motion_issues/{id}.json)."""
    return get_release_root_path() / "data" / "motion_issues" / f"{piece_id:03d}.json"


def load_motion_issues(piece_id):
    """Load motion issues data for a specific piece."""
    issues_path = get_motion_issues_path(piece_id)
    if issues_path.exists():
        with open(issues_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_motion_issues(piece_id, issues_data):
    """Save motion issues data for a specific piece."""
    issues_path = get_motion_issues_path(piece_id)
    issues_path.parent.mkdir(parents=True, exist_ok=True)
    with open(issues_path, "w", encoding="utf-8") as f:
        json.dump(issues_data, f, indent=2, ensure_ascii=False)


@app.route("/motion_issues/<int:piece_id>", methods=["GET"])
def get_piece_motion_issues(piece_id):
    """Return motion issues for a specific piece."""
    piece_issues = load_motion_issues(piece_id)
    response = jsonify(piece_issues)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/motion_issues/<int:piece_id>", methods=["POST", "OPTIONS"])
def add_motion_issue(piece_id):
    """Add a new motion issue for a specific piece."""
    from datetime import datetime
    import uuid
    
    # Handle CORS preflight request
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    try:
        data = request.get_json() or {}
        start_time = data.get("start_time")
        end_time = data.get("end_time")
        note = data.get("note", "")
        
        if start_time is None or end_time is None:
            response = jsonify({"error": "start_time and end_time are required"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 400
        
        if start_time >= end_time:
            response = jsonify({"error": "start_time must be less than end_time"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 400
        
        issues = load_motion_issues(piece_id)
        
        new_issue = {
            "id": str(uuid.uuid4()),
            "start_time": float(start_time),
            "end_time": float(end_time),
            "note": note,
            "created_at": datetime.now().isoformat()
        }
        
        issues.append(new_issue)
        save_motion_issues(piece_id, issues)
        
        print(f"Added motion issue for piece {piece_id}: {start_time}s ~ {end_time}s")
        
        response = jsonify({
            "success": True,
            "issue": new_issue
        })
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
        
    except Exception as e:
        print(f"Error adding motion issue: {e}")
        import traceback
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500


@app.route("/motion_issues/<int:piece_id>/<issue_id>", methods=["DELETE", "OPTIONS"])
def delete_motion_issue(piece_id, issue_id):
    """Delete a motion issue."""
    
    # Handle CORS preflight request
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    try:
        issues = load_motion_issues(piece_id)
        
        if not issues:
            response = jsonify({"error": "No issues found for this piece"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 404
        
        # Find and remove the issue
        original_len = len(issues)
        issues = [i for i in issues if i["id"] != issue_id]
        
        if len(issues) == original_len:
            response = jsonify({"error": "Issue not found"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 404
        
        # Save updated issues (or delete file if empty)
        if issues:
            save_motion_issues(piece_id, issues)
        else:
            # Delete file if no issues left
            issues_path = get_motion_issues_path(piece_id)
            if issues_path.exists():
                issues_path.unlink()
        
        print(f"Deleted motion issue {issue_id} from piece {piece_id}")
        
        response = jsonify({"success": True})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
        
    except Exception as e:
        print(f"Error deleting motion issue: {e}")
        import traceback
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500


# ==========================================
# Post-Playing Segment API
# Mark the segment after piano playing ends
# ==========================================

def get_post_playing_path(piece_id):
    """Return post playing file path for a specific piece (./data/post_playing/{id}.json)."""
    return get_release_root_path() / "data" / "post_playing" / f"{piece_id:03d}.json"


def load_post_playing(piece_id):
    """Load post playing data for a specific piece."""
    path = get_post_playing_path(piece_id)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_post_playing(piece_id, data):
    """Save post playing data for a specific piece."""
    path = get_post_playing_path(piece_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@app.route("/post_playing/<int:piece_id>", methods=["GET"])
def get_post_playing(piece_id):
    """Return post-playing segment for a specific piece."""
    segment = load_post_playing(piece_id)
    response = jsonify(segment)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/post_playing/<int:piece_id>", methods=["POST", "OPTIONS"])
def save_post_playing_segment(piece_id):
    """Save post-playing segment for a specific piece (overwrites existing)."""
    # Handle CORS preflight request
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    try:
        req_data = request.get_json() or {}
        start_time = req_data.get("start_time")
        end_time = req_data.get("end_time")
        
        if start_time is None or end_time is None:
            response = jsonify({"error": "start_time and end_time are required"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 400
        
        if start_time >= end_time:
            response = jsonify({"error": "start_time must be less than end_time"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 400
        
        segment_data = {
            "start_time": float(start_time),
            "end_time": float(end_time)
        }
        save_post_playing(piece_id, segment_data)
        
        print(f"Saved post-playing segment for piece {piece_id}: {start_time}s ~ {end_time}s")
        
        response = jsonify({
            "success": True,
            "segment": segment_data
        })
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
        
    except Exception as e:
        print(f"Error saving post-playing segment: {e}")
        import traceback
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500


@app.route("/post_playing/<int:piece_id>", methods=["DELETE", "OPTIONS"])
def delete_post_playing_segment(piece_id):
    """Delete post-playing segment for a specific piece."""
    
    # Handle CORS preflight request
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    try:
        path = get_post_playing_path(piece_id)
        
        if not path.exists():
            response = jsonify({"error": "No post-playing segment found for this piece"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 404
        
        path.unlink()
        
        print(f"Deleted post-playing segment for piece {piece_id}")
        
        response = jsonify({"success": True})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
        
    except Exception as e:
        print(f"Error deleting post-playing segment: {e}")
        import traceback
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500


# ==========================================
# Test Segments API
# Mark segments suitable for user evaluation (TEST dataset only)
# ==========================================

def get_test_segments_path(piece_id):
    """Return test segments file path for a specific piece (./data/test_segments/{id}.json)."""
    return get_release_root_path() / "data" / "test_segments" / f"{piece_id:03d}.json"


def load_test_segments(piece_id):
    """Load test segments data for a specific piece."""
    segments_path = get_test_segments_path(piece_id)
    if segments_path.exists():
        with open(segments_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_test_segments(piece_id, segments_data):
    """Save test segments data for a specific piece."""
    segments_path = get_test_segments_path(piece_id)
    segments_path.parent.mkdir(parents=True, exist_ok=True)
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments_data, f, indent=2, ensure_ascii=False)


@app.route("/test_segments/<int:piece_id>", methods=["GET"])
def get_test_segments(piece_id):
    """Return test segments for a specific piece."""
    segments = load_test_segments(piece_id)
    response = jsonify(segments)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/test_segments/<int:piece_id>", methods=["POST", "OPTIONS"])
def add_test_segment(piece_id):
    """Add a new test segment for a specific piece."""
    from datetime import datetime
    import uuid
    
    # Handle CORS preflight request
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    try:
        req_data = request.get_json() or {}
        start_time = req_data.get("start_time")
        end_time = req_data.get("end_time")
        note = req_data.get("note", "")
        
        if start_time is None or end_time is None:
            response = jsonify({"error": "start_time and end_time are required"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 400
        
        if start_time >= end_time:
            response = jsonify({"error": "start_time must be less than end_time"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 400
        
        # Load existing segments
        segments = load_test_segments(piece_id)
        
        new_segment = {
            "id": str(uuid.uuid4()),
            "start_time": float(start_time),
            "end_time": float(end_time),
            "note": note,
            "created_at": datetime.now().isoformat()
        }
        
        segments.append(new_segment)
        save_test_segments(piece_id, segments)
        
        print(f"Added test segment for piece {piece_id}: {start_time}s ~ {end_time}s")
        
        response = jsonify({
            "success": True,
            "segment": new_segment
        })
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
        
    except Exception as e:
        print(f"Error adding test segment: {e}")
        import traceback
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500


@app.route("/test_segments/<int:piece_id>/<segment_id>", methods=["DELETE", "OPTIONS"])
def delete_test_segment(piece_id, segment_id):
    """Delete a test segment."""
    
    # Handle CORS preflight request
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    try:
        segments = load_test_segments(piece_id)
        
        if not segments:
            response = jsonify({"error": "No segments found for this piece"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 404
        
        # Find and remove the segment
        original_len = len(segments)
        segments = [s for s in segments if s["id"] != segment_id]
        
        if len(segments) == original_len:
            response = jsonify({"error": "Segment not found"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 404
        
        # Save updated segments (or delete file if empty)
        if segments:
            save_test_segments(piece_id, segments)
        else:
            # Delete file if no segments left
            segments_path = get_test_segments_path(piece_id)
            if segments_path.exists():
                segments_path.unlink()
        
        print(f"Deleted test segment {segment_id} from piece {piece_id}")
        
        response = jsonify({"success": True})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
        
    except Exception as e:
        print(f"Error deleting test segment: {e}")
        import traceback
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500


# ==========================================
# Annotation Progress API
# Track where user left off during annotation
# ==========================================

def get_annotation_progress_path(piece_id):
    """Return annotation progress file path for a specific piece."""
    return get_release_root_path() / "data" / "annotation_progress" / f"{piece_id:03d}.json"


def load_annotation_progress(piece_id):
    """Load annotation progress data for a specific piece."""
    progress_path = get_annotation_progress_path(piece_id)
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_annotation_progress(piece_id, progress_data):
    """Save annotation progress data for a specific piece."""
    progress_path = get_annotation_progress_path(piece_id)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)


@app.route("/annotation_progress/<int:piece_id>", methods=["GET"])
def get_annotation_progress(piece_id):
    """Return annotation progress for a specific piece."""
    progress = load_annotation_progress(piece_id)
    if progress is None:
        response = jsonify(None)
    else:
        response = jsonify(progress)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/annotation_progress/<int:piece_id>", methods=["POST", "OPTIONS"])
def update_annotation_progress(piece_id):
    """Update annotation progress for a specific piece."""
    from datetime import datetime
    
    # Handle CORS preflight request
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    try:
        req_data = request.get_json() or {}
        last_frame = req_data.get("last_frame")
        last_time_seconds = req_data.get("last_time_seconds")
        
        if last_frame is None or last_time_seconds is None:
            response = jsonify({"error": "last_frame and last_time_seconds are required"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 400
        
        progress_data = {
            "last_frame": int(last_frame),
            "last_time_seconds": float(last_time_seconds),
            "updated_at": datetime.now().isoformat()
        }
        
        save_annotation_progress(piece_id, progress_data)
        
        print(f"Saved annotation progress for piece {piece_id}: frame {last_frame}, time {last_time_seconds:.2f}s")
        
        response = jsonify({
            "success": True,
            "progress": progress_data
        })
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
        
    except Exception as e:
        print(f"Error saving annotation progress: {e}")
        import traceback
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500


@app.route("/hitting_points")
def get_hitting_points():
    """Return hitting points data."""
    # Try multiple possible locations for hitting_point.json
    possible_paths = [
        get_release_root_path() / "for_elise" / "metadata" / "hitting_point.json",
        get_release_root_path() / "merge" / "hitting_points.pkl",  # Fallback to pickle
    ]
    
    # Try JSON first
    json_path = possible_paths[0]
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # JSON data is already in list format, return as-is
        return jsonify(data)
    
    # Fallback to pickle
    pickle_path = possible_paths[1]
    if pickle_path.exists():
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        
        # Convert numpy arrays to lists
        result = {
            "hitting_points": data["hitting_points"].tolist(),
            "keyboard_bounds": {
                "min": data["keyboard_bounds"]["min"].tolist(),
                "max": data["keyboard_bounds"]["max"].tolist(),
                "center": data["keyboard_bounds"]["center"].tolist(),
                "size": data["keyboard_bounds"]["size"].tolist(),
            },
            "key_features": [
                {
                    "key_idx": kf["key_idx"],
                    "midi_note": kf["midi_note"],
                    "pitch_name": kf["pitch_name"],
                    "is_black_key": kf["is_black_key"],
                    "hitting_point": kf["hitting_point"].tolist(),
                }
                for kf in data["key_features"]
            ]
        }
        return jsonify(result)
    
    return jsonify({
        "error": "Hitting points data not found",
        "searched_paths": [str(p) for p in possible_paths]
    }), 404


@app.route("/midi_notes/<int:piece_id>")
def get_midi_notes(piece_id):
    """Return MIDI note events for a piece."""
    if not MIDO_AVAILABLE:
        print(f"MIDI notes API called but mido is not available (MIDO_AVAILABLE={MIDO_AVAILABLE})")
        response = jsonify({"error": "mido library not available", "MIDO_AVAILABLE": MIDO_AVAILABLE})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 503
    
    try:
        data_path = get_data_path(piece_id)
        midi_path = data_path / "midi.mid"
        
        if not midi_path.exists():
            response = jsonify({
                "error": "MIDI file not found",
                "path": str(midi_path)
            })
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 404
        
        # Load motion data to get frame count and FPS
        motion_path = data_path / "motion.pkl"
        if not motion_path.exists():
            response = jsonify({"error": "motion.pkl not found"})
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, 404
        
        with open(motion_path, 'rb') as f:
            motion = pickle.load(f)
        
        # Determine frame count
        num_frames = 0
        if 'left' in motion and 'joints' in motion['left']:
            num_frames = motion['left']['joints'].shape[0]
        elif 'right' in motion and 'joints' in motion['right']:
            num_frames = motion['right']['joints'].shape[0]
        
        # FPS (default to 60000/1001 = 59.94, NTSC standard)
        fps = 60000.0 / 1001.0
        
        # Parse MIDI file
        midi = mido.MidiFile(midi_path)
        
        # MIDI note number -> key index conversion (MIDI 21 = A0 = key index 0)
        MIDI_OFFSET = 21
        
        # Track active notes: key_idx -> onset_time
        active_notes = {}
        # Result: List of {key_idx, onset_frame, offset_frame}
        note_events = []
        
        for track in midi.tracks:
            track_time = 0.0
            
            for msg in track:
                # Convert delta time to seconds
                track_time += mido.tick2second(msg.time, midi.ticks_per_beat, 500000)
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    key_idx = msg.note - MIDI_OFFSET
                    if 0 <= key_idx < 88:
                        active_notes[key_idx] = track_time
                        
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    key_idx = msg.note - MIDI_OFFSET
                    if 0 <= key_idx < 88 and key_idx in active_notes:
                        onset_time = active_notes.pop(key_idx)
                        onset_frame = int(onset_time * fps)
                        offset_frame = int(track_time * fps)
                        
                        # Clamp to valid frame range
                        onset_frame = max(0, min(onset_frame, num_frames - 1))
                        offset_frame = max(0, min(offset_frame, num_frames - 1))
                        
                        note_events.append({
                            "key_idx": int(key_idx),
                            "onset_frame": int(onset_frame),
                            "offset_frame": int(offset_frame)
                        })
        
        # Handle unclosed notes (until last frame)
        for key_idx, onset_time in active_notes.items():
            onset_frame = int(onset_time * fps)
            onset_frame = max(0, min(onset_frame, num_frames - 1))
            note_events.append({
                "key_idx": int(key_idx),
                "onset_frame": int(onset_frame),
                "offset_frame": int(num_frames - 1)
            })
        
        # Sort by key_idx, then onset_frame
        note_events.sort(key=lambda x: (x["key_idx"], x["onset_frame"]))
        
        response = jsonify({
            "fps": fps,
            "num_frames": int(num_frames),
            "notes": note_events
        })
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500


if __name__ == "__main__":
    import sys
    PORT = 8080
    
    # Kill existing process using the port
    kill_process_on_port(PORT)
    
    # Check gunicorn availability
    use_gunicorn = "--gunicorn" in sys.argv or "-g" in sys.argv
    
    if use_gunicorn:
        try:
            import gunicorn.app.base
            
            class StandaloneApplication(gunicorn.app.base.BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()
                
                def load_config(self):
                    for key, value in self.options.items():
                        if key in self.cfg.settings and value is not None:
                            self.cfg.set(key.lower(), value)
                
                def load(self):
                    return self.application
            
            # Set number of workers based on CPU core count
            import multiprocessing
            workers = multiprocessing.cpu_count() * 2 + 1
            
            options = {
                'bind': f'0.0.0.0:{PORT}',
                'workers': workers,
                'worker_class': 'gevent',  # Use async workers
                'worker_connections': 1000,
                'timeout': 120,
                'keepalive': 5,
            }
            
            print(f"Starting Gunicorn with {workers} workers (gevent)...")
            StandaloneApplication(app, options).run()
            
        except ImportError:
            print("gunicorn not installed, falling back to Flask dev server")
            app.run(host="0.0.0.0", port=PORT, threaded=True)
    else:
        # Default: Flask development server (threaded mode)
        print("Starting Flask development server (use --gunicorn for production)")
        app.run(host="0.0.0.0", port=PORT, threaded=True)
