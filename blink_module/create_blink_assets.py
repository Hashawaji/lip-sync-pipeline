import cv2
import numpy as np
import dlib
import argparse
import os
import sys

# --- Constants ---
MASK_LANDMARKS = list(range(17, 27)) + list(range(36, 48))
ALIGN_LANDMARKS = list(range(17, 68)) 
JPEG_QUALITY = 95 # High quality JPEG
BBOX_PADDING = 15 # Pixels to add around the mask

# --- Helper Functions ---

def get_landmarks(image, detector, predictor):
    try:
        if image.ndim > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        rects = detector(gray, 1)
        if len(rects) == 0: return None
        shape = predictor(gray, rects[0])
        return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
    except Exception as e:
        print(f"Error in get_landmarks: {e}")
        return None

def create_feathered_mask(image_shape, landmarks):
    mask = np.zeros(image_shape[:2], dtype=np.float32)
    mask_points = landmarks[MASK_LANDMARKS]
    hull = cv2.convexHull(mask_points)
    cv2.fillConvexPoly(mask, np.int32(hull), (1.0))
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (101, 101), 0)
    mask = np.clip(mask, 0.0, 1.0)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # Return as 3-channel

def get_crop_bbox(landmarks, image_shape):
    """Calculates a tight bounding box around the eye landmarks."""
    mask_points = landmarks[MASK_LANDMARKS]
    x_min = np.min(mask_points[:, 0]) - BBOX_PADDING
    x_max = np.max(mask_points[:, 0]) + BBOX_PADDING
    y_min = np.min(mask_points[:, 1]) - BBOX_PADDING
    y_max = np.max(mask_points[:, 1]) + BBOX_PADDING
    
    # Clamp to image boundaries
    x_min = int(max(0, x_min))
    y_min = int(max(0, y_min))
    x_max = int(min(image_shape[1], x_max))
    y_max = int(min(image_shape[0], y_max))
    
    # Return as [x, y, w, h]
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def crop_and_compress(frame, bbox):
    """Crops frame with bbox [x,y,w,h] and JPEG-encodes it."""
    x, y, w, h = bbox
    cropped_frame = frame[y:y+h, x:x+w]
    
    # Encode to JPEG in memory
    _, jpeg_bytes = cv2.imencode('.jpg', cropped_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return jpeg_bytes

# --- Main Asset Creation Function ---

def create_blink_assets(blink_video_path, source_image_path, start_time, peak_time, end_time, dlib_model_path, output_path):
    print("Loading dlib face detector and landmark predictor...")
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(dlib_model_path)
    except Exception as e:
        print(f"Error loading dlib model: {e}"); sys.exit(1)

    print(f"Loading source image: {source_image_path}")
    anchor_image = cv2.imread(source_image_path)
    if anchor_image is None: raise IOError(f"Could not read source_image")
    
    anchor_landmarks = get_landmarks(anchor_image, detector, predictor)
    if anchor_landmarks is None: raise ValueError("No face found in source_image.")
        
    print("Creating feathered alpha mask...")
    alpha_mask = create_feathered_mask(anchor_image.shape, anchor_landmarks)
    
    # --- NEW: Cropping ---
    print("Calculating eye region bounding box...")
    bbox = get_crop_bbox(anchor_landmarks, anchor_image.shape)
    x, y, w, h = bbox
    print(f"  - Crop box (x,y,w,h): [{x}, {y}, {w}, {h}]")
    
    # Crop the anchor image and mask
    cropped_anchor_image = anchor_image[y:y+h, x:x+w]
    cropped_alpha_mask = alpha_mask[y:y+h, x:x+w]
    
    # Compress the anchor image crop
    _, anchor_eye_jpeg = cv2.imencode('.jpg', cropped_anchor_image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    
    print(f"Loading blink video: {blink_video_path}")
    cap = cv2.VideoCapture(blink_video_path)
    if not cap.isOpened(): raise IOError(f"Could not open blink_video")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    peak_frame = int(peak_time * fps)
    end_frame = int(end_time * fps)
    
    if not (start_frame < peak_frame < end_frame):
        raise ValueError("Timings must be in order: start_time < peak_time < end_time.")
        
    print(f"Extracting, stabilizing, and compressing frames {start_frame} to {end_frame}...")
    
    stabilized_blink_frames = []
    
    for i in range(end_frame + 1):
        ret, frame = cap.read()
        if not ret: break
        if i < start_frame: continue
            
        frame_landmarks = get_landmarks(frame, detector, predictor)
        if frame_landmarks is None:
            print(f"Warning: No face in blink video frame {i}. Skipping."); continue
            
        M, _ = cv2.estimateAffinePartial2D(
            frame_landmarks[ALIGN_LANDMARKS], 
            anchor_landmarks[ALIGN_LANDMARKS],
            method=cv2.RANSAC
        )
        stabilized_frame = cv2.warpAffine(frame, M, (anchor_image.shape[1], anchor_image.shape[0]))
        
        # --- NEW: Crop and compress the stabilized frame ---
        compressed_frame = crop_and_compress(stabilized_frame, bbox)
        stabilized_blink_frames.append(compressed_frame)

    cap.release()
    
    if not stabilized_blink_frames:
        raise ValueError("Could not extract any blink frames.")

    print(f"Extracted {len(stabilized_blink_frames)} total frames.")
    
    # --- Build Blink Cycle Library (with compressed JPEGs) ---
    peak_index = peak_frame - start_frame
    if peak_index < 0 or peak_index >= len(stabilized_blink_frames):
        peak_index = len(stabilized_blink_frames) // 2
        if peak_index == 0: peak_index = 1

    closed_eye_jpeg = stabilized_blink_frames[peak_index]
    open_to_close_jpegs = stabilized_blink_frames[:peak_index]
    close_to_open_jpegs = stabilized_blink_frames[peak_index + 1:]

    blink_cycles_jpeg = {}
    blink_cycles_jpeg['normal'] = [anchor_eye_jpeg] + open_to_close_jpegs + [closed_eye_jpeg] + close_to_open_jpegs
    blink_cycles_jpeg['fast'] = [anchor_eye_jpeg] + open_to_close_jpegs[::2] + [closed_eye_jpeg] + close_to_open_jpegs[::2]
    hold_frames = [closed_eye_jpeg] * 8 
    blink_cycles_jpeg['slow'] = [anchor_eye_jpeg] + open_to_close_jpegs + hold_frames + close_to_open_jpegs

    print(f"\nSaving compressed assets to {output_path}...")
    
    np.savez_compressed(
        output_path,
        anchor_landmarks=anchor_landmarks,
        cropped_alpha_mask=cropped_alpha_mask, # Save the small mask
        bbox=np.array(bbox),                # Save the crop coordinates
        anchor_eye_jpeg=anchor_eye_jpeg,    # Save the compressed anchor
        normal=np.array(blink_cycles_jpeg['normal'], dtype=object),
        fast=np.array(blink_cycles_jpeg['fast'], dtype=object),
        slow=np.array(blink_cycles_jpeg['slow'], dtype=object)
    )
    
    print("--- [ASSET CREATION COMPLETE] ---")
    print(f"Compressed blink library file created: {output_path}")

# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a compressed blink asset library.")
    parser.add_argument("--source_image", required=True, help="Path to the original static portrait image.")
    parser.add_argument("--blink_video", required=True, help="Path to the video with blinks.")
    parser.add_argument("--output_file", required=True, help="Path to save the final asset file (e.g., 'actor_blinks.npz').")
    parser.add_argument("--start_time", required=True, type=float, help="Start time (in seconds) of the 'open-to-close' motion.")
    parser.add_argument("--peak_time", required=True, type=float, help="Time (in seconds) when the eye is *fully* closed.")
    parser.add_argument("--end_time", required=True, type=float, help="Time (in seconds) when the eye is *fully* open again.")
    parser.add_argument("--dlib_model", required=True, help="Path to 'shape_predictor_68_face_landmarks.dat' file.")
    
    args = parser.parse_args()
    
    create_blink_assets(
        args.blink_video, args.source_image, 
        args.start_time, args.peak_time, args.end_time,
        args.dlib_model, args.output_file
    )