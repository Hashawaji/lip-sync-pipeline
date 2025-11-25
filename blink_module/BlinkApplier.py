import cv2
import numpy as np
import dlib
import random
import os
import sys
import subprocess
import tempfile

class BlinkApplier:
    """
    A class to apply pre-generated eye blinks to a lip-sync video.

    This class loads all necessary models and blink assets once upon
    initialization, making it highly efficient for repeated use in an
    application like Streamlit.

    It handles both dynamic head (real-time landmark detection) and
    static head (pre-warped, high-speed) compositing.

    It also automatically muxes the audio from the original lip-sync
    video into the final output video using FFmpeg.
    """
    
    # --- Constants ---
    _ALIGN_LANDMARKS = list(range(17, 68)) 

    def __init__(self, dlib_model_path, blink_assets_path):
        """
        Initializes the BlinkApplier by loading all necessary models and assets.

        Args:
            dlib_model_path (str): Path to 'shape_predictor_68_face_landmarks.dat'.
            blink_assets_path (str): Path to the actor's '.npz' blink asset file.
        """
        print("[BlinkApplier] Initializing...")
        
        # 1. Load dlib models
        if not os.path.exists(dlib_model_path):
            raise FileNotFoundError(f"Dlib model not found at: {dlib_model_path}")
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(dlib_model_path)
            print("[BlinkApplier] Dlib models loaded.")
        except Exception as e:
            raise IOError(f"Error loading dlib model: {e}")

        # 2. Load Compressed Blink Asset Library
        print(f"[BlinkApplier] Loading compressed assets from: {blink_assets_path}")
        if not os.path.exists(blink_assets_path):
            raise FileNotFoundError(f"Blink assets not found at: {blink_assets_path}")
        try:
            assets = np.load(blink_assets_path, allow_pickle=True)
            self.anchor_landmarks = assets['anchor_landmarks']
            self.cropped_mask = assets['cropped_alpha_mask']
            self.bbox = assets['bbox'] # [x, y, w, h]
            
            # Decode all JPEG-compressed frames into memory
            print("[BlinkApplier] Decoding blink cycles...")
            self.anchor_eye_patch = self._decode_jpeg(assets['anchor_eye_jpeg'])
            self.blink_cycles_decoded = {
                'normal': [self._decode_jpeg(f) for f in assets['normal']],
                'fast':   [self._decode_jpeg(f) for f in assets['fast']],
                'slow':   [self._decode_jpeg(f) for f in assets['slow']]
            }
            print(f"  - Cropped mask shape: {self.cropped_mask.shape}")
            print(f"  - Decoded {len(self.blink_cycles_decoded['normal'])} 'normal' frames.")
            
        except Exception as e:
            raise IOError(f"Error loading blink asset file '{blink_assets_path}': {e}")
            
        print("[BlinkApplier] Initialization complete.")
        
    def _decode_jpeg(self, jpeg_bytes):
        """Helper to decode JPEG bytes to a BGR numpy array."""
        return cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR).astype(np.float32)

    # --- New Method: Apply Blinks to Frames in Memory ---
    def apply_blinks_to_frames(self, frames, blink_start_frames, fps, static_head=True):
        """
        Apply blinks directly to a list of frames in memory (NO video I/O).
        This is much more efficient than processing a video file.
        
        Args:
            frames: List of numpy arrays (BGR frames)
            blink_start_frames: Set of frame numbers where blinks should start
            fps: Frames per second (for progress tracking)
            static_head: If True, use pre-warped optimization
            
        Returns:
            List of frames with blinks applied
        """
        if not blink_start_frames or len(blink_start_frames) == 0:
            print("[BlinkApplier] No blinks to apply")
            return frames
        
        print(f"[BlinkApplier] Applying blinks to {len(frames)} frames...")
        
        # Pre-warp blink assets if static head
        pre_warped_cache = {}
        pre_warped_mask = None
        inv_pre_warped_mask = None
        dest_bbox = None
        
        if static_head and len(frames) > 0:
            first_frame = frames[0].astype(np.float32) if frames[0].dtype == np.uint8 else frames[0]
            first_frame_landmarks = self._get_landmarks(first_frame.astype(np.uint8))
            
            if first_frame_landmarks is not None:
                M_full, _ = cv2.estimateAffinePartial2D(
                    self.anchor_landmarks[self._ALIGN_LANDMARKS],
                    first_frame_landmarks[self._ALIGN_LANDMARKS],
                    method=cv2.RANSAC
                )
                
                (pre_warped_cache, 
                 pre_warped_mask, 
                 inv_pre_warped_mask, 
                 dest_bbox) = self._pre_warp_blink_cycles(M_full, first_frame)
                
                print(f"[BlinkApplier] Pre-warped blink assets for static head")
            else:
                print("[BlinkApplier] Warning: No face detected in first frame, using dynamic mode")
                static_head = False
        
        # Apply blinks frame by frame
        width, height = frames[0].shape[1], frames[0].shape[0]
        current_blink_frames = None
        blink_frame_idx = 0
        frames_modified = 0
        
        for frame_num in range(len(frames)):
            # Check if starting a new blink
            if current_blink_frames is None and frame_num in blink_start_frames:
                cycle_name = random.choice(list(self.blink_cycles_decoded.keys()))
                current_blink_frames = pre_warped_cache[cycle_name] if static_head else self.blink_cycles_decoded[cycle_name]
                blink_frame_idx = 0
            
            # Apply blink if in progress
            if current_blink_frames is not None:
                patch_to_apply = current_blink_frames[blink_frame_idx]
                dest_frame = frames[frame_num].astype(np.float32) if frames[frame_num].dtype == np.uint8 else frames[frame_num]
                
                if static_head and pre_warped_mask is not None and inv_pre_warped_mask is not None and dest_bbox is not None:
                    # Optimized static path
                    x, y, w, h = dest_bbox
                    if x + w <= width and y + h <= height and x >= 0 and y >= 0:
                        roi = dest_frame[y:y+h, x:x+w]
                        blended_roi = (patch_to_apply * pre_warped_mask) + (roi * inv_pre_warped_mask)
                        dest_frame[y:y+h, x:x+w] = blended_roi
                        frames[frame_num] = np.clip(dest_frame, 0, 255).astype(np.uint8)
                        frames_modified += 1
                else:
                    # Dynamic path (real-time landmark detection)
                    dest_landmarks = self._get_landmarks(dest_frame.astype(np.uint8))
                    if dest_landmarks is not None:
                        M_full, _ = cv2.estimateAffinePartial2D(
                            self.anchor_landmarks[self._ALIGN_LANDMARKS],
                            dest_landmarks[self._ALIGN_LANDMARKS],
                            method=cv2.RANSAC
                        )
                        warped_patch, warped_mask, inv_warped_mask, bbox = self._warp_patch(
                            M_full, patch_to_apply, self.cropped_mask, dest_frame
                        )
                        x, y, w, h = bbox
                        if x + w <= width and y + h <= height and x >= 0 and y >= 0:
                            roi = dest_frame[y:y+h, x:x+w]
                            blended_roi = (warped_patch * warped_mask) + (roi * inv_warped_mask)
                            dest_frame[y:y+h, x:x+w] = blended_roi
                            frames[frame_num] = np.clip(dest_frame, 0, 255).astype(np.uint8)
                            frames_modified += 1
                
                # Advance blink sequence
                blink_frame_idx += 1
                if blink_frame_idx >= len(current_blink_frames):
                    current_blink_frames = None
        
        print(f"[BlinkApplier] Modified {frames_modified} frames with blinks")
        return frames

    # --- Public Main Method ---
    def process_video(self, lip_video_path, output_video_path, 
                        blink_start_frames, duration=None, static_head=True, 
                        status_callback=None):
        """
        Apply pre-generated eye blinks to a lip-sync video (OPTIMIZED VERSION).
        
        Only processes frames where blinks occur. If no blinks are scheduled,
        simply copies the video with audio re-muxing.
        
        Args:
            lip_video_path (str): Path to input video with lip-sync animation.
            output_video_path (str): Path to save final video with blinks and audio.
            blink_start_frames (set): Set of frame numbers where blinks should start.
                Use BlinkScheduler to generate this based on phoneme pauses.
            duration (float, optional): Max duration in seconds to process. None = full video.
            static_head (bool): If True, use optimized pre-warping for static heads.
                If False, detect landmarks per frame (slower, for moving heads).
            status_callback (callable, optional): Function to call with status messages.
        """
        
        def _status(message):
            print(message)
            if status_callback: status_callback(message)

        # === OPTIMIZATION: Quick exit if no blinks scheduled ===
        if not blink_start_frames or len(blink_start_frames) == 0:
            _status("[BlinkApplier] No blinks scheduled. Copying video with audio re-mux...")
            # try:
            #     self._mux_audio(lip_video_path, lip_video_path, output_video_path)
            #     _status(f"Video copied with audio: {output_video_path}")
            #     return
            # except Exception as e:
            #     _status(f"Error during copy: {e}")
            #     raise
            return

        # === Calculate which frames need processing ===
        # Determine blink sequence length (use max cycle length for safety)
        max_blink_length = max(
            len(self.blink_cycles_decoded['normal']),
            len(self.blink_cycles_decoded['fast']),
            len(self.blink_cycles_decoded['slow'])
        )
        
        # Build set of ALL frames that need processing (blink start + sequence frames)
        frames_to_process = set()
        for start_frame in blink_start_frames:
            for offset in range(max_blink_length):
                frames_to_process.add(start_frame + offset)
        
        _status(f"[BlinkApplier] {len(blink_start_frames)} blinks scheduled affecting {len(frames_to_process)} frames total")

        # Define cap_in and out with default values for the finally block
        cap_in = None
        out = None

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
            silent_video_path = temp_video_file.name

        try:
            _status("[BlinkApplier] Starting optimized video synthesis...")
            cap_in = cv2.VideoCapture(lip_video_path)
            if not cap_in.isOpened():
                raise IOError(f"Could not open lip_video: {lip_video_path}")

            width = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap_in.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))
            
            max_frames_to_process = total_frames
            if duration is not None and duration > 0:
                calc_frames = int(duration * fps)
                if calc_frames < total_frames:
                    max_frames_to_process = calc_frames

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(silent_video_path, fourcc, fps, (width, height))
            
            _status(f"Output: {width}x{height} @ {fps:.2f} FPS. Total video frames: {total_frames}")
            _status(f"Optimization: Only processing {len(frames_to_process)} frames (skipping {total_frames - len(frames_to_process)} frames)")

            # --- Pre-computation ---
            pre_warped_cache = {}
            pre_warped_mask = None
            inv_pre_warped_mask = None
            dest_bbox = None # [x, y, w, h]
            
            if static_head:
                _status("Static head mode enabled. Pre-warping blink assets...")
                # Read first frame for pre-warping
                cap_in.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, first_frame = cap_in.read()
                if not ret: raise IOError("Cannot read first frame.")
                    
                first_frame_landmarks = self._get_landmarks(first_frame)
                if first_frame_landmarks is None:
                    raise RuntimeError("No face in first frame. Cannot use static_head=True.")
                    
                M_full, _ = cv2.estimateAffinePartial2D(
                    self.anchor_landmarks[self._ALIGN_LANDMARKS],
                    first_frame_landmarks[self._ALIGN_LANDMARKS],
                    method=cv2.RANSAC
                )
                
                # Pre-warp only the blink cycles (no anchor patch)
                (pre_warped_cache, 
                 pre_warped_mask, 
                 inv_pre_warped_mask, 
                 dest_bbox) = self._pre_warp_blink_cycles(M_full, first_frame.astype(np.float32))

                _status(f"  - Pre-warp complete. Target patch location: {dest_bbox}")
            
            else:
                _status("Dynamic head mode enabled. Processing frame-by-frame.")

            # === OPTIMIZED Main Loop: Only process frames with blinks ===
            cap_in.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
            frame_num = 0
            current_blink_frames = None
            blink_frame_idx = 0
            frames_processed = 0
            
            while frame_num < max_frames_to_process:
                # Check if this frame needs processing
                needs_processing = frame_num in frames_to_process
                
                if needs_processing:
                    # Read and process this frame
                    ret, dest_frame = cap_in.read()
                    if not ret: break
                    
                    # Progress logging
                    if frames_processed % 50 == 0:
                        _status(f"  Processing blink frame {frames_processed}/{len(frames_to_process)} (video frame {frame_num}/{max_frames_to_process})")
                    
                    # Check if starting a new blink
                    if current_blink_frames is None and frame_num in blink_start_frames:
                        cycle_name = random.choice(list(self.blink_cycles_decoded.keys()))
                        current_blink_frames = pre_warped_cache[cycle_name] if static_head else self.blink_cycles_decoded[cycle_name]
                        blink_frame_idx = 0
                    
                    # Apply blink if in progress
                    if current_blink_frames is not None:
                        patch_to_apply = current_blink_frames[blink_frame_idx]
                        
                        if static_head and pre_warped_mask is not None and inv_pre_warped_mask is not None and dest_bbox is not None:
                            # --- OPTIMIZED STATIC PATH ---
                            x, y, w, h = dest_bbox
                            if x + w <= width and y + h <= height and x >= 0 and y >= 0:
                                roi = dest_frame[y:y+h, x:x+w].astype(np.float32)
                                blended_roi = (patch_to_apply * pre_warped_mask) + (roi * inv_pre_warped_mask)
                                dest_frame[y:y+h, x:x+w] = np.clip(blended_roi, 0, 255).astype(np.uint8)
                        
                        else:
                            # --- DYNAMIC HEAD PATH ---
                            dest_landmarks = self._get_landmarks(dest_frame)
                            if dest_landmarks is not None:
                                M_full, _ = cv2.estimateAffinePartial2D(
                                    self.anchor_landmarks[self._ALIGN_LANDMARKS],
                                    dest_landmarks[self._ALIGN_LANDMARKS],
                                    method=cv2.RANSAC
                                )
                                warped_patch, warped_mask, inv_warped_mask, bbox = self._warp_patch(
                                    M_full, patch_to_apply, self.cropped_mask, dest_frame.astype(np.float32)
                                )
                                x, y, w, h = bbox
                                if x + w <= width and y + h <= height and x >= 0 and y >= 0:
                                    roi = dest_frame[y:y+h, x:x+w].astype(np.float32)
                                    blended_roi = (warped_patch * warped_mask) + (roi * inv_warped_mask)
                                    dest_frame[y:y+h, x:x+w] = np.clip(blended_roi, 0, 255).astype(np.uint8)
                        
                        # Advance blink sequence
                        blink_frame_idx += 1
                        if blink_frame_idx >= len(current_blink_frames):
                            current_blink_frames = None  # Blink finished
                    
                    out.write(dest_frame)
                    frames_processed += 1
                    
                else:
                    # Frame doesn't need processing - just copy it directly
                    ret, dest_frame = cap_in.read()
                    if not ret: break
                    out.write(dest_frame)
                
                frame_num += 1
            
            efficiency = (1 - len(frames_to_process) / max_frames_to_process) * 100 if max_frames_to_process > 0 else 0
            _status(f"Finished! Processed {frames_processed} blink frames, copied {frame_num - frames_processed} unchanged frames ({efficiency:.1f}% skip rate)")
            
        finally:
            # Safely release resources
            if cap_in:
                cap_in.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

        # --- Mux Audio with FFmpeg ---
        try:
            _status(f"Muxing audio from {lip_video_path}...")
            self._mux_audio(silent_video_path, lip_video_path, output_video_path)
            _status(f"Final video with audio saved to: {output_video_path}")
            
        except Exception as e:
            _status(f"Error during audio muxing: {e}")
            os.replace(silent_video_path, output_video_path)
            raise e
        finally:
            if os.path.exists(silent_video_path):
                os.remove(silent_video_path)
                
    # --- Internal Helper Methods ---

    def _get_landmarks(self, image):
        try:
            if image.ndim > 2: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else: gray = image
            rects = self.detector(gray, 1)
            if len(rects) == 0: return None
            shape = self.predictor(gray, rects[0])
            return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
        except Exception: return None

    def _correct_colors(self, src, dest, mask):
        try:
            bool_mask = (mask[:,:,0] > 0.5)
            if bool_mask.sum() < 100: return src
            for i in range(3):
                src_ch, dest_ch = src[:,:,i], dest[:,:,i]
                src_mean, src_std = cv2.meanStdDev(src_ch, mask=bool_mask.astype(np.uint8))
                dest_mean, dest_std = cv2.meanStdDev(dest_ch, mask=bool_mask.astype(np.uint8))
                src_mean, src_std = src_mean[0][0], src_std[0][0]
                dest_mean, dest_std = dest_mean[0][0], dest_std[0][0]
                if src_std < 1e-6: src_std = 1e-6
                a = dest_std / src_std
                b = dest_mean - a * src_mean
                src[:,:,i] = src_ch * a + b
            return src
        except Exception: return src

    # --- BUG FIX #2: Corrected patch transform logic ---
    def _get_patch_transform(self, M_full, img_shape):
        """
        Calculates the new (x,y,w,h) bounding box and the
        new 2x3 affine matrix to warp *just* the patch.
        """
        src_x, src_y, src_w, src_h = self.bbox
        
        # 1. Define the 3 corners of the *source crop*
        src_corners = np.array([
            [0, 0],
            [src_w, 0],
            [0, src_h]
        ], dtype=np.float32)
        
        # 2. Define the 3 corners of the *source crop* in *full image space*
        src_corners_full = np.array([
            [src_x, src_y],
            [src_x + src_w, src_y],
            [src_x, src_y + src_h]
        ], dtype=np.float32)

        # 3. Find where these 3 corners land in the *destination image*
        dest_corners_full = cv2.transform(src_corners_full.reshape(1, -1, 2), M_full).reshape(-1, 2)
        
        # 4. Find the new bounding box [x,y,w,h] for the destination patch
        x_min, y_min = np.min(dest_corners_full, axis=0)
        x_max, y_max = np.max(dest_corners_full, axis=0)
        
        dest_bbox = [
            int(x_min), 
            int(y_min), 
            int(np.ceil(x_max - x_min)), 
            int(np.ceil(y_max - y_min))
        ]
        
        # 5. Find the 3 corners relative to the new bounding box's origin
        dest_corners_local = dest_corners_full - (x_min, y_min)
        
        # 6. Calculate the affine matrix that maps src_corners -> dest_corners_local
        # This maps the original crop (0,0,w,h) to the new warped crop (0,0,w',h')
        M_crop = cv2.getAffineTransform(src_corners, dest_corners_local)
        
        # 7. Clamp destination bbox to image boundaries
        dest_bbox[0] = max(0, dest_bbox[0])
        dest_bbox[1] = max(0, dest_bbox[1])
        dest_bbox[2] = min(img_shape[1] - dest_bbox[0], dest_bbox[2])
        dest_bbox[3] = min(img_shape[0] - dest_bbox[1], dest_bbox[3])

        return M_crop, dest_bbox
    # ----------------------------------------------------

    def _warp_patch(self, M_full, patch, mask, dest_frame):
        """Warp and color-correct a single patch for dynamic mode."""
        M_crop, (x, y, w, h) = self._get_patch_transform(M_full, dest_frame.shape)
        
        # Get the destination ROI for color correction
        dest_roi = dest_frame[y:y+h, x:x+w]
        
        # Warp mask and patch
        warped_mask = cv2.warpAffine(mask, M_crop, (w, h))
        warped_patch = cv2.warpAffine(patch, M_crop, (w, h))
        
        # Color correct
        warped_patch = self._correct_colors(warped_patch, dest_roi, warped_mask)
        inv_warped_mask = 1.0 - warped_mask
        
        return warped_patch, warped_mask, inv_warped_mask, [x, y, w, h]

    def _pre_warp_blink_cycles(self, M_full, first_frame_float):
        """Pre-warps only blink cycle patches for static mode (no anchor patch)."""
        
        M_crop, (x, y, w, h) = self._get_patch_transform(M_full, first_frame_float.shape)
        dest_bbox = [x, y, w, h]
        
        # Get the destination ROI from the first frame for color correction
        dest_roi = first_frame_float[y:y+h, x:x+w]
        
        # Warp the mask ONCE
        pre_warped_mask = cv2.warpAffine(self.cropped_mask, M_crop, (w, h))
        inv_pre_warped_mask = 1.0 - pre_warped_mask

        # --- Pre-warp all BLINK cycles ---
        pre_warped_cache = {}
        for cycle_name, frames in self.blink_cycles_decoded.items():
            pre_warped_frames = []
            for patch_float in frames:
                warped_patch = cv2.warpAffine(patch_float, M_crop, (w, h))
                warped_patch = self._correct_colors(warped_patch, dest_roi, pre_warped_mask)
                pre_warped_frames.append(warped_patch)
            pre_warped_cache[cycle_name] = pre_warped_frames
            
        return pre_warped_cache, pre_warped_mask, inv_pre_warped_mask, dest_bbox

    def _mux_audio(self, video_input_path, audio_input_path, output_path):
        # Re-encode with browser-compatible H.264 settings and mux audio
        # H.264 High Profile Level 4.0, yuv420p pixel format, +faststart for streaming
        command = [
            'ffmpeg', '-y',
            '-i', video_input_path,
            '-i', audio_input_path,
            '-c:v', 'libx264',
            '-profile:v', 'high',
            '-level:v', '4.0',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-ar', '44100',
            '-ac', '2',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            '-movflags', '+faststart',
            output_path
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print("--- FFmpeg Error ---", e.stderr.decode())
            raise RuntimeError(f"FFmpeg failed: {e.stderr.decode()}")
        except FileNotFoundError:
            raise FileNotFoundError("FFmpeg not found. Please ensure it is in PATH.")