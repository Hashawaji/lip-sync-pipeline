"""
SyncNet Evaluator

Evaluates lip-sync quality using SyncNet embeddings and simple heuristics.

Metrics computed:
- LSE-D: Lip Sync Error - Distance (lower is better)
- LSE-C: Lip Sync Error - Confidence (higher is better)
- AV Offset: Estimated audio-visual misalignment in frames
- AV Correlation: Correlation between mouth motion and audio energy
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from .model import SyncNetModel


class SyncNetEvaluator:
    """
    Evaluates lip-sync using the official SyncNet model.
    
    Computes:
    - LSE-D (Lip Sync Error - Distance): Euclidean distance between embeddings
    - LSE-C (Lip Sync Error - Confidence): Cosine similarity confidence
    
    Usage:
        evaluator = SyncNetEvaluator()
        results = evaluator.evaluate("video.mp4")
        print(f"LSE-C: {results['lse_c']:.3f}")
    """
    
    WEIGHTS_URL = "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model"
    
    def __init__(self, 
                 weights_path: Optional[str] = None,
                 device: str = 'auto'):
        """
        Initialize SyncNet evaluator.
        
        Args:
            weights_path: Path to syncnet_v2.model. Auto-downloads if not found.
            device: 'auto', 'cuda', 'mps', or 'cpu'
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for SyncNet evaluation")
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for audio feature extraction")
        
        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = SyncNetModel().to(self.device)
        
        # Find or download weights
        if weights_path is None:
            weights_path = self._get_default_weights_path()
        
        if not os.path.exists(weights_path):
            self._download_weights(weights_path)
        
        self._load_weights(weights_path)
        self.model.eval()
    
    def _get_default_weights_path(self) -> str:
        """Get default path for model weights."""
        module_dir = Path(__file__).parent
        return str(module_dir / "models" / "syncnet_v2.model")
    
    def _download_weights(self, path: str):
        """Download SyncNet weights from Oxford VGG."""
        import urllib.request
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading SyncNet weights to {path}...")
        urllib.request.urlretrieve(self.WEIGHTS_URL, path)
        print("Download complete!")
    
    def _load_weights(self, path: str):
        """Load pretrained weights."""
        print(f"Loading weights from {path}")
        
        # Oxford weights are Python 2 pickle, need encoding fix
        checkpoint = torch.load(path, map_location=self.device, 
                                weights_only=False, encoding='latin1')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            # Assume it's a direct state dict
            state_dict = checkpoint
        
        # Try to load, show warning if architecture mismatch
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            warnings.warn(f"Strict loading failed, trying non-strict: {e}")
            self.model.load_state_dict(state_dict, strict=False)
        
        print("✓ Weights loaded successfully")
    
    def extract_mouth_crops(self, 
                           video_path: str,
                           crop_size: int = 224) -> np.ndarray:
        """
        Extract mouth region crops from video.
        
        Args:
            video_path: Path to video file
            crop_size: Size of mouth crop (224x224 for SyncNet v2)
        
        Returns:
            Array of shape (N, 3, H, W) - RGB mouth crops in CHW format
        """
        cap = cv2.VideoCapture(video_path)
        crops = []
        
        # Try to use face detection, fallback to center crop
        try:
            import dlib
            detector = dlib.get_frontal_face_detector()
            predictor_path = self._find_shape_predictor()
            if predictor_path:
                predictor = dlib.shape_predictor(predictor_path)
                use_landmarks = True
            else:
                use_landmarks = False
        except ImportError:
            use_landmarks = False
            warnings.warn("dlib not available, using center crop for face/mouth region")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if use_landmarks:
                crop = self._extract_face_with_landmarks(frame_rgb, gray, detector, predictor, crop_size)
            else:
                crop = self._extract_face_center(frame_rgb, crop_size)
            
            if crop is not None:
                # Convert HWC to CHW format
                crop = crop.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
                crops.append(crop)
            
            frame_count += 1
        
        cap.release()
        
        if len(crops) == 0:
            raise ValueError(f"No frames extracted from {video_path}")
        
        return np.array(crops, dtype=np.float32)
    
    def _find_shape_predictor(self) -> Optional[str]:
        """Find dlib shape predictor file."""
        common_paths = [
            "shape_predictor_68_face_landmarks.dat",
            os.path.expanduser("~/.dlib/shape_predictor_68_face_landmarks.dat"),
            "/usr/share/dlib/shape_predictor_68_face_landmarks.dat",
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _extract_face_with_landmarks(self, frame_rgb, gray, detector, predictor, crop_size):
        """Extract lower face region using dlib landmarks."""
        faces = detector(gray, 1)
        if len(faces) == 0:
            return self._extract_face_center(frame_rgb, crop_size)
        
        shape = predictor(gray, faces[0])
        
        # Get face bounds from landmarks (focus on lower face for lip sync)
        # Use chin (8) and sides of face (0-16) for bounds
        face_points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        face_points = np.array(face_points)
        
        # Get bounding box of lower face (nose tip to chin)
        # Landmarks 27-35 are nose, 48-67 are mouth, 0-16 are jaw
        lower_face_y = int(np.mean([shape.part(i).y for i in [27, 30]]))
        chin_y = shape.part(8).y
        
        x_min = min(shape.part(i).x for i in range(17))
        x_max = max(shape.part(i).x for i in range(17))
        
        # Add padding
        h, w = frame_rgb.shape[:2]
        pad_x = int((x_max - x_min) * 0.1)
        pad_y = int((chin_y - lower_face_y) * 0.1)
        
        x1 = max(0, x_min - pad_x)
        x2 = min(w, x_max + pad_x)
        y1 = max(0, lower_face_y - pad_y)
        y2 = min(h, chin_y + pad_y)
        
        crop = frame_rgb[y1:y2, x1:x2]
        
        if crop.size == 0:
            return self._extract_face_center(frame_rgb, crop_size)
        
        return cv2.resize(crop, (crop_size, crop_size))
    
    def _extract_face_center(self, frame_rgb, crop_size):
        """
        Extract face region from center of frame.
        
        SyncNet expects the FULL FACE (not just lips) for computing 
        audio-visual embeddings. The model uses 3D convolutions over
        5 consecutive frames to capture lip motion.
        """
        h, w = frame_rgb.shape[:2]
        
        # For portrait videos where face fills frame:
        # Crop a square region centered on the face
        # Face is typically in upper 2/3 of portrait frames
        
        # Center horizontally, position vertically based on aspect ratio
        cx = w // 2
        
        if h > w:
            # Tall portrait (like 1024x1536) - face is usually upper portion
            cy = int(h * 0.35)  # Face center around 35% from top
        else:
            # Square or wide - face centered
            cy = int(h * 0.45)
        
        # Crop size should capture the full face
        # Use smaller dimension to ensure square crop
        crop_size_orig = min(w, h)
        
        # Make crop slightly smaller than full width to focus on face
        half = int(crop_size_orig * 0.45)
        
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)
        y1 = max(0, cy - half)
        y2 = min(h, cy + half)
        
        # Ensure square
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w > crop_h:
            diff = crop_w - crop_h
            x1 += diff // 2
            x2 -= diff // 2
        elif crop_h > crop_w:
            diff = crop_h - crop_w
            y1 += diff // 2
            y2 -= diff // 2
        
        crop = frame_rgb[y1:y2, x1:x2]
        
        if crop.size == 0:
            # Fallback to center crop
            half = min(w, h) // 2
            x1 = max(0, cx - half)
            x2 = min(w, cx + half)
            y1 = max(0, cy - half)
            y2 = min(h, cy + half)
            crop = frame_rgb[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
        
        return cv2.resize(crop, (crop_size, crop_size))
    
    def extract_mfcc(self, 
                    audio_path: str,
                    video_fps: float = 25.0) -> np.ndarray:
        """
        Extract MFCC features aligned to video frames.
        
        Args:
            audio_path: Path to audio file (or video with audio)
            video_fps: Video framerate for alignment
        
        Returns:
            Array of shape (N, 13, 20) - MFCC features per frame
        """
        import tempfile
        import subprocess
        
        # If input is video, extract audio first using ffmpeg
        if audio_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_path = temp_audio.name
            
            try:
                # Extract audio using ffmpeg
                cmd = [
                    'ffmpeg', '-y', '-i', audio_path,
                    '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    temp_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise ValueError(f"Video has no audio track or ffmpeg failed: {result.stderr}")
                y, sr = librosa.load(temp_path, sr=16000)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        else:
            # Load audio directly
            y, sr = librosa.load(audio_path, sr=16000)
        
        # Compute MFCC
        hop_length = int(sr / video_fps)  # One MFCC frame per video frame
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, 
                                     hop_length=hop_length,
                                     n_fft=2048)
        
        # mfcc shape: (13, T)
        # Need to create windows of 20 frames
        window_size = 20
        n_frames = mfcc.shape[1] - window_size + 1
        
        mfcc_windows = []
        for i in range(n_frames):
            window = mfcc[:, i:i+window_size]  # (13, 20)
            mfcc_windows.append(window)
        
        return np.array(mfcc_windows)  # (N, 13, 20)
    
    def evaluate(self, 
                video_path: str,
                audio_path: Optional[str] = None) -> Dict:
        """
        Evaluate lip-sync quality of a video.
        
        Args:
            video_path: Path to video file
            audio_path: Optional separate audio file (uses video audio if None)
        
        Returns:
            Dictionary with evaluation metrics
        """
        if audio_path is None:
            audio_path = video_path
        
        # Get video FPS
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        print(f"Extracting mouth crops from {video_path}...")
        mouth_crops = self.extract_mouth_crops(video_path)
        
        print(f"Extracting MFCC from {audio_path}...")
        mfcc = self.extract_mfcc(audio_path, fps)
        
        # Align lengths
        min_len = min(len(mouth_crops) - 4, len(mfcc))  # Need 5 frames for SyncNet
        
        if min_len <= 0:
            raise ValueError("Video too short for SyncNet evaluation")
        
        print(f"Computing embeddings for {min_len} frame windows...")
        
        # Process in batches
        batch_size = 16  # Smaller batch due to 224x224 input size
        face_embeddings = []
        audio_embeddings = []
        
        with torch.no_grad():
            for i in range(0, min_len, batch_size):
                end_idx = min(i + batch_size, min_len)
                
                # Prepare face batch (5 consecutive frames)
                # mouth_crops: (N, 3, H, W)
                # Need: (B, 3, 5, H, W) for 3D conv
                face_batch = []
                for j in range(i, end_idx):
                    frames = mouth_crops[j:j+5]  # (5, 3, H, W)
                    # Transpose to (3, 5, H, W)
                    frames = np.transpose(frames, (1, 0, 2, 3))
                    face_batch.append(frames)
                
                face_tensor = torch.FloatTensor(np.array(face_batch))
                face_tensor = face_tensor.to(self.device)
                
                # Normalize to [-1, 1]
                face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
                
                # Prepare audio batch
                audio_batch = mfcc[i:end_idx]
                audio_tensor = torch.FloatTensor(audio_batch)
                audio_tensor = audio_tensor.unsqueeze(1)  # Add channel dim: (B, 1, 13, 20)
                audio_tensor = audio_tensor.to(self.device)
                
                # Forward pass
                face_emb = self.model.forward_face(face_tensor)
                audio_emb = self.model.forward_audio(audio_tensor)
                
                face_embeddings.append(face_emb.cpu().numpy())
                audio_embeddings.append(audio_emb.cpu().numpy())
        
        face_embeddings = np.vstack(face_embeddings)
        audio_embeddings = np.vstack(audio_embeddings)
        
        # Compute metrics
        face_tensor = torch.FloatTensor(face_embeddings)
        audio_tensor = torch.FloatTensor(audio_embeddings)
        
        # LSE-D (distance)
        distances = SyncNetModel.compute_distance(face_tensor, audio_tensor)
        lse_d = distances.mean().item()
        
        # LSE-C (confidence/similarity)
        similarities = SyncNetModel.compute_similarity(face_tensor, audio_tensor)
        lse_c = similarities.mean().item()
        
        # Compute offset using cross-correlation
        offset = self._compute_offset(face_embeddings, audio_embeddings)
        
        results = {
            'lse_d': lse_d,
            'lse_c': lse_c,
            'offset_frames': offset,
            'offset_ms': offset * (1000 / fps),
            'n_windows': len(face_embeddings),
            'video_fps': fps,
            'quality_rating': self._rate_quality(lse_c),
        }
        
        self._print_results(results)
        return results
    
    def _compute_offset(self, 
                       face_emb: np.ndarray, 
                       audio_emb: np.ndarray,
                       max_offset: int = 60) -> int:  # Increased from 15 to detect larger offsets
        """Compute optimal A/V offset using cross-correlation."""
        # Compute similarity at different offsets
        best_offset = 0
        best_sim = -float('inf')
        
        for offset in range(-max_offset, max_offset + 1):
            if offset < 0:
                f = face_emb[-offset:]
                a = audio_emb[:offset]
            elif offset > 0:
                f = face_emb[:-offset]
                a = audio_emb[offset:]
            else:
                f = face_emb
                a = audio_emb
            
            if len(f) == 0 or len(a) == 0:
                continue
            
            min_len = min(len(f), len(a))
            f = f[:min_len]
            a = a[:min_len]
            
            # Cosine similarity
            f_norm = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
            a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
            sim = np.mean(np.sum(f_norm * a_norm, axis=1))
            
            if sim > best_sim:
                best_sim = sim
                best_offset = offset
        
        return best_offset
    
    def _rate_quality(self, lse_c: float) -> str:
        """Rate sync quality based on LSE-C score."""
        if lse_c >= 0.8:
            return "Excellent"
        elif lse_c >= 0.6:
            return "Good"
        elif lse_c >= 0.4:
            return "Fair"
        elif lse_c >= 0.2:
            return "Poor"
        else:
            return "Very Poor"
    
    def _print_results(self, results: Dict):
        """Print formatted results."""
        print("\n" + "="*50)
        print("SyncNet Evaluation Results")
        print("="*50)
        print(f"LSE-D (Distance):     {results['lse_d']:.4f}  (lower is better)")
        print(f"LSE-C (Confidence):   {results['lse_c']:.4f}  (higher is better)")
        print(f"A/V Offset:           {results['offset_frames']} frames ({results['offset_ms']:.1f}ms)")
        print(f"Quality Rating:       {results['quality_rating']}")
        print(f"Analyzed Windows:     {results['n_windows']}")
        print("="*50 + "\n")
    
    def compare_videos(self,
                      video_paths: List[str],
                      labels: Optional[List[str]] = None) -> Dict:
        """
        Compare multiple videos.
        
        Returns comparison table and rankings.
        """
        if labels is None:
            labels = [Path(p).stem for p in video_paths]
        
        results = {}
        for path, label in zip(video_paths, labels):
            print(f"\nEvaluating: {label}")
            results[label] = self.evaluate(path)
        
        # Print comparison
        print("\n" + "="*70)
        print("COMPARISON TABLE")
        print("="*70)
        print(f"{'Video':<25} {'LSE-D':>10} {'LSE-C':>10} {'Offset':>10} {'Quality':<12}")
        print("-"*70)
        
        for label in labels:
            r = results[label]
            print(f"{label:<25} {r['lse_d']:>10.4f} {r['lse_c']:>10.4f} "
                  f"{r['offset_frames']:>8}fr {r['quality_rating']:<12}")
        
        print("="*70)
        
        return results


class SimpleAVEvaluator:
    """
    Simple audio-visual correlation evaluator.
    
    Does NOT require pre-trained weights. Uses:
    - Mouth motion (optical flow or landmarks)
    - Audio energy envelope
    - Correlation between the two
    
    Good for quick sanity checks and when SyncNet weights unavailable.
    """
    
    def __init__(self):
        """Initialize simple evaluator."""
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for audio analysis")
    
    def evaluate(self, video_path: str) -> Dict:
        """
        Evaluate using simple audio-visual correlation.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with metrics
        """
        # Extract audio energy
        audio_energy = self._extract_audio_energy(video_path)
        
        # Extract mouth motion
        mouth_motion, fps = self._extract_mouth_motion(video_path)
        
        # Align lengths
        min_len = min(len(audio_energy), len(mouth_motion))
        audio_energy = audio_energy[:min_len]
        mouth_motion = mouth_motion[:min_len]
        
        # Compute correlation
        correlation = np.corrcoef(audio_energy, mouth_motion)[0, 1]
        
        # Compute correlation at different offsets
        best_offset, best_corr = self._find_best_offset(audio_energy, mouth_motion)
        
        results = {
            'av_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'av_correlation_best': float(best_corr) if not np.isnan(best_corr) else 0.0,
            'sync_offset_frames': int(best_offset),
            'n_frames': min_len,
            'quality': self._rate_quality(best_corr),
            'audio_energy_stats': {
                'mean': float(np.mean(audio_energy)),
                'std': float(np.std(audio_energy)),
                'max': float(np.max(audio_energy)),
            },
            'mouth_movement_stats': {
                'mean': float(np.mean(mouth_motion)),
                'std': float(np.std(mouth_motion)),
                'max': float(np.max(mouth_motion)),
            },
        }
        
        self._print_results(results)
        return results
    
    def _extract_audio_energy(self, video_path: str) -> np.ndarray:
        """Extract audio energy envelope."""
        y, sr = librosa.load(video_path, sr=16000)
        
        # Get video FPS for alignment
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Compute RMS energy per video frame
        hop_length = int(sr / fps)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        return rms
    
    def _extract_mouth_motion(self, video_path: str) -> Tuple[np.ndarray, float]:
        """Extract mouth motion using optical flow on mouth region."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        motions = []
        prev_gray = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Focus on mouth region (center-bottom of frame)
            h, w = gray.shape
            y1 = int(h * 0.6)
            y2 = int(h * 0.9)
            x1 = int(w * 0.3)
            x2 = int(w * 0.7)
            mouth_region = gray[y1:y2, x1:x2]
            
            if prev_gray is not None:
                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, mouth_region, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                
                # Magnitude of motion
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motions.append(np.mean(magnitude))
            else:
                motions.append(0.0)
            
            prev_gray = mouth_region.copy()
        
        cap.release()
        return np.array(motions), fps
    
    def _find_best_offset(self, 
                         audio: np.ndarray, 
                         motion: np.ndarray,
                         max_offset: int = 15) -> Tuple[int, float]:
        """Find offset with best correlation."""
        best_offset = 0
        best_corr = -1
        
        for offset in range(-max_offset, max_offset + 1):
            if offset < 0:
                a = audio[-offset:]
                m = motion[:offset]
            elif offset > 0:
                a = audio[:-offset]
                m = motion[offset:]
            else:
                a = audio
                m = motion
            
            min_len = min(len(a), len(m))
            if min_len < 10:
                continue
                
            corr = np.corrcoef(a[:min_len], m[:min_len])[0, 1]
            
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_offset = offset
        
        return best_offset, best_corr
    
    def _rate_quality(self, correlation: float) -> str:
        """Rate quality based on A/V correlation."""
        if np.isnan(correlation):
            return "Unknown"
        if correlation >= 0.6:
            return "Excellent"
        elif correlation >= 0.4:
            return "Good"
        elif correlation >= 0.2:
            return "Fair"
        elif correlation >= 0.0:
            return "Poor"
        else:
            return "Very Poor"
    
    def _print_results(self, results: Dict):
        """Print formatted results."""
        print("\n" + "="*50)
        print("Simple A/V Correlation Results")
        print("="*50)
        print(f"A/V Correlation:      {results['av_correlation']:.4f}")
        print(f"Best Correlation:     {results['av_correlation_best']:.4f}")
        print(f"Optimal Offset:       {results['sync_offset_frames']} frames")
        print(f"Quality Rating:       {results['quality']}")
        print(f"Frames Analyzed:      {results['n_frames']}")
        print("="*50 + "\n")
    
    def compare_videos(self,
                      video_paths: List[str],
                      labels: Optional[List[str]] = None) -> Dict:
        """Compare multiple videos."""
        if labels is None:
            labels = [Path(p).stem for p in video_paths]
        
        results = {}
        for path, label in zip(video_paths, labels):
            print(f"\n--- Evaluating: {label} ---")
            results[label] = self.evaluate(path)
        
        # Print comparison
        print("\n" + "="*70)
        print("COMPARISON TABLE (Simple Evaluator)")
        print("="*70)
        print(f"{'Video':<25} {'Correlation':>12} {'BestCorr':>12} {'Offset':>10} {'Quality':<10}")
        print("-"*70)
        
        for label in labels:
            r = results[label]
            print(f"{label:<25} {r['av_correlation']:>12.4f} {r['av_correlation_best']:>12.4f} "
                  f"{r['sync_offset_frames']:>8}fr {r['quality']:<10}")
        
        print("="*70)
        
        return results
