"""
SyncNet Evaluation Module

Provides audio-visual synchronization metrics for lip-sync evaluation.
Uses the official Oxford VGG SyncNet model.

Metrics:
- LSE-D (Lip Sync Error - Distance): Euclidean distance between A/V embeddings
- LSE-C (Lip Sync Error - Confidence): Cosine similarity confidence score
- Offset: Estimated audio-visual offset in frames
"""

from .evaluator import SyncNetEvaluator, SimpleAVEvaluator
from .model import SyncNetModel

__all__ = ['SyncNetEvaluator', 'SimpleAVEvaluator', 'SyncNetModel']
