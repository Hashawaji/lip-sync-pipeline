"""
SyncNet Model Architecture

Based on the official SyncNet implementation:
https://github.com/joonson/syncnet_python

The model consists of two streams:
- Face encoder: Processes lip region crops using 3D convolutions
- Audio encoder: Processes MFCC features using 2D convolutions

Both produce 1024-dimensional embeddings for comparison.

Weight naming in checkpoint:
- netcnnlip: 3D CNN for lip/face (Conv3d layers)
- netfclip: FC layers for lip embedding
- netcnnaud: 2D CNN for audio/MFCC
- netfcaud: FC layers for audio embedding
"""

import torch
import torch.nn as nn


class SyncNetModel(nn.Module):
    """
    SyncNet v2 model for audio-visual synchronization.
    
    Architecture matches official Oxford VGG pretrained weights.
    
    Input:
        - face: (B, 3, 5, H, W) - RGB, 5 frames, 112x112 face crops
        - audio: (B, 1, 13, 20) - MFCC features for 0.2s audio
    
    Output:
        - face_embedding: (B, 1024)
        - audio_embedding: (B, 1024)
    """
    
    def __init__(self):
        super(SyncNetModel, self).__init__()
        
        # ============================================
        # Face/Lip encoder - uses 3D convolutions
        # Input: (B, 3, 5, 112, 112) - RGB, 5 frames
        # ============================================
        self.netcnnlip = nn.Sequential(
            # Conv1: (B, 3, 5, 112, 112) -> (B, 96, 1, 56, 56) after pool
            nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            
            # Conv2: (B, 96, 1, 27, 27) -> (B, 256, 1, 27, 27)
            nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # Conv3: (B, 256, 1, 7, 7) -> (B, 256, 1, 7, 7)
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            
            # Conv6 (collapse spatial)
            nn.Conv3d(256, 512, kernel_size=(1, 6, 6), stride=1, padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
        
        # Face FC layers
        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
        )
        
        # ============================================
        # Audio encoder - uses 2D convolutions
        # Input: (B, 1, 13, 20) - MFCC spectrogram
        # ============================================
        self.netcnnaud = nn.Sequential(
            # Conv1
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=1),
            
            # Conv2
            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
            
            # Conv3
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            
            # Conv6 (collapse spatial)
            nn.Conv2d(256, 512, kernel_size=(5, 4), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Audio FC layers
        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
        )
    
    def forward_face(self, x):
        """
        Extract face embedding from lip crops.
        
        Args:
            x: (B, 3, 5, H, W) - RGB, 5 consecutive frames, HxW face crops
        """
        out = self.netcnnlip(x)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.netfclip(out)
        return out
    
    def forward_audio(self, x):
        """
        Extract audio embedding from MFCC.
        
        Args:
            x: (B, 1, 13, 20) - MFCC features
        """
        out = self.netcnnaud(x)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.netfcaud(out)
        return out
    
    def forward(self, face, audio):
        """
        Forward pass returning both embeddings.
        
        Returns:
            face_emb: (B, 1024)
            audio_emb: (B, 1024)
        """
        face_emb = self.forward_face(face)
        audio_emb = self.forward_audio(audio)
        return face_emb, audio_emb
    
    @staticmethod
    def compute_distance(face_emb, audio_emb):
        """Compute Euclidean distance (LSE-D). Lower is better."""
        return torch.sqrt(torch.sum((face_emb - audio_emb) ** 2, dim=1))
    
    @staticmethod
    def compute_similarity(face_emb, audio_emb):
        """Compute cosine similarity (for LSE-C). Higher is better."""
        face_norm = face_emb / (torch.norm(face_emb, dim=1, keepdim=True) + 1e-8)
        audio_norm = audio_emb / (torch.norm(audio_emb, dim=1, keepdim=True) + 1e-8)
        return torch.sum(face_norm * audio_norm, dim=1)


class SyncNetColor(SyncNetModel):
    """Alias - base SyncNet already supports RGB."""
    pass
