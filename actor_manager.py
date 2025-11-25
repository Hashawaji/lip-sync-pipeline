"""
Actor Manager - Handles actor configuration and voice generation
"""
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActorConfig:
    """Configuration for a single actor"""
    id: str
    name: str
    display_name: str
    description: str
    viseme_path: Optional[Path]  # Can be None if no visemes available
    voice_config: Dict
    blink_assets_path: Optional[Path] = None  # Path to extracted_blinks.npz
    thumbnail: Optional[str] = None
    
    @property
    def voice_backend(self) -> str:
        """Get the TTS backend for this actor"""
        return self.voice_config.get('backend', 'gtts')
    
    @property
    def voice_params(self) -> Dict:
        """Get voice parameters for TTS"""
        return self.voice_config.get('params', {})
    
    @property
    def has_visemes(self) -> bool:
        """Check if actor has viseme library available"""
        return self.viseme_path is not None and self.viseme_path.exists()
    
    @property
    def has_blink_assets(self) -> bool:
        """Check if actor has blink assets available"""
        return self.blink_assets_path is not None and self.blink_assets_path.exists()


class ActorManager:
    """Manages multiple actors and their configurations"""
    
    def __init__(self, actors_dir: Optional[Path] = None):
        """
        Initialize the actor manager
        
        Args:
            actors_dir: Path to the actors directory (default: ./actors)
        """
        if actors_dir is None:
            actors_dir = Path(__file__).parent / "actors"
        
        self.actors_dir = Path(actors_dir)
        self.config_file = self.actors_dir / "actors_config.yaml"
        self.actors: Dict[str, ActorConfig] = {}
        
        self._load_actors()
    
    def _load_actors(self):
        """Load all actor configurations"""
        if not self.actors_dir.exists():
            logger.warning(f"Actors directory not found: {self.actors_dir}")
            return
        
        # Load central config if exists
        central_config = {}
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                central_config = yaml.safe_load(f) or {}
        
        # Scan for actor directories
        for actor_dir in self.actors_dir.iterdir():
            if not actor_dir.is_dir() or actor_dir.name.startswith('.'):
                continue
            
            try:
                actor = self._load_actor(actor_dir, central_config)
                if actor:
                    self.actors[actor.id] = actor
            except Exception as e:
                logger.error(f"Failed to load actor from {actor_dir}: {e}")
    
    def _load_actor(self, actor_dir: Path, central_config: Dict) -> Optional[ActorConfig]:
        """Load a single actor configuration"""
        actor_id = actor_dir.name
        
        # Load actor-specific metadata
        metadata_file = actor_dir / "metadata.yaml"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = yaml.safe_load(f) or {}
        else:
            metadata = {}
        
        # Check for viseme directory (try visemes_library first, then visemes as fallback)
        viseme_dir = actor_dir / "visemes_library"
        if not viseme_dir.exists():
            viseme_dir = actor_dir / "visemes"
            if not viseme_dir.exists():
                # Fallback: check if actor_dir itself contains visemes
                if any(actor_dir.glob("*.png")) or any(actor_dir.glob("*.jpg")):
                    viseme_dir = actor_dir
                else:
                    logger.warning(f"No visemes_library found for actor: {actor_id}")
                    # Still create the actor config but mark viseme_dir as None
                    viseme_dir = None
        
        # Merge configurations (actor-specific overrides central)
        actor_config = central_config.get('actors', {}).get(actor_id, {})
        actor_config.update(metadata)
        
        # Check for blink assets
        blink_assets_path = None
        if 'blink_assets' in actor_config and actor_config['blink_assets']:
            # Blink assets specified in config
            blink_assets_file = actor_dir / actor_config['blink_assets']
            if blink_assets_file.exists():
                blink_assets_path = blink_assets_file
            else:
                logger.warning(f"Blink assets specified but not found: {blink_assets_file}")
        else:
            # Try default location: actor_dir/blink_assets/extracted_blinks.npz
            default_blink_path = actor_dir / "blink_assets" / "extracted_blinks.npz"
            if default_blink_path.exists():
                blink_assets_path = default_blink_path
        
        # Create ActorConfig (viseme_path can be None if no visemes found)
        return ActorConfig(
            id=actor_id,
            name=actor_config.get('name', actor_id),
            display_name=actor_config.get('display_name', actor_id.replace('_', ' ').title()),
            description=actor_config.get('description', 'No description available'),
            viseme_path=viseme_dir,
            voice_config=actor_config.get('voice', {}),
            blink_assets_path=blink_assets_path,
            thumbnail=actor_config.get('thumbnail')
        )
    
    def get_actor(self, actor_id: str) -> Optional[ActorConfig]:
        """Get actor configuration by ID"""
        return self.actors.get(actor_id)
    
    def list_actors(self) -> List[ActorConfig]:
        """Get list of all available actors"""
        return list(self.actors.values())
    
    def get_actor_choices(self) -> Dict[str, str]:
        """Get actor choices for UI (id -> display_name)"""
        return {actor.id: actor.display_name for actor in self.actors.values()}
    
    def generate_actor_voice(self, actor_id: str, text: str, output_path: str):
        """Generate voice for specific actor"""
        actor = self.get_actor(actor_id)
        if not actor:
            raise ValueError(f"Actor not found: {actor_id}")
        
        backend = actor.voice_backend
        voice_params = actor.voice_params
        
        # Extract parameters for text_to_speech
        tts_params = {}
        
        # Common parameters
        if 'lang' in voice_params:
            tts_params['lang'] = voice_params['lang']
        if 'speed' in voice_params:
            tts_params['speed'] = voice_params['speed']
        if 'voice' in voice_params:
            tts_params['voice'] = voice_params['voice']
        
        # Handle backend-specific parameters
        if backend == 'gtts' and ('tld' in voice_params or 'slow' in voice_params):
            # Add gTTS-specific parameters
            tts_params['tld'] = voice_params.get('tld', 'com')
            tts_params['slow'] = voice_params.get('slow', False)
        
        # Use the standard text_to_speech function (supports edge, kokoro, espeak, gtts)
        from tts_engine import text_to_speech
        
        backend_literal = backend if backend in ['espeak', 'gtts', 'kokoro', 'edge', 'auto'] else 'auto'
        
        text_to_speech(
            text=text,
            output_path=output_path,
            backend=backend_literal,  # type: ignore
            **tts_params
        )