#!/usr/bin/env python3
"""
Lip-Sync Pipeline - Streamlit Web App
Modern web interface for generating lip-sync videos from text
"""
import streamlit as st
import subprocess
import time
from pathlib import Path
import sys
import os

# Add Montreal-Forced-Aligner to path BEFORE importing other modules
sys.path.insert(0, str(Path(__file__).parent / "Montreal-Forced-Aligner"))

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Now import modules that depend on Montreal-Forced-Aligner
from fast_phoneme_extractor import FastPhonemeExtractor
from tts_engine import text_to_speech
from actor_manager import ActorManager  # Import ActorManager

# Import BlinkApplier
sys.path.insert(0, str(Path(__file__).parent / "blink_module"))
from BlinkApplier import BlinkApplier
from BlinkScheduler import BlinkScheduler


# Page config
st.set_page_config(
    page_title="Lip-Sync Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .step-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
    .actor-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #e0e0e0;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    /* Limit video player size */
    video {
        max-width: 100%;
        max-height: 600px;
        width: auto !important;
        height: auto !important;
        object-fit: contain;
    }
    </style>
""", unsafe_allow_html=True)


def log_message(container, message, level="info"):
    """Display log message with appropriate styling"""
    if level == "success":
        container.success(message)
    elif level == "error":
        container.error(message)
    elif level == "warning":
        container.warning(message)
    elif level == "step":
        container.markdown(f'<div class="step-box">🔵 {message}</div>', unsafe_allow_html=True)
    else:
        container.info(message)


def get_cached_mfa_extractor():
    """
    Get or create cached MFA extractor.
    Loads models once and reuses them across video generations.
    """
    if 'mfa_extractor' not in st.session_state:
        with st.spinner("🔄 Loading MFA models (one-time initialization)..."):
            start_time = time.time()
            st.session_state.mfa_extractor = FastPhonemeExtractor()
            load_time = time.time() - start_time
            st.success(f"✓ MFA models loaded and cached in {load_time:.2f}s")
    return st.session_state.mfa_extractor


def get_cached_blink_applier(actor_id, actor_manager):
    """
    Get or create cached BlinkApplier for a specific actor.
    Loads dlib models and blink assets once per actor.
    
    Args:
        actor_id: The actor ID
        actor_manager: ActorManager instance
        
    Returns:
        BlinkApplier instance or None if actor has no blink assets
    """
    # Initialize cache dictionary if needed
    if 'blink_appliers' not in st.session_state:
        st.session_state.blink_appliers = {}
    
    # Get actor info
    actor = actor_manager.get_actor(actor_id)
    if not actor or not actor.has_blink_assets:
        return None
    
    # Return cached applier if available
    if actor_id in st.session_state.blink_appliers:
        return st.session_state.blink_appliers[actor_id]
    
    # Load and cache new applier
    dlib_model_path = Path(__file__).parent / "blink_module" / "assets" / "shape_predictor_68_face_landmarks.dat"
    
    if not dlib_model_path.exists():
        return None
    
    with st.spinner(f"🔄 Loading blink models for {actor.display_name} (one-time initialization)..."):
        start_time = time.time()
        blink_applier = BlinkApplier(
            dlib_model_path=str(dlib_model_path),
            blink_assets_path=str(actor.blink_assets_path)
        )
        load_time = time.time() - start_time
        st.session_state.blink_appliers[actor_id] = blink_applier
        st.success(f"✓ Blink models cached for {actor.display_name} in {load_time:.2f}s")
    
    return blink_applier


def generate_video_pipeline(text, output_name, actor_id, actor_manager, progress_bar, status_text, log_container):
    """Run the complete video generation pipeline"""
    
    try:
        # Get actor configuration
        actor = actor_manager.get_actor(actor_id)
        if not actor:
            raise ValueError(f"Actor not found: {actor_id}")
        
        # Check if actor has viseme library
        if not actor.has_visemes:
            raise ValueError(f"No viseme library available for actor: {actor.display_name}")
        
        output_dir = Path("outputs") / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_steps = 5  # Updated: 5 steps (blinks now integrated in step 4)
        step_times = {}  # Track timing for each step
        
        # Step 1: Generate text file
        step1_start = time.time()
        status_text.text("Step 1/5: Generating text file...")
        progress_bar.progress(1 / total_steps)
        log_message(log_container, "📝 Saving text file...", "step")
        
        text_file = output_dir / "text.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        step1_time = time.time() - step1_start
        step_times['Text File Generation'] = step1_time
        log_message(log_container, f"✓ Text file saved: {text_file} (Time: {step1_time:.3f}s)", "info")
        time.sleep(0.3)
        
        # Step 2: Generate audio with actor-specific voice
        step2_start = time.time()
        status_text.text(f"Step 2/5: Generating audio with {actor.display_name}'s voice...")
        progress_bar.progress(2 / total_steps)
        log_message(log_container, f"🎤 Generating audio with {actor.display_name}'s voice ({actor.voice_backend})...", "step")
        
        audio_file = output_dir / "audio.mp3"
        actor_manager.generate_actor_voice(actor_id, text, str(audio_file))
        
        step2_time = time.time() - step2_start
        step_times['Audio Generation (TTS)'] = step2_time
        log_message(log_container, f"✓ Audio generated (Time: {step2_time:.3f}s)", "success")
        time.sleep(0.3)
        
        # Step 3: MFA processing
        step3_start = time.time()
        status_text.text("Step 3/5: Running MFA phoneme alignment...")
        progress_bar.progress(3 / total_steps)
        log_message(log_container, "🔬 Running Montreal Forced Aligner...", "step")

        try:
            # Define paths
            text_file = output_dir / "text.txt"
            audio_file = output_dir / "audio.mp3"
            phoneme_json = output_dir / "complete_phoneme_alignments.json"
            transform_script = Path(__file__).parent / "phoneme-json-transformation.py"
            
            # Step 3a: Get cached FastPhonemeExtractor (no reload!)
            with log_container.expander("📋 MFA Alignment Details", expanded=False):
                mfa_log = st.empty()
                
                try:
                    # Use cached extractor - models already in memory!
                    extractor = get_cached_mfa_extractor()
                    mfa_log.info("✓ Using cached MFA models (already in memory)")
                except Exception as e:
                    mfa_log.error(f"Failed to get MFA extractor: {e}")
                    raise
                
                # Step 3b: Extract phonemes
                mfa_log.info(f"Extracting phonemes from: {audio_file.name}")
                try:
                    result = extractor.extract_from_files(audio_file, text_file)
                    mfa_log.info(f"✓ Extraction complete. Result type: {type(result)}")
                    
                    # Debug: Check result structure
                    if not isinstance(result, dict):
                        raise ValueError(f"Expected dict result, got {type(result)}")
                    
                    if 'files' not in result:
                        mfa_log.error(f"Result keys: {list(result.keys())}")
                        raise KeyError(f"'files' key not found in result. Available keys: {list(result.keys())}")
                    
                    if 'audio' not in result['files']:
                        raise KeyError(f"'audio' key not found in result['files']. Available keys: {list(result['files'].keys())}")
                    
                except Exception as e:
                    mfa_log.error(f"Failed to extract phonemes: {e}")
                    raise
                
                # Step 3c: Save result as complete_phoneme_alignments.json
                try:
                    extractor.save_result(result, phoneme_json)
                    phoneme_count = result['files']['audio']['phoneme_count']
                    word_count = result['files']['audio']['word_count']
                    mfa_log.success(f"✓ Saved phoneme alignments: {phoneme_count} phonemes, {word_count} words")
                except Exception as e:
                    mfa_log.error(f"Failed to save result: {e}")
                    raise
            
            step3_time = time.time() - step3_start
            step_times['MFA Phoneme Alignment'] = step3_time
            log_message(log_container, f"✓ MFA alignment completed (Time: {step3_time:.1f}s = {step3_time/60:.2f} min)", "success")
            
            # Step 3d: Run phoneme JSON transformation
            status_text.text("Step 3.5/5: Running phoneme transformation...")
            log_message(log_container, "🔄 Transforming phoneme JSON...", "step")
            
            if not phoneme_json.exists():
                raise Exception(f"Phoneme JSON not found: {phoneme_json}")
            
            # Run transformation script
            transform_cmd = [
                sys.executable,
                str(transform_script),
                "--input_json", str(phoneme_json),
                "--transform-type", "fixed-length"
            ]
            
            transform_result = subprocess.run(
                transform_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if transform_result.returncode != 0:
                raise Exception(f"Phoneme transformation failed: {transform_result.stderr}")
            
            log_message(log_container, "✓ Phoneme transformation completed", "success")
            
            # Clean corpus folder if it exists
            corpus_dir = Path(__file__).parent / "mfa_workspace" / "corpus"
            if corpus_dir.exists():
                import shutil
                shutil.rmtree(corpus_dir)
                corpus_dir.mkdir(parents=True, exist_ok=True)
                log_message(log_container, "✓ Cleaned corpus folder", "info")
        except Exception as e:
            log_message(log_container, f"✗ MFA processing failed: {e}", "error")
            raise
        time.sleep(0.3)
        
        # Step 4: Generate video with actor-specific visemes and blinks
        step4_start = time.time()
        status_text.text(f"Step 4/5: Generating lip-sync video with {actor.display_name}...")
        progress_bar.progress(4 / total_steps)
        log_message(log_container, f"🎬 Creating video using {actor.display_name}'s viseme library...", "step")
        
        # Get cached BlinkApplier if actor has blink assets
        blink_applier = None
        if actor.has_blink_assets:
            log_message(log_container, f"👁️ Loading blink models for {actor.display_name}...", "info")
            blink_applier = get_cached_blink_applier(actor_id, actor_manager)
            if blink_applier:
                log_message(log_container, "✓ Blink models loaded (will apply during video generation)", "success")
        
        from video_generator import generate_video
        
        video_file = output_dir / "output_video.mp4"
        json_file = output_dir / "complete_phoneme_alignments_w_reps_fixed_len.json"
        viseme_library = actor.viseme_path  # Use actor's viseme path
        
        if not viseme_library.exists():
            raise Exception(f"Actor viseme library not found at: {viseme_library}")
        
        if not json_file.exists():
            raise Exception(f"Phoneme alignment JSON not found: {json_file}")
        
        # Generate video with integrated blink application
        video_file = generate_video(
            triphone_visemes_dir=str(viseme_library),
            json_path=str(json_file),
            audio_path=str(audio_file),
            output_path=str(video_file),
            blink_applier=blink_applier,  # Pass blink applier
            actor_blink_assets_path=str(actor.blink_assets_path) if actor.has_blink_assets else None
        )
        
        step4_time = time.time() - step4_start
        step_times['Video Generation (with blinks)'] = step4_time
        log_message(log_container, f"✓ Video generated with blinks applied (Time: {step4_time:.1f}s = {step4_time/60:.2f} min)", "success")
        time.sleep(0.3)
        
        # Step 5: Skipped (blinks already applied during video generation)
        final_video_file = video_file
        log_message(log_container, "✓ Blinks were applied during video generation (no separate step needed)", "info")
        
        time.sleep(0.3)
        
        # Step 5: Complete
        status_text.text("Step 5/5: Complete!")
        progress_bar.progress(1.0)
        
        st.balloons()  # Celebration animation!
        
        # Calculate total time
        total_time = sum(step_times.values())
        
        # Display timing summary in a collapsed expander
        with log_container.expander("⏱️ Performance Breakdown", expanded=False):
            timing_summary = ""
            for step_name, step_time in step_times.items():
                percentage = (step_time / total_time) * 100
                if step_time >= 60:
                    time_str = f"{step_time:.1f}s ({step_time/60:.2f} min)"
                else:
                    time_str = f"{step_time:.3f}s"
                timing_summary += f"- **{step_name}**: {time_str} ({percentage:.1f}%)\n"
            timing_summary += f"\n**Total Time**: {total_time:.1f}s ({total_time/60:.2f} min)"
            st.markdown(timing_summary)
        
        log_container.markdown(f"""
        <div class="success-box">
            <h3>🎉 Pipeline Completed Successfully!</h3>
            <p><strong>Actor:</strong> {actor.display_name}</p>
            <p><strong>Output directory:</strong> <code>{output_dir}</code></p>
            <p><strong>Video file:</strong> <code>{Path(final_video_file).name}</code></p>
            <p><strong>Total processing time:</strong> {total_time:.1f}s ({total_time/60:.2f} minutes)</p>
        </div>
        """, unsafe_allow_html=True)
        
        return True, str(final_video_file), str(output_dir), step_times
        
    except Exception as e:
        log_message(log_container, f"✗ Error: {str(e)}", "error")
        import traceback
        with log_container.expander("🐛 Full Error Traceback", expanded=True):
            st.code(traceback.format_exc())
        return False, None, None, {}


def main():
    """Main Streamlit app"""
    
    # Initialize actor manager
    actor_manager = ActorManager()
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Lip-Sync Video Generator</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Actor selection with enhanced display
        actors = actor_manager.list_actors()
        
        if not actors:
            st.error("⚠️ No actors found! Please add actor directories to the 'actors/' folder.")
            selected_actor_id = None
        else:
            st.subheader("🎭 Select Actor")
            
            # Create radio buttons for actor selection with Cleopatra as default
            # Set Cleopatra (actor_1) as default if available
            default_actor = "actor_1" if any(actor.id == "actor_1" for actor in actors) else actors[0].id
            
            # Create actor choices with default indicator
            actor_choices = {}
            for actor in actors:
                display_name = actor.display_name
                if actor.id == default_actor:
                    display_name += " ⭐ (Default)"
                actor_choices[actor.id] = display_name
            
            selected_actor_id = st.radio(
                "Choose an actor:",
                options=list(actor_choices.keys()),
                index=list(actor_choices.keys()).index(default_actor),
                format_func=lambda x: actor_choices[x],
                label_visibility="collapsed"
            )
            
            # Display selected actor details
            if selected_actor_id:
                selected_actor = actor_manager.get_actor(selected_actor_id)
                
                if selected_actor:
                    # Show warning if no viseme library available
                    if not selected_actor.has_visemes:
                        st.error(f"⚠️ No viseme library available for {selected_actor.display_name}")
                        st.info("Please add a 'visemes_library' folder to this actor's directory.")
                    
                    with st.expander("📋 Actor Details", expanded=False):
                        st.markdown(f"**Name:** {selected_actor.display_name}")
                        st.markdown(f"**Description:** {selected_actor.description}")
                        st.markdown(f"**Voice Backend:** {selected_actor.voice_backend}")
                        if selected_actor.has_visemes and selected_actor.viseme_path:
                            st.markdown(f"**Viseme Library:** `{selected_actor.viseme_path.name}` ✓")
                        else:
                            st.markdown(f"**Viseme Library:** Not available ❌")
                        if selected_actor.has_blink_assets:
                            st.markdown(f"**Blink Assets:** Available ✓")
                        else:
                            st.markdown(f"**Blink Assets:** Not available (optional)")
        
        st.markdown("---")
        
        output_name = st.text_input(
            "Output Folder Name",
            value="video_output",
            help="Name of the folder inside 'outputs/' directory"
        )
        
        st.markdown("---")
        
        st.markdown("### 🎞️ Video Settings")
        st.info("**Fixed Settings:**\n- Base FPS: 25\n- Interpolated FPS: 50 (smooth)\n- Format: H.264 MP4 (browser-compatible)")
        
        st.markdown("---")
        st.markdown("### 📊 Pipeline Steps")
        st.markdown("""
        1. 📝 Generate text file
        2. 🎤 Generate audio (TTS)
        3. 🔬 MFA phoneme alignment
        4. 🎬 Generate video + blinks
        5. ✅ Complete
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This tool generates lip-synced videos from text using:
        - Actor-specific voices (TTS)
        - Montreal Forced Aligner (MFA)
        - Triphone-based viseme matching
        - Intelligent blink scheduling
        - Frame interpolation (25→50 FPS)
        """)
        
        st.markdown("---")
        st.markdown("### 💾 Model Cache Status")
        
        # MFA cache status
        mfa_cached = 'mfa_extractor' in st.session_state
        if mfa_cached:
            st.markdown("🟢 **MFA Models**: Loaded in memory")
        else:
            st.markdown("⚪ **MFA Models**: Not loaded (will load on first use)")
        
        # Blink cache status
        if 'blink_appliers' in st.session_state and st.session_state.blink_appliers:
            cached_actors = list(st.session_state.blink_appliers.keys())
            st.markdown(f"🟢 **Blink Models**: {len(cached_actors)} actor(s) cached")
            with st.expander("📋 Cached Actors", expanded=False):
                for cached_actor_id in cached_actors:
                    cached_actor = actor_manager.get_actor(cached_actor_id)
                    if cached_actor:
                        st.markdown(f"- {cached_actor.display_name}")
        else:
            st.markdown("⚪ **Blink Models**: Not loaded (will load on first use)")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Input Text")
        text_input = st.text_area(
            "Enter the text you want to convert to a lip-sync video:",
            value="Hello world, this is a test of the lip-sync video generator!",
            height=150,
            help="Enter any text you want to be spoken in the video"
        )
        
        generate_button = st.button("🎬 Generate Video", type="primary", use_container_width=True)
    
    with col2:
        # Empty space for visual balance
        st.markdown("")
    
    # Progress and logs section
    if generate_button:
        if not text_input.strip():
            st.error("⚠️ Please enter some text!")
        elif not selected_actor_id:
            st.error("⚠️ Please select an actor from the sidebar!")
        else:
            selected_actor = actor_manager.get_actor(selected_actor_id)
            if not selected_actor or not selected_actor.has_visemes:
                st.error(f"⚠️ No viseme library available for selected actor!")
                st.info("Please add a 'visemes_library' folder with viseme images to this actor's directory.")
            else:
                st.markdown("---")
                st.subheader("🔄 Processing")
                
                # Progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                st.markdown("### 📋 Logs")
                log_container = st.container()
                
                # Run the pipeline
                start_time = time.time()
                
                success, video_path, output_dir, step_times = generate_video_pipeline(
                    text_input,
                    output_name,
                    selected_actor_id,
                    actor_manager,
                    progress_bar,
                    status_text,
                    log_container
                )
            
                elapsed_time = time.time() - start_time
                
                if success:
                    st.markdown("---")
                    st.subheader("🎥 Output")
                    
                    # Video player
                    if video_path and Path(video_path).exists():
                        st.markdown("#### 📺 Generated Video")
                        
                        # Create a container with max width for video
                        video_col1, video_col2, video_col3 = st.columns([1, 3, 1])
                        with video_col2:
                            # Force Streamlit to reload video by reading it into memory
                            # This avoids cache issues with file IDs
                            with open(video_path, 'rb') as video_file:
                                video_bytes = video_file.read()
                            st.video(video_bytes)
                        
                        # Download button
                        with open(video_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download Video",
                                data=f,
                                file_name=f"{output_name}.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                    
                    # Show output files
                    with st.expander("📁 Output Files", expanded=False):
                        output_path = Path(output_dir)
                        if output_path.exists():
                            files = list(output_path.iterdir())
                            for file in sorted(files):
                                file_size = file.stat().st_size / 1024  # KB
                                st.text(f"📄 {file.name} ({file_size:.1f} KB)")
                else:
                    st.error("❌ Video generation failed. Check the logs above for details.")


if __name__ == "__main__":
    main()
