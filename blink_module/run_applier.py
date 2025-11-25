import sys
import os
from BlinkApplier import BlinkApplier

# --- Configuration ---
LIP_VIDEO = "lip_sync.mp4"
ASSET_FILE = "my_actor_blinks.npz"
OUTPUT_VIDEO = "final_video_with_blinks.mp4"
DLIB_MODEL = "shape_predictor_68_face_landmarks.dat"
DURATION = 30
STATIC_HEAD = True

def run_blink_pipeline():
    print("Starting blink application pipeline...")
    
    # 1. Check for files
    if not os.path.exists(LIP_VIDEO):
        print(f"Error: Input video not found: {LIP_VIDEO}"); sys.exit(1)
    if not os.path.exists(ASSET_FILE):
        print(f"Error: Asset file not found: {ASSET_FILE}"); sys.exit(1)
    if not os.path.exists(DLIB_MODEL):
        print(f"Error: Dlib model not found: {DLIB_MODEL}"); sys.exit(1)
        
    try:
        # 2. Initialize the applier (loads models and assets)
        print("Loading BlinkApplier...")
        applier = BlinkApplier(
            dlib_model_path=DLIB_MODEL,
            blink_assets_path=ASSET_FILE
        )
        print("...BlinkApplier loaded.")
        
        # 3. Process the video
        print(f"Processing '{LIP_VIDEO}'...")
        applier.process_video(
            lip_video_path=LIP_VIDEO,
            output_video_path=OUTPUT_VIDEO,
            blink_prob=0.03, # Default value
            duration=DURATION,
            static_head=STATIC_HEAD,
            status_callback=lambda msg: print(f"[PROGRESS] {msg}") # Simple console callback
        )
        
        print(f"\n[SUCCESS] Pipeline complete. Final video saved to: {OUTPUT_VIDEO}")
        
    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        print(e)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_blink_pipeline()