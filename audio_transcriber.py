"""
Audio transcription module using Faster Whisper.
This module handles transcribing audio files to text.
"""

import os
import subprocess
import sys
import torch
import time
import shutil

def install_packages():
    """Install required packages if not already installed."""
    packages = ["faster-whisper"]
    
    for package in packages:
        try:
            if package == "faster-whisper":
                import faster_whisper
                print("✅ faster-whisper already installed.")
            else:
                __import__(package.replace("-", "_"))
                print(f"✅ {package} already installed.")
        except ImportError:
            print(f"📦 Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed successfully.")
            except Exception as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
    return True

def create_safe_temp_dir():
    """Create a safe temporary directory for intermediate files."""
    base_dir = "./temp_whisper"
    # Create with timestamp to avoid conflicts
    temp_dir = f"{base_dir}_{int(time.time())}"
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    print(f"📁 Created temporary directory: {temp_dir}")
    return temp_dir

def cleanup_temp_dir(temp_dir):
    """Clean up temporary directory."""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"⚠️ Failed to clean up temporary directory: {e}")

def transcribe_audio(audio_path, model_size="medium"):
    """
    Transcribe audio file using Whisper model.
    
    Args:
        audio_path (str): Path to audio file
        model_size (str): Whisper model size (options: tiny, base, small, medium, large-v2)
        
    Returns:
        list: List of transcribed text segments
    """
    # Install packages if needed
    if not install_packages():
        raise ImportError("Failed to install required packages")
    
    # Import after ensuring it's installed
    from faster_whisper import WhisperModel
    
    # Verify the audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ Audio file not found: {audio_path}")
    
    # Create temporary directory for any intermediate files
    temp_dir = create_safe_temp_dir()
    
    try:
        print(f"🧠 Loading Whisper model ({model_size}) on {'GPU' if torch.cuda.is_available() else 'CPU'}...")
        
        # Determine device and compute type
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        
        # Set download directory for model files
        # This helps avoid problems with special characters in paths
        os.environ["XDG_CACHE_HOME"] = temp_dir
        os.environ["HF_HOME"] = os.path.join(temp_dir, "huggingface")
        
        # Initialize the model with safe download paths
        model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type,
            download_root=os.path.join(temp_dir, "models")
        )
        
        print("📝 Transcribing audio...")
        segments, info = model.transcribe(
            audio_path, 
            beam_size=5,
            word_timestamps=False  # Set to True if you want word-level timestamps
        )
        
        # Collect segments
        transcript_segments = []
        for segment in segments:
            transcript_segments.append(segment.text)
        
        print(f"✅ Transcription complete! Found {len(transcript_segments)} segments.")
        return transcript_segments
        
    except Exception as e:
        print(f"❌ Transcription error: {str(e)}")
        raise
    finally:
        # Always clean up temporary directory
        cleanup_temp_dir(temp_dir)

# Test function
if __name__ == "__main__":
    audio_path = input("Enter path to audio file: ")
    try:
        transcript = transcribe_audio(audio_path)
        print("\nTranscript Preview:")
        print("-" * 50)
        full_text = "\n".join(transcript)
        print(full_text[:500] + ("..." if len(full_text) > 500 else ""))
        print("-" * 50)
    except Exception as e:
        print(f"Failed: {e}")