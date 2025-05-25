"""
YouTube audio downloader module with enhanced filename sanitization.
This module handles downloading audio from YouTube videos.
"""

import os
import subprocess
import sys
import re
import unicodedata

def sanitize_filename(filename):
    """
    Sanitize the filename to remove invalid characters and make it filesystem-friendly.
    
    Args:
        filename (str): The original filename
        
    Returns:
        str: A sanitized filename safe for most filesystems
    """
    # Remove invalid characters
    filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
    
    # Replace apostrophes with underscores (apostrophes can cause issues in some contexts)
    filename = filename.replace("'", "_")
    
    # Replace spaces with underscores
    filename = filename.replace(" ", "_")
    
    # Normalize unicode characters
    filename = unicodedata.normalize('NFKD', filename)
    
    # Remove any remaining non-ASCII characters
    filename = ''.join(c for c in filename if ord(c) < 128)
    
    # Truncate if too long (most filesystems have limits)
    max_length = 100  # Safe value for most filesystems
    if len(filename) > max_length:
        # Keep extension if present
        parts = filename.rsplit('.', 1)
        if len(parts) > 1:
            filename = parts[0][:max_length - len(parts[1]) - 1] + '.' + parts[1]
        else:
            filename = filename[:max_length]
    
    # Ensure we don't end up with an empty filename
    if not filename:
        filename = "youtube_audio"
        
    return filename

def install_yt_dlp():
    """Install yt-dlp if not already installed."""
    try:
        import yt_dlp
        print("✅ yt-dlp already installed.")
        return True
    except ImportError:
        print("📦 Installing yt-dlp...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
            print("✅ yt-dlp installed successfully.")
            return True
        except Exception as e:
            print(f"❌ Failed to install yt-dlp: {e}")
            return False

def download_youtube_audio(url, output_dir="./audios"):
    """
    Download audio from a YouTube video.
    
    Args:
        url (str): YouTube video URL
        output_dir (str): Directory to save audio files
        
    Returns:
        str: Path to the downloaded audio file
    """
    # Install yt-dlp if needed
    if not install_yt_dlp():
        raise ImportError("Failed to install required package: yt-dlp")
    
    # Import yt_dlp after ensuring it's installed
    import yt_dlp
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Temporary path for info extraction
    try:
        # Extract info first without downloading
        print("📋 Fetching video information...")
        ydl_opts_info = {
            'quiet': True,
            'skip_download': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'video')
            
        # Sanitize the title for safe filename
        safe_title = sanitize_filename(title)
        print(f"🔤 Video title: {title}")
        print(f"🔤 Safe filename: {safe_title}")
        
        # Output template with sanitized name
        output_template = os.path.join(output_dir, safe_title + ".%(ext)s")
        
        # Common ffmpeg locations for Windows
        potential_ffmpeg_paths = [
            "C:\\ffmpeg\\bin",
            "C:\\Program Files\\ffmpeg\\bin",
            os.path.join(os.environ.get("USERPROFILE", ""), "ffmpeg", "bin")
        ]
        
        # Find ffmpeg path
        ffmpeg_path = None
        for path in potential_ffmpeg_paths:
            if os.path.exists(path):
                ffmpeg_path = path
                print(f"🔍 Found ffmpeg at: {ffmpeg_path}")
                break
        
        # yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_template,
            'quiet': False,
        }
        
        # Add ffmpeg location if found
        if ffmpeg_path:
            ydl_opts['ffmpeg_location'] = ffmpeg_path
        
        print("📥 Downloading audio...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Expected output file path
        mp3_path = os.path.join(output_dir, safe_title + ".mp3")
        
        # Check if the file exists
        if not os.path.exists(mp3_path):
            print("⚠️ Could not find expected output file. Looking for alternatives...")
            # Try to find any MP3 file that was recently created
            mp3_files = [f for f in os.listdir(output_dir) if f.endswith('.mp3')]
            if mp3_files:
                # Sort by creation time (newest first)
                mp3_files.sort(key=lambda f: os.path.getctime(os.path.join(output_dir, f)), reverse=True)
                mp3_path = os.path.join(output_dir, mp3_files[0])
                print(f"🔄 Using alternative file: {mp3_path}")
            else:
                raise FileNotFoundError("❌ No MP3 files found in output directory")
        
        print(f"✅ Audio saved to: {mp3_path}")
        return mp3_path
            
    except Exception as e:
        print(f"❌ Download error: {str(e)}")
        raise

# Test function
if __name__ == "__main__":
    url = input("Enter YouTube URL: ")
    try:
        audio_path = download_youtube_audio(url)
        print(f"Success! Audio downloaded to: {audio_path}")
    except Exception as e:
        print(f"Failed: {e}")