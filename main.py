from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import speech_recognition as sr
import librosa
import soundfile as sf
import io
import os
import tempfile
from typing import Optional
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Transcription API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recognizer
recognizer = sr.Recognizer()

# Language mapping
LANGUAGE_MAPPING = {
    "english": "en-US",
    "hindi": "hi-IN", 
    "tamil": "ta-IN"
}

# Supported audio formats
SUPPORTED_FORMATS = {
    '.wav': 'WAV',
    '.mp3': 'MP3', 
    '.flac': 'FLAC',
    '.ogg': 'OGG',
    '.m4a': 'M4A',
    '.aac': 'AAC',
    '.wma': 'WMA'
}

def convert_audio_to_wav(audio_file: bytes, file_extension: str) -> bytes:
    """Convert audio file to WAV format using librosa and soundfile"""
    try:
        # Create a temporary file to work with
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_input:
            temp_input.write(audio_file)
            temp_input_path = temp_input.name
        
        try:
            # Load audio using librosa
            logger.info(f"Loading audio file: {file_extension}")
            y, sr_audio = librosa.load(temp_input_path, sr=None)
            
            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
            
            # Resample to 16kHz if needed (optimal for speech recognition)
            if sr_audio != 16000:
                y = librosa.resample(y, orig_sr=sr_audio, target_sr=16000)
                sr_audio = 16000
            
            # Create WAV buffer
            wav_buffer = io.BytesIO()
            
            # Save as WAV using soundfile
            sf.write(wav_buffer, y, sr_audio, format='WAV', subtype='PCM_16')
            wav_buffer.seek(0)
            
            return wav_buffer.getvalue()
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
                
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail=f"Error processing audio file. Supported formats: {', '.join(SUPPORTED_FORMATS.keys())}. Error: {str(e)}"
        )

def transcribe_audio(audio_data: bytes, language: str) -> str:
    """Transcribe audio using speech recognition"""
    try:
        # Create audio file object
        audio_file = sr.AudioFile(io.BytesIO(audio_data))
        
        with audio_file as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Record audio
            audio = recognizer.record(source)
        
        # Perform transcription
        transcription = recognizer.recognize_google(
            audio, 
            language=LANGUAGE_MAPPING.get(language.lower(), "en-US")
        )
        
        return transcription
    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="Speech could not be understood")
    except sr.RequestError as e:
        logger.error(f"Speech recognition service error: {str(e)}")
        raise HTTPException(status_code=500, detail="Speech recognition service error")
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "message": "Audio Transcription API is running",
        "supported_formats": list(SUPPORTED_FORMATS.keys()),
        "libraries": {
            "librosa": "Audio processing",
            "soundfile": "Audio I/O",
            "speech_recognition": "Speech recognition"
        }
    }

@app.post("/transcribe")
async def transcribe_audio_file(
    file: UploadFile = File(...),
    language: str = Form("english")
):
    """
    Upload and transcribe an audio file
    
    Args:
        file: Audio file to transcribe
        language: Language of the audio (english, hindi, tamil)
    
    Returns:
        JSON response with transcription text
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Validate language
        if language.lower() not in LANGUAGE_MAPPING:
            raise HTTPException(
                status_code=400, 
                detail="Language must be one of: english, hindi, tamil"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Get file extension
        file_extension = os.path.splitext(file.filename)[1] if file.filename else '.wav'
        
        # Validate file format
        if file_extension.lower() not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {file_extension}. Supported formats: {', '.join(SUPPORTED_FORMATS.keys())}"
            )
        
        # Convert to WAV if needed
        if file_extension.lower() != '.wav':
            logger.info(f"Converting {file_extension} to WAV format")
            file_content = convert_audio_to_wav(file_content, file_extension)
        
        # Transcribe audio
        transcription = transcribe_audio(file_content, language)
        
        return JSONResponse({
            "success": True,
            "transcription": transcription,
            "language": language,
            "filename": file.filename,
            "format": SUPPORTED_FORMATS.get(file_extension.lower(), "Unknown")
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
