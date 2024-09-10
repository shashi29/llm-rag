import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
    OPENAI_TEMPERATURE=os.getenv('OPENAI_TEMPERATURE')
    OPENAI_MODEL=os.getenv('OPENAI_MODEL')
    OPENAI_TOP_P=os.getenv('OPENAI_TOP_P')
    
    
    TEMP_DIR=os.getenv('TEMP_DIR')
    WHISPER_MODEL=os.getenv('WHISPER_MODEL')
    AUDIO_MODEL=os.getenv('AUDIO_MODEL')
    
    HARD_CODED_TOKEN=os.getenv('HARD_CODED_TOKEN')