from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    EMAIL_ADDRESS: str = ""
    SENDGRID_API_KEY: str = ""
    REDIS_URL: str = ""
    RSA: str = ""
    NDB: str = ""
    CS: str = ""
    AWS_REGION: str = "us-east-1"
    ACCESS_KEY_ID: str = ""
    SECRET_ACCESS_KEY: str = ""
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = ""
    S3_BUCKET_NAME: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields in .env

    @classmethod
    def load_settings(cls):
        """Load settings, falling back to Streamlit secrets if available"""
        # Check if running in Streamlit environment
        try:
            import streamlit as st
            # If we can access st.secrets, use it
            if hasattr(st, 'secrets') and st.secrets:
                return cls(
                    EMAIL_ADDRESS=st.secrets.get("EMAIL_ADDRESS", ""),
                    SENDGRID_API_KEY=st.secrets.get("SENDGRID_API_KEY", ""),
                    REDIS_URL=st.secrets.get("REDIS_URL", ""),
                    RSA=st.secrets.get("RSA", ""),
                    NDB=st.secrets.get("NDB", ""),
                    CS=st.secrets.get("CS", ""),
                    AWS_REGION=st.secrets.get("AWS_REGION", "us-east-1"),
                    ACCESS_KEY_ID=st.secrets.get("ACCESS_KEY_ID", ""),
                    SECRET_ACCESS_KEY=st.secrets.get("SECRET_ACCESS_KEY", ""),
                    PINECONE_API_KEY=st.secrets.get("PINECONE_API_KEY", ""),
                    PINECONE_INDEX_NAME=st.secrets.get("PINECONE_INDEX_NAME", ""),
                    S3_BUCKET_NAME=st.secrets.get("S3_BUCKET_NAME", ""),
                )
        except (ImportError, Exception):
            pass
        
        # Fallback to .env file or environment variables
        try:
            return cls()
        except Exception as e:
            print(f"Warning: Could not load settings: {e}")
            return cls()

# Load settings
settings = Settings.load_settings()
