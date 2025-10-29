"""
Configuration Module for Digital Health Card System
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
import dotenv

# Force load environment variables
dotenv.load_dotenv(override=True)


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    APP_NAME: str = "Digital Health Card System"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = Field(default=False)
    BASE_URL: str = Field(default="http://localhost:8000")
    
    # Database
    MONGODB_URL: str = Field(default="mongodb://localhost:27017")
    DATABASE_NAME: str = Field(default="digital_health_card")
    
    # Cloudinary
    CLOUDINARY_CLOUD_NAME: str = Field(default="")
    CLOUDINARY_API_KEY: str = Field(default="")
    CLOUDINARY_API_SECRET: str = Field(default="")
    CLOUD_NAME: str = Field(default="")
    CLOUD_API_KEY: str = Field(default="")
    CLOUD_API_SECRET: str = Field(default="")
    
    # JWT Authentication - FIXED: Provide default that meets validation
    JWT_SECRET: str = Field(
        default="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2",
        min_length=32
    )
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRY_DAYS: int = Field(default=30)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7)
    
    # CORS
    CORS_ORIGINS: str = Field(default="http://localhost:3000,http://localhost:5173")
    
    # AI Configuration
    GEMINI_API_KEY: Optional[str] = Field(default=None)
    AI_MODEL: str = Field(default="gemini-1.5-flash")
    AI_MAX_TOKENS: int = Field(default=1000)
    AI_TEMPERATURE: float = Field(default=0.7)
    
    # Google Maps API
    GOOGLE_MAPS_API_KEY: Optional[str] = Field(default=None)
    
    # Fitbit Integration
    FITBIT_CLIENT_ID: Optional[str] = Field(default=None)
    FITBIT_CLIENT_SECRET: Optional[str] = Field(default=None)
    
    # File Upload
    MAX_FILE_SIZE_MB: int = Field(default=10)
    ALLOWED_FILE_TYPES: str = Field(default="application/pdf,image/jpeg,image/png,image/jpg")
    MAX_UPLOAD_SIZE: int = Field(default=10 * 1024 * 1024)
    ALLOWED_EXTENSIONS: List[str] = Field(default=[".pdf", ".jpg", ".jpeg", ".png"])
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60)
    
    # Medication Settings
    MEDICATION_REMINDER_ADVANCE_MINUTES: int = Field(default=15)
    MEDICATION_ADHERENCE_THRESHOLD: float = Field(default=80.0)
    
    # Appointment Settings
    APPOINTMENT_REMINDER_HOURS: int = Field(default=24)
    APPOINTMENT_CANCELLATION_HOURS: int = Field(default=24)
    
    # Email Configuration
    SMTP_HOST: Optional[str] = Field(default=None)
    SMTP_PORT: int = Field(default=587)
    SMTP_USER: Optional[str] = Field(default=None)
    SMTP_PASSWORD: Optional[str] = Field(default=None)
    SMTP_FROM_EMAIL: Optional[str] = Field(default=None)
    
    # SMS Configuration
    TWILIO_ACCOUNT_SID: Optional[str] = Field(default=None)
    TWILIO_AUTH_TOKEN: Optional[str] = Field(default=None)
    TWILIO_PHONE_NUMBER: Optional[str] = Field(default=None)
    
    # OCR Configuration
    TESSERACT_PATH: Optional[str] = Field(default=None)
    OCR_LANGUAGE: str = Field(default="eng")
    
    # Data Retention
    AUDIT_LOG_RETENTION_DAYS: int = Field(default=90)
    CHAT_HISTORY_RETENTION_DAYS: int = Field(default=180)
    
    # Notification Settings
    ENABLE_EMAIL_NOTIFICATIONS: bool = Field(default=False)
    ENABLE_SMS_NOTIFICATIONS: bool = Field(default=False)
    ENABLE_PUSH_NOTIFICATIONS: bool = Field(default=True)
    
    # Hospital Tracking Settings
    DEFAULT_HOSPITAL_SEARCH_RADIUS_KM: int = Field(default=10)
    MAX_HOSPITAL_SEARCH_RADIUS_KM: int = Field(default=50)
    
    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        if isinstance(self.CORS_ORIGINS, str):
            return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
        return []
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )


# Create settings instance
settings = Settings()

# Rest of your config classes...
class DatabaseConfig:
    USERS_COLLECTION = "users"
    HOSPITALS_COLLECTION = "hospitals"
    PRESCRIPTIONS_COLLECTION = "prescriptions"
    APPOINTMENTS_COLLECTION = "appointments"
    CHATS_COLLECTION = "chats"
    MEDICATIONS_COLLECTION = "medications"
    VITALS_COLLECTION = "vitals"
    NOTIFICATIONS_COLLECTION = "notifications"

db_config = DatabaseConfig()

class SecurityConfig:
    MIN_PASSWORD_LENGTH = 8
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGIT = True
    REQUIRE_SPECIAL_CHAR = False
    ROLES = ["admin", "doctor", "patient"]

security_config = SecurityConfig()

class VitalsConfig:
    NORMAL_RANGES = {
        "heart_rate": {"min": 60, "max": 100, "unit": "bpm"},
        "blood_pressure_systolic": {"min": 90, "max": 140, "unit": "mmHg"},
        "blood_pressure_diastolic": {"min": 60, "max": 90, "unit": "mmHg"},
    }

vitals_config = VitalsConfig()

__all__ = ["settings", "db_config", "security_config", "vitals_config"]
