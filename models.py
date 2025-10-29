from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date

# ==================== AUTH MODELS ====================

class UserLogin(BaseModel):
    """User login credentials"""
    email: EmailStr
    password: str = Field(..., min_length=8)

class UserSignup(BaseModel):
    """User signup with role-based fields"""
    full_name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: str = Field(..., pattern="^(patient|doctor|admin)$")
    
    # Patient specific fields
    phone_number: Optional[str] = None
    address: Optional[str] = None
    date_of_birth: Optional[str] = None
    blood_group: Optional[str] = None
    gender: Optional[str] = None
    emergency_contact: Optional[Dict[str, str]] = None
    
    # Doctor specific fields
    hospital_id: Optional[str] = None
    specialization: Optional[str] = None
    license_number: Optional[str] = None
    experience_years: Optional[int] = None
    availability: Optional[List[str]] = None
    
    # Admin specific fields
    hospital_name: Optional[str] = None
    hospital_address: Optional[str] = None
    phone: Optional[str] = None

    @validator('blood_group')
    def validate_blood_group(cls, v):
        if v:
            valid_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
            if v not in valid_groups:
                raise ValueError('Invalid blood group')
        return v

    @validator('gender')
    def validate_gender(cls, v):
        if v:
            valid_genders = ['Male', 'Female', 'Other']
            if v not in valid_genders:
                raise ValueError('Invalid gender')
        return v

class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    user_id: str
    role: str
    email: str

class UserResponse(BaseModel):
    """User information response"""
    id: str
    full_name: str
    email: str
    role: str
    is_active: bool = True
    created_at: datetime
    
    # Optional fields based on role
    phone_number: Optional[str] = None
    address: Optional[str] = None
    date_of_birth: Optional[str] = None
    blood_group: Optional[str] = None
    gender: Optional[str] = None
    emergency_contact: Optional[Dict[str, str]] = None
    
    hospital_id: Optional[str] = None
    specialization: Optional[str] = None
    license_number: Optional[str] = None
    experience_years: Optional[int] = None
    
    hospital_name: Optional[str] = None
    hospital_address: Optional[str] = None
    
    last_login: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# ==================== PROFILE MODELS ====================

class PatientProfile(BaseModel):
    """Patient profile update model"""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    phone_number: Optional[str] = None
    address: Optional[str] = None
    date_of_birth: Optional[str] = None
    blood_group: Optional[str] = None
    gender: Optional[str] = None
    emergency_contact: Optional[Dict[str, str]] = None

    @validator('blood_group')
    def validate_blood_group(cls, v):
        if v:
            valid_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
            if v not in valid_groups:
                raise ValueError('Invalid blood group')
        return v

class DoctorProfile(BaseModel):
    """Doctor profile update model"""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    specialization: Optional[str] = None
    license_number: Optional[str] = None
    phone: Optional[str] = None
    experience_years: Optional[int] = Field(None, ge=0, le=70)
    availability: Optional[List[str]] = None

class AdminProfile(BaseModel):
    """Admin profile update model"""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    hospital_name: Optional[str] = None
    hospital_address: Optional[str] = None
    phone: Optional[str] = None

# ==================== HOSPITAL MODELS ====================

class Location(BaseModel):
    """GeoJSON location model"""
    type: str = "Point"
    coordinates: List[float]  # [longitude, latitude]

class HospitalCreate(BaseModel):
    """Create hospital model"""
    name: str = Field(..., min_length=2, max_length=200)
    address: str = Field(..., min_length=5)
    phone: str
    email: EmailStr
    website: Optional[str] = None
    location: Optional[Location] = None
    services: List[str] = []
    operating_hours: Optional[str] = "9:00 AM - 5:00 PM"

class HospitalUpdate(BaseModel):
    """Update hospital model"""
    name: Optional[str] = Field(None, min_length=2, max_length=200)
    address: Optional[str] = Field(None, min_length=5)
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    website: Optional[str] = None
    location: Optional[Location] = None
    services: Optional[List[str]] = None
    operating_hours: Optional[str] = None

class HospitalResponse(BaseModel):
    """Hospital response model"""
    id: str
    name: str
    address: str
    phone: str
    email: str
    website: Optional[str] = None
    location: Optional[Location] = None
    services: List[str] = []
    operating_hours: Optional[str] = None
    is_pinned: bool = False
    is_demo: bool = False
    is_real: bool = False
    rating: Optional[float] = None
    total_ratings: Optional[int] = None
    admin_id: Optional[str] = None
    place_id: Optional[str] = None
    distance: Optional[str] = None
    created_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Hospital(BaseModel):
    """Hospital model"""
    name: str
    address: str
    phone: str
    email: str
    website: Optional[str] = None
    location: Optional[Location] = None
    services: List[str] = []
    operating_hours: Optional[str] = None

# ==================== PRESCRIPTION MODELS ====================

class PrescriptionResponse(BaseModel):
    id: str
    patient_id: str
    patient_name: str
    file_name: str
    file_type: str
    file_size: int
    file_path: Optional[str] = None  # ← Make optional (legacy field)
    file_url: Optional[str] = None   # ← Cloudinary URL
    cloudinary_public_id: Optional[str] = None  # ← Cloudinary ID
    file_format: Optional[str] = None  # ← Format from Cloudinary
    doctor_name: Optional[str] = None
    date_prescribed: Optional[str] = None
    notes: Optional[str] = None
    uploaded_at: datetime
    ocr_processed: bool = False
    ai_processed: bool = False
    extracted_text: Optional[str] = None
    summary: Optional[str] = None
    medications: List[Dict[str, Any]] = []

    class Config:
        from_attributes = True
class Medication(BaseModel):
    """Medication model"""
    name: str
    dosage: str
    frequency: str
    duration: str
    instructions: Optional[str] = None
    times: List[str] = []

class PrescriptionUpload(BaseModel):
    """Prescription upload metadata"""
    doctor_name: Optional[str] = None
    date_prescribed: Optional[str] = None
    notes: Optional[str] = None

class PrescriptionResponse(BaseModel):
    """Prescription response model"""
    id: str
    patient_id: str
    patient_name: str
    file_path: str
    file_name: str
    file_type: str
    file_size: int
    doctor_name: Optional[str] = None
    date_prescribed: Optional[str] = None
    notes: Optional[str] = None
    uploaded_at: datetime
    ocr_processed: bool = False
    ai_processed: bool = False
    summary: Optional[str] = None
    medications: List[Dict[str, Any]] = []
    extracted_text: Optional[str] = None
    processed_at: Optional[datetime] = None
    extracted_at: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Prescription(BaseModel):
    """Prescription model"""
    patient_id: str
    patient_name: str
    file_path: str
    file_name: str
    file_type: str
    doctor_name: Optional[str] = None
    date_prescribed: Optional[str] = None
    notes: Optional[str] = None

# ==================== AI & PRESCRIPTION CHAT MODELS ====================

class AIPromptRequest(BaseModel):
    """AI prompt request for prescription chat"""
    prompt: str = Field(..., min_length=1, max_length=500)

class PrescriptionChatResponse(BaseModel):
    """Prescription AI chat response"""
    message: str
    user_message: str
    ai_response: str
    prescription_id: str
    created_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# ==================== APPOINTMENT MODELS ====================
class AppointmentFormData(BaseModel):
    """Appointment form data with hospital selection"""
    hospital_id: str  # ✅ REQUIRED: Must select a hospital
    symptoms: str
    preferred_date: str
    preferred_time: str
    appointment_type: str = "General Consultation"
    reason_for_visit: str
    medical_history: Optional[str] = None
    current_medications: Optional[str] = None
    allergies: Optional[str] = None
    insurance_info: Optional[str] = None
class AppointmentCreate(BaseModel):
    """Create appointment model (legacy support)"""
    hospital_id: str
    symptoms: str = Field(..., min_length=10, max_length=1000)
    preferred_date: str
    preferred_time: str
    appointment_type: Optional[str] = "general"

    @validator('preferred_date')
    def validate_date(cls, v):
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError('Invalid date format. Use YYYY-MM-DD')
        return v

class AppointmentUpdate(BaseModel):
    """Update appointment model"""
    symptoms: Optional[str] = None
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    status: Optional[str] = None
    doctor_notes: Optional[str] = None

class AppointmentResponse(BaseModel):
    """Appointment response model"""
    id: str
    patient_id: str
    patient_name: str
    patient_phone: Optional[str] = None
    patient_email: Optional[str] = None
    hospital_id: str
    hospital_name: str
    
    # Basic appointment info
    symptoms: str
    preferred_date: str
    preferred_time: str
    appointment_type: str = "general"
    
    # Detailed form data
    reason_for_visit: Optional[str] = None
    medical_history: Optional[str] = None
    current_medications: Optional[str] = None
    allergies: Optional[str] = None
    insurance_info: Optional[str] = None
    
    # Status and assignment
    status: str = "pending"
    doctor_id: Optional[str] = None
    doctor_name: Optional[str] = None
    scheduled_date: Optional[str] = None
    scheduled_time: Optional[str] = None
    
    # Notes
    notes: Optional[str] = None
    doctor_notes: Optional[str] = None
    admin_notes: Optional[str] = None
    
    # Cancellation info
    cancellation_reason: Optional[str] = None
    cancelled_by: Optional[str] = None
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    cancelled_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    rescheduled_at: Optional[datetime] = None
    reschedule_reason: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Appointment(BaseModel):
    """Appointment model"""
    patient_id: str
    patient_name: str
    hospital_id: str
    hospital_name: str
    symptoms: str
    preferred_date: str
    preferred_time: str
    status: str = "pending"

class DoctorAssignment(BaseModel):
    """Doctor assignment to appointment"""
    doctor_id: str
    scheduled_date: Optional[str] = None
    scheduled_time: Optional[str] = None
    notes: Optional[str] = None

# ==================== HOSPITAL TRACKING MODELS ====================

class NearbyHospitalsRequest(BaseModel):
    """Request for nearby hospitals"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    radius: int = Field(default=10, ge=1, le=50, description="Search radius in kilometers")

class NearbyHospitalsResponse(BaseModel):
    """Response for nearby hospitals"""
    hospitals: List[HospitalResponse]
    total: int
    center: Dict[str, float]
    radius_km: int

# ==================== CHAT MODELS ====================

class ChatMessageCreate(BaseModel):
    """Create chat message model"""
    receiver_id: str
    message: str = Field(..., min_length=1, max_length=2000)
    message_type: Optional[str] = "text"

class ChatMessage(BaseModel):
    """Chat message model"""
    sender_id: str
    sender_name: str
    sender_role: str
    receiver_id: str
    receiver_name: str
    receiver_role: str
    message: str
    message_type: str = "text"
    appointment_id: Optional[str] = None
    read: bool = False
    created_at: datetime
    read_at: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# ==================== MEDICATION REMINDER MODELS ====================

class MedicationReminderCreate(BaseModel):
    """Create medication reminder model"""
    medication_name: str = Field(..., min_length=2, max_length=200)
    dosage: str
    frequency: str
    times: List[str] = []
    start_date: str
    duration_days: int = Field(..., ge=1, le=365)
    instructions: Optional[str] = None
    prescription_id: Optional[str] = None

    @validator('start_date')
    def validate_start_date(cls, v):
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError('Invalid date format. Use YYYY-MM-DD')
        return v

    @validator('times')
    def validate_times(cls, v):
        for time_str in v:
            try:
                datetime.strptime(time_str, "%H:%M")
            except ValueError:
                raise ValueError(f'Invalid time format: {time_str}. Use HH:MM')
        return v

class MedicationReminder(BaseModel):
    """Medication reminder model"""
    patient_id: str
    medication_name: str
    dosage: str
    frequency: str
    times: List[str]
    start_date: str
    duration_days: int
    instructions: Optional[str] = None
    prescription_id: Optional[str] = None
    active: bool = True
    created_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# ==================== HEALTH VITALS MODELS ====================

class HealthVitalsCreate(BaseModel):
    """Create health vitals model"""
    heart_rate: Optional[int] = Field(None, ge=30, le=250)
    blood_pressure_systolic: Optional[int] = Field(None, ge=50, le=300)
    blood_pressure_diastolic: Optional[int] = Field(None, ge=30, le=200)
    temperature: Optional[float] = Field(None, ge=35.0, le=45.0)
    oxygen_saturation: Optional[int] = Field(None, ge=0, le=100)
    blood_sugar: Optional[int] = Field(None, ge=0, le=600)
    weight: Optional[float] = Field(None, ge=1.0, le=500.0)
    height: Optional[float] = Field(None, ge=30.0, le=300.0)
    steps: Optional[int] = Field(None, ge=0)
    sleep_hours: Optional[float] = Field(None, ge=0.0, le=24.0)
    notes: Optional[str] = None
    source: Optional[str] = "manual"  # manual, fitbit, google_fit

class HealthVitals(BaseModel):
    """Health vitals model"""
    patient_id: str
    heart_rate: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    temperature: Optional[float] = None
    oxygen_saturation: Optional[int] = None
    blood_sugar: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    steps: Optional[int] = None
    sleep_hours: Optional[float] = None
    notes: Optional[str] = None
    source: str = "manual"
    recorded_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# ==================== NOTIFICATION MODELS ====================

class NotificationResponse(BaseModel):
    """Notification response model"""
    id: str
    user_id: str
    type: str
    title: str
    message: str
    data: Dict[str, Any] = {}
    read: bool = False
    created_at: datetime
    read_at: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Notification(BaseModel):
    """Notification model"""
    user_id: str
    type: str
    title: str
    message: str
    data: Dict[str, Any] = {}
    read: bool = False
    created_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# ==================== ANALYTICS MODELS ====================

class HealthTrend(BaseModel):
    """Health trend data point"""
    date: str
    value: float

class HealthTrendsResponse(BaseModel):
    """Health trends response"""
    heart_rate: List[HealthTrend] = []
    blood_pressure_systolic: List[HealthTrend] = []
    blood_pressure_diastolic: List[HealthTrend] = []
    steps: List[HealthTrend] = []
    sleep_hours: List[HealthTrend] = []

class DashboardStats(BaseModel):
    """Dashboard statistics"""
    total_appointments: int = 0
    pending_appointments: int = 0
    confirmed_appointments: int = 0
    completed_appointments: int = 0
    cancelled_appointments: int = 0
    today_appointments: int = 0
    total_doctors: Optional[int] = 0
    total_patients: Optional[int] = 0

# ==================== QR CODE MODELS ====================

class QRCodeData(BaseModel):
    """QR code data for patient"""
    patient_id: str
    patient_name: str
    patient_email: str
    blood_group: Optional[str] = None
    emergency_contact: Optional[Dict[str, str]] = None
    date_of_birth: Optional[str] = None
    phone_number: Optional[str] = None
    generated_at: str

# ==================== SEARCH MODELS ====================

class SearchQuery(BaseModel):
    """Search query model"""
    query: str = Field(..., min_length=2, max_length=100)
    filters: Optional[Dict[str, Any]] = None
    limit: int = Field(20, ge=1, le=100)

# ==================== RESPONSE MODELS ====================

class MessageResponse(BaseModel):
    """Generic message response"""
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    status_code: int
    timestamp: str
    details: Optional[Dict[str, Any]] = None

class SuccessResponse(BaseModel):
    """Success response model"""
    success: bool = True
    message: str
    data: Optional[Any] = None

# ==================== SYSTEM MODELS ====================

class SystemInfo(BaseModel):
    """System information model"""
    system: str = "Digital Health Card"
    version: str = "2.0.0"
    statistics: Dict[str, int]
    ai_features: Dict[str, bool]
    timestamp: str

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    database: str
    version: str = "2.0.0"
    ai_configured: bool = False
    google_maps_configured: bool = False
    error: Optional[str] = None

# ==================== EXPORTS ====================
# Export all models for easy import in main.py
__all__ = [
    # Auth Models
    'UserLogin',
    'UserSignup',
    'TokenResponse',
    'UserResponse',
    
    # Profile Models
    'PatientProfile',
    'DoctorProfile',
    'AdminProfile',
    
    # Hospital Models
    'Location',
    'Hospital',
    'HospitalCreate',
    'HospitalUpdate',
    'HospitalResponse',
    'NearbyHospitalsRequest',
    'NearbyHospitalsResponse',
    
    # Prescription Models
    'Medication',
    'Prescription',
    'PrescriptionUpload',
    'PrescriptionResponse',
    'AIPromptRequest',
    'PrescriptionChatResponse',
    
    # Appointment Models
    'Appointment',
    'AppointmentCreate',
    'AppointmentFormData',
    'AppointmentUpdate',
    'AppointmentResponse',
    'DoctorAssignment',
    
    # Chat Models
    'ChatMessage',
    'ChatMessageCreate',
    
    # Medication Reminder Models
    'MedicationReminder',
    'MedicationReminderCreate',
    
    # Health Vitals Models
    'HealthVitals',
    'HealthVitalsCreate',
    
    # Notification Models
    'Notification',
    'NotificationResponse',
    
    # Analytics Models
    'HealthTrend',
    'HealthTrendsResponse',
    'DashboardStats',
    
    # QR Code Models
    'QRCodeData',
    
    # Search Models
    'SearchQuery',
    
    # Response Models
    'MessageResponse',
    'ErrorResponse',
    'SuccessResponse',
    
    # System Models
    'SystemInfo',
    'HealthCheckResponse'
]