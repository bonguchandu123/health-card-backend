from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, WebSocket, WebSocketDisconnect, Query, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import jwt
import bcrypt
import logging
import os
import io
import shutil
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import requests
import json
from config import settings
from models import (
    UserSignup, UserLogin, UserResponse, TokenResponse,
    PatientProfile, DoctorProfile, AdminProfile,
    Hospital, HospitalResponse, HospitalCreate, HospitalUpdate,
    Prescription, PrescriptionUpload, PrescriptionResponse,
    Appointment, AppointmentCreate, AppointmentResponse, AppointmentUpdate,
    ChatMessage, ChatMessageCreate,
    MedicationReminder, MedicationReminderCreate,
    HealthVitals, HealthVitalsCreate,
    NotificationResponse, DoctorAssignment,
    AppointmentFormData, AIPromptRequest, NearbyHospitalsRequest
)

# Add this import at the top with other imports
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Gemini AI
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("✅ Gemini AI configured successfully")
else:
    gemini_model = None
    logger.warning("⚠️  Gemini API key not configured")

if settings.CLOUDINARY_CLOUD_NAME and settings.CLOUDINARY_API_KEY and settings.CLOUDINARY_API_SECRET:
    cloudinary.config(
        cloud_name=settings.CLOUDINARY_CLOUD_NAME,
        api_key=settings.CLOUDINARY_API_KEY,
        api_secret=settings.CLOUDINARY_API_SECRET,
        secure=True
    )
    logger.info("✅ Cloudinary configured successfully")
else:
    logger.warning("⚠️  Cloudinary credentials not configured")
# Initialize FastAPI
app = FastAPI(
    title="Digital Health Card API",
    version="2.0.0",
    description="Production-ready Digital Health Card System with AI integration",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# MongoDB Connection
client: Optional[AsyncIOMotorClient] = None
db = None

# Collections
users_collection = None
hospitals_collection = None
prescriptions_collection = None
appointments_collection = None
chats_collection = None
medications_collection = None
vitals_collection = None
notifications_collection = None

# ==================== CONNECTION MANAGER ====================

class ConnectionManager:
    """Manages WebSocket connections for real-time communication"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_roles: Dict[str, str] = {}

    async def connect(self, websocket: WebSocket, user_id: str, role: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_roles[user_id] = role
        logger.info(f"User {user_id} ({role}) connected via WebSocket")

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            del self.user_roles[user_id]
            logger.info(f"User {user_id} disconnected from WebSocket")

    async def send_personal_message(self, message: dict, user_id: str):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
                logger.info(f"Message sent to user {user_id}")
            except Exception as e:
                logger.error(f"Error sending message to {user_id}: {str(e)}")
                self.disconnect(user_id)

    async def broadcast_to_role(self, message: dict, role: str):
        """Broadcast message to all users of a specific role"""
        for user_id, user_role in self.user_roles.items():
            if user_role == role:
                await self.send_personal_message(message, user_id)

manager = ConnectionManager()

# ==================== HELPER FUNCTIONS ====================

def create_jwt_token(user_id: str, role: str, email: str) -> str:
    """Create JWT token with enhanced payload"""
    payload = {
        "user_id": user_id,
        "role": role,
        "email": email,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(days=settings.JWT_EXPIRY_DAYS)
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

def verify_jwt_token(token: str) -> dict:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Expired token attempt")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        logger.warning("Invalid token attempt")
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Dependency to get current authenticated user"""
    token = credentials.credentials
    payload = verify_jwt_token(token)
    
    user = await users_collection.find_one({"_id": ObjectId(payload["user_id"])})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user["id"] = str(user["_id"])
    return user

def require_role(*allowed_roles: str):
    """Decorator to check user role"""
    async def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user["role"] not in allowed_roles:
            logger.warning(f"Unauthorized access attempt by {current_user['role']} to {allowed_roles} endpoint")
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required roles: {', '.join(allowed_roles)}"
            )
        return current_user
    return role_checker

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def serialize_doc(doc: dict) -> dict:
    """Convert MongoDB document to JSON-serializable dict"""
    if doc and "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    if "password" in doc:
        del doc["password"]
    return doc

async def create_notification(
    user_id: str,
    notification_type: str,
    title: str,
    message: str,
    data: dict = None
) -> dict:
    """Create and send notification"""
    notification = {
        "user_id": user_id,
        "type": notification_type,
        "title": title,
        "message": message,
        "data": data or {},
        "read": False,
        "created_at": datetime.utcnow()
    }
    
    result = await notifications_collection.insert_one(notification)
    notification["_id"] = str(result.inserted_id)
    
    # Send via WebSocket
    await manager.send_personal_message({
        "type": "notification",
        "data": notification
    }, user_id)
    
    logger.info(f"Notification created for user {user_id}: {title}")
    return notification

async def extract_text_from_prescription(file_path: str, file_type: str) -> str:
    """Extract text from prescription using Gemini Vision"""
    try:
        if not gemini_model:
            return "AI service not configured. Please set GEMINI_API_KEY in environment."
        
        # Read the file
        if file_type.startswith('image/'):
            image = Image.open(file_path)
            
            # Generate content with Gemini Vision
            prompt = """Analyze this medical prescription image and extract all information in a structured format:
            
1. Patient Information (if visible)
2. Doctor Information (name, hospital, contact)
3. Date of Prescription
4. Medications - List each with:
   - Medicine name
   - Dosage (e.g., 500mg)
   - Frequency (e.g., twice daily)
   - Duration (e.g., 7 days)
   - Special instructions
5. Diagnosis or symptoms mentioned
6. Any special warnings or instructions

Format the output clearly and comprehensively. Be precise with medical terminology."""
            
            response = gemini_model.generate_content([prompt, image])
            return response.text
        
        elif file_type == 'application/pdf':
            return "PDF extraction requires conversion to image. Please use image format (JPG/PNG) for accurate extraction."
        
        return "Unsupported file type for text extraction"
        
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return f"Error extracting text: {str(e)}"

async def summarize_prescription_with_ai(extracted_text: str) -> dict:
    """Summarize prescription and extract medications using Gemini AI"""
    try:
        if not gemini_model:
            return {
                "summary": "AI service not configured",
                "medications": []
            }
        
        prompt = f"""Based on this prescription text, provide:

1. A concise medical summary (2-3 sentences explaining the condition and treatment plan)
2. Complete list of medications in JSON format

Prescription Text:
{extracted_text}

Provide response in this EXACT JSON format (no markdown, no extra text):
{{
    "summary": "Brief summary of prescription and treatment plan",
    "medications": [
        {{
            "name": "Medicine Name",
            "dosage": "Amount (e.g., 500mg)",
            "frequency": "How often (e.g., Three times daily)",
            "duration": "How long (e.g., 7 days)",
            "instructions": "Special instructions (e.g., Take after meals)",
            "times": ["08:00", "14:00", "20:00"]
        }}
    ]
}}

Important: 
- For "times", suggest specific times based on frequency (e.g., three times daily = ["08:00", "14:00", "20:00"])
- Use 24-hour format for times
- If no specific medications found, return empty medications array
- Return ONLY valid JSON, no markdown formatting"""
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up the response to extract JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return {
            "summary": "Unable to parse prescription summary. Please try re-uploading a clearer image.",
            "medications": []
        }
    except Exception as e:
        logger.error(f"Error summarizing prescription: {str(e)}")
        return {
            "summary": f"Error processing prescription: {str(e)}",
            "medications": []
        }

async def chat_with_prescription_ai(prescription_data: dict, user_message: str) -> str:
    """Chat with AI about prescription details"""
    try:
        if not gemini_model:
            return "AI service not configured. Please set up Gemini API key in environment variables."
        
        context = f"""You are a helpful medical information assistant. Answer questions about this prescription:

**Prescription Summary:** {prescription_data.get('summary', 'N/A')}

**Medications:**
{json.dumps(prescription_data.get('medications', []), indent=2)}

**Extracted Details:**
{prescription_data.get('extracted_text', 'N/A')}

**Patient Question:** {user_message}

Instructions:
- Provide helpful, accurate information about the prescription
- If asked about side effects, drug interactions, or dosage changes, advise consulting the prescribing doctor
- Be clear and concise
- If information is not in the prescription, state that clearly
- Never provide medical advice or suggest changes to prescribed medications

Answer the patient's question now:"""
        
        response = gemini_model.generate_content(context)
        return response.text
        
    except Exception as e:
        logger.error(f"Error in AI chat: {str(e)}")
        return "I'm having trouble processing your question right now. Please try again or consult your doctor for medical advice."

async def get_nearby_hospitals_from_google(latitude: float, longitude: float, radius: int) -> List[dict]:
    """Fetch nearby hospitals from Google Places API"""
    try:
        if not settings.GOOGLE_MAPS_API_KEY:
            logger.warning("Google Maps API key not configured")
            return []
        
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{latitude},{longitude}",
            "radius": radius * 1000,  # Convert km to meters
            "type": "hospital",
            "key": settings.GOOGLE_MAPS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get("status") != "OK":
            logger.error(f"Google Places API error: {data.get('status')}")
            return []
        
        hospitals = []
        for place in data.get("results", []):
            hospital = {
                "name": place.get("name"),
                "address": place.get("vicinity"),
                "location": {
                    "type": "Point",
                    "coordinates": [
                        place["geometry"]["location"]["lng"],
                        place["geometry"]["location"]["lat"]
                    ]
                },
                "rating": place.get("rating"),
                "total_ratings": place.get("user_ratings_total"),
                "is_open": place.get("opening_hours", {}).get("open_now"),
                "place_id": place.get("place_id"),
                "is_real": True,
                "phone": "Contact via Google Maps",
                "services": ["Emergency Care", "General Medicine"]
            }
            hospitals.append(hospital)
        
        return hospitals
        
    except Exception as e:
        logger.error(f"Error fetching nearby hospitals: {str(e)}")
        return []


async def upload_to_cloudinary(file: UploadFile, folder: str, resource_type: str = "auto") -> dict:
    """Upload file to Cloudinary and return URL and metadata"""
    try:
        # Read file content
        file_content = await file.read()
        
        # Generate unique filename WITHOUT extension
        timestamp = int(datetime.utcnow().timestamp())
        filename_without_ext = os.path.splitext(file.filename)[0]  # Remove extension
        unique_filename = f"{timestamp}_{filename_without_ext}"
        
        # Prepare upload parameters
        upload_params = {
            "folder": folder,
            "public_id": unique_filename,
            "resource_type": resource_type,
            "overwrite": True,
            "invalidate": True
        }
        
        # For images, preserve the format
        if resource_type == "image":
            file_ext = os.path.splitext(file.filename)[1].replace('.', '')  # Get extension without dot
            if file_ext.lower() in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                upload_params["format"] = file_ext.lower()
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            file_content,
            **upload_params
        )
        
        # Reset file pointer for potential reuse
        await file.seek(0)
        
        return {
            "url": upload_result["secure_url"],
            "public_id": upload_result["public_id"],
            "format": upload_result["format"],
            "resource_type": upload_result["resource_type"],
            "bytes": upload_result["bytes"],
            "width": upload_result.get("width"),
            "height": upload_result.get("height"),
            "created_at": upload_result["created_at"]
        }
    except Exception as e:
        logger.error(f"Cloudinary upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
async def delete_from_cloudinary(public_id: str, resource_type: str = "image") -> bool:
    """Delete file from Cloudinary"""
    try:
        result = cloudinary.uploader.destroy(public_id, resource_type=resource_type)
        return result.get("result") == "ok"
    except Exception as e:
        logger.error(f"Cloudinary deletion error: {str(e)}")
        return False

async def extract_text_from_cloudinary_image(image_url: str) -> str:
    """Extract text from Cloudinary-hosted image using Gemini Vision"""
    try:
        if not gemini_model:
            return "AI service not configured. Please set GEMINI_API_KEY in environment."
        
        # Download image from Cloudinary URL
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Open image from bytes
        image = Image.open(io.BytesIO(response.content))
        
        # Generate content with Gemini Vision
        prompt = """Analyze this medical prescription image and extract all information in a structured format:
        
1. Patient Information (if visible)
2. Doctor Information (name, hospital, contact)
3. Date of Prescription
4. Medications - List each with:
   - Medicine name
   - Dosage (e.g., 500mg)
   - Frequency (e.g., twice daily)
   - Duration (e.g., 7 days)
   - Special instructions
5. Diagnosis or symptoms mentioned
6. Any special warnings or instructions

Format the output clearly and comprehensively. Be precise with medical terminology."""
        
        response = gemini_model.generate_content([prompt, image])
        return response.text
        
    except Exception as e:
        logger.error(f"Error extracting text from Cloudinary image: {str(e)}")
        return f"Error extracting text: {str(e)}"
# ==================== STARTUP & SHUTDOWN ====================

async def create_medication_seed_data():
    """Create seed data for medications feature demo"""
    try:
        # Find demo patient
        demo_patient = await users_collection.find_one({"email": "patient@demo.com"})
        if not demo_patient:
            logger.warning("Demo patient not found, skipping medication seed data")
            return
        
        patient_id = str(demo_patient["_id"])
        
        # Check if seed data already exists
        existing_meds = await medications_collection.count_documents({"patient_id": patient_id})
        if existing_meds > 0:
            logger.info("Medication seed data already exists")
            return
        
        # Get current date
        from datetime import date
        today = date.today()
        
        # ==================== SEED MEDICATIONS ====================
        medications = [
            {
                "patient_id": patient_id,
                "prescription_id": None,
                "medication_name": "Amoxicillin",
                "dosage": "500mg",
                "frequency": "Three times daily",
                "times": ["08:00", "14:00", "20:00"],
                "start_date": (today - timedelta(days=5)).isoformat(),
                "duration_days": 10,
                "instructions": "Take with food. Complete the full course even if you feel better.",
                "active": True,
                "created_at": datetime.utcnow() - timedelta(days=5)
            },
            {
                "patient_id": patient_id,
                "prescription_id": None,
                "medication_name": "Metformin",
                "dosage": "850mg",
                "frequency": "Twice daily",
                "times": ["08:00", "20:00"],
                "start_date": (today - timedelta(days=30)).isoformat(),
                "duration_days": 90,
                "instructions": "Take with meals. Monitor blood sugar levels regularly.",
                "active": True,
                "created_at": datetime.utcnow() - timedelta(days=30)
            },
            {
                "patient_id": patient_id,
                "prescription_id": None,
                "medication_name": "Lisinopril",
                "dosage": "10mg",
                "frequency": "Once daily",
                "times": ["08:00"],
                "start_date": (today - timedelta(days=60)).isoformat(),
                "duration_days": 180,
                "instructions": "Take in the morning. Monitor blood pressure daily.",
                "active": True,
                "created_at": datetime.utcnow() - timedelta(days=60)
            },
            {
                "patient_id": patient_id,
                "prescription_id": None,
                "medication_name": "Vitamin D3",
                "dosage": "2000 IU",
                "frequency": "Once daily",
                "times": ["09:00"],
                "start_date": (today - timedelta(days=45)).isoformat(),
                "duration_days": 365,
                "instructions": "Take with breakfast for better absorption.",
                "active": True,
                "created_at": datetime.utcnow() - timedelta(days=45)
            },
            {
                "patient_id": patient_id,
                "prescription_id": None,
                "medication_name": "Omeprazole",
                "dosage": "20mg",
                "frequency": "Once daily",
                "times": ["07:30"],
                "start_date": (today - timedelta(days=15)).isoformat(),
                "duration_days": 30,
                "instructions": "Take 30 minutes before breakfast on an empty stomach.",
                "active": True,
                "created_at": datetime.utcnow() - timedelta(days=15)
            },
            {
                "patient_id": patient_id,
                "prescription_id": None,
                "medication_name": "Atorvastatin",
                "dosage": "20mg",
                "frequency": "Once daily",
                "times": ["21:00"],
                "start_date": (today - timedelta(days=90)).isoformat(),
                "duration_days": 365,
                "instructions": "Take in the evening. Avoid grapefruit juice.",
                "active": True,
                "created_at": datetime.utcnow() - timedelta(days=90)
            }
        ]
        
        # Insert medications
        result = await medications_collection.insert_many(medications)
        logger.info(f"✅ Created {len(result.inserted_ids)} medication reminders")
        
        # ==================== CREATE MEDICATION LOGS (HISTORY) ====================
        # Create realistic medication logs for the past 7 days
        medication_logs = []
        
        for med in medications:
            med_id = str(result.inserted_ids[medications.index(med)])
            
            # Create logs for past 7 days
            for days_ago in range(7, 0, -1):
                log_date = today - timedelta(days=days_ago)
                
                # Simulate ~85% adherence rate
                for time in med["times"]:
                    # 85% chance of taking medication
                    import random
                    if random.random() < 0.85:
                        log_time = datetime.combine(log_date, datetime.strptime(time, "%H:%M").time())
                        
                        medication_logs.append({
                            "reminder_id": med_id,
                            "patient_id": patient_id,
                            "medication_name": med["medication_name"],
                            "dosage": med["dosage"],
                            "taken_at": log_time.isoformat(),
                            "created_at": log_time
                        })
        
        # Insert logs
        if medication_logs:
            await db["medication_logs"].insert_many(medication_logs)
            logger.info(f"✅ Created {len(medication_logs)} medication logs (history)")
        
        # ==================== CREATE TODAY'S LOGS ====================
        # Mark some of today's medications as taken
        today_logs = []
        current_time = datetime.now()
        
        for med in medications:
            med_id = str(result.inserted_ids[medications.index(med)])
            
            for time in med["times"]:
                scheduled_time = datetime.strptime(time, "%H:%M").time()
                
                # If scheduled time has passed, 70% chance it's taken
                if scheduled_time < current_time.time() and random.random() < 0.70:
                    log_datetime = datetime.combine(today, scheduled_time)
                    
                    today_logs.append({
                        "reminder_id": med_id,
                        "patient_id": patient_id,
                        "medication_name": med["medication_name"],
                        "dosage": med["dosage"],
                        "taken_at": log_datetime.isoformat(),
                        "created_at": log_datetime
                    })
        
        if today_logs:
            await db["medication_logs"].insert_many(today_logs)
            logger.info(f"✅ Created {len(today_logs)} logs for today")
        
        logger.info("=" * 60)
        logger.info("💊 MEDICATION SEED DATA CREATED")
        logger.info("=" * 60)
        logger.info("Active Medications: 6")
        logger.info("  - Amoxicillin (Antibiotic) - 3x daily")
        logger.info("  - Metformin (Diabetes) - 2x daily")
        logger.info("  - Lisinopril (Blood Pressure) - 1x daily")
        logger.info("  - Vitamin D3 (Supplement) - 1x daily")
        logger.info("  - Omeprazole (Acid Reflux) - 1x daily")
        logger.info("  - Atorvastatin (Cholesterol) - 1x daily")
        logger.info(f"Total Logs Created: {len(medication_logs) + len(today_logs)}")
        logger.info("Adherence Rate: ~85%")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error creating medication seed data: {str(e)}")
@app.on_event("startup")
async def startup_event():
    """Initialize database connection and create demo data"""
    global client, db
    global users_collection, hospitals_collection, prescriptions_collection
    global appointments_collection, chats_collection, medications_collection
    global vitals_collection, notifications_collection
    
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        db = client[settings.DATABASE_NAME]
        
        # Initialize collections
        users_collection = db["users"]
        hospitals_collection = db["hospitals"]
        prescriptions_collection = db["prescriptions"]
        appointments_collection = db["appointments"]
        chats_collection = db["chats"]
        medications_collection = db["medications"]
        vitals_collection = db["vitals"]
        notifications_collection = db["notifications"]
        
        # Create indexes for better performance
        await users_collection.create_index("email", unique=True)
        await appointments_collection.create_index([("patient_id", 1), ("created_at", -1)])
        await prescriptions_collection.create_index([("patient_id", 1), ("uploaded_at", -1)])
        await chats_collection.create_index([("sender_id", 1), ("receiver_id", 1)])
        await notifications_collection.create_index([("user_id", 1), ("created_at", -1)])
     
        await create_demo_data()
        await create_medication_seed_data()
        
        logger.info("✅ Application startup complete")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {str(e)}")
        raise
@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection"""
    if client:
        client.close()
        logger.info("Database connection closed")

async def create_demo_data():
    """Create CarePlus Hospital demo with admin and doctor"""
    try:
        # Check if demo data exists
        existing_hospital = await hospitals_collection.find_one({"name": "CarePlus Hospital"})
        if existing_hospital:
            logger.info("Demo CarePlus Hospital already exists")
            return

        # Create CarePlus Hospital (Pinned Demo Hospital)
        careplus_hospital = {
            "name": "CarePlus Hospital",
            "address": "123 Healthcare Avenue, Medical District, New York, NY 10001",
            "phone": "+1-555-CARE-001",
            "email": "contact@careplu hospital.com",
            "website": "https://careplus-hospital.com",
            "location": {
                "type": "Point",
                "coordinates": [-74.0060, 40.7128]
            },
            "services": ["Emergency Care 24/7", "General Medicine", "Surgery", "Pediatrics", "Cardiology", "Orthopedics"],
            "is_pinned": True,
            "is_demo": True,
            "rating": 4.8,
            "total_ratings": 1250,
            "created_at": datetime.utcnow(),
            "operating_hours": "24/7 Emergency Services"
        }
        hospital_result = await hospitals_collection.insert_one(careplus_hospital)
        hospital_id = str(hospital_result.inserted_id)
        logger.info(f"✅ Created CarePlus Hospital: {hospital_id}")

        # Create CarePlus Admin
        admin_data = {
            "full_name": "Dr. Sarah Johnson - CarePlus Admin",
            "email": "admin@careplus.com",
            "password": hash_password("CareAdmin@123"),
            "role": "admin",
            "hospital_name": "CarePlus Hospital",
            "hospital_address": "123 Healthcare Avenue, Medical District",
            "hospital_id": hospital_id,
            "phone": "+1-555-CARE-100",
            "created_at": datetime.utcnow(),
            "is_active": True
        }
        admin_result = await users_collection.insert_one(admin_data)
        logger.info(f"✅ Created CarePlus Admin: admin@careplus.com")

        # Create CarePlus Doctor
        doctor_data = {
            "full_name": "Dr. Michael Chen",
            "email": "doctor@careplus.com",
            "password": hash_password("CareDoc@123"),
            "role": "doctor",
            "hospital_id": hospital_id,
            "specialization": "General Physician & Family Medicine",
            "license_number": "CARE-DOC-12345",
            "phone": "+1-555-CARE-200",
            "experience_years": 12,
            "created_at": datetime.utcnow(),
            "is_active": True,
            "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        }
        doctor_result = await users_collection.insert_one(doctor_data)
        logger.info(f"✅ Created CarePlus Doctor: doctor@careplus.com")

        # Create Demo Patient
        patient_data = {
            "full_name": "John Doe",
            "email": "patient@demo.com",
            "password": hash_password("Patient@123"),
            "role": "patient",
            "phone_number": "+1-555-PATIENT-01",
            "address": "456 Patient Street, Residential Area, NY 10002",
            "date_of_birth": "1990-05-15",
            "blood_group": "O+",
            "gender": "Male",
            "emergency_contact": {
                "name": "Jane Doe",
                "relationship": "Spouse",
                "phone": "+1-555-EMERGENCY-01"
            },
            "created_at": datetime.utcnow(),
            "is_active": True
        }
        patient_result = await users_collection.insert_one(patient_data)
        logger.info(f"✅ Created Demo Patient: patient@demo.com")

        logger.info("=" * 60)
        logger.info("🏥 DEMO CREDENTIALS")
        logger.info("=" * 60)
        logger.info("Admin:   admin@careplus.com / CareAdmin@123")
        logger.info("Doctor:  doctor@careplus.com / CareDoc@123")
        logger.info("Patient: patient@demo.com / Patient@123")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error creating demo data: {str(e)}")

# ==================== AUTH ENDPOINTS ====================

@app.get("/api/v1/public/hospitals", tags=["Public"])
async def get_public_hospitals():
    """
    Get list of hospitals for doctor signup (no auth required)
    Only shows hospitals created by admins who have logged in at least once
    """
    try:
        # Find all admins who have logged in (have last_login field)
        active_admins = await users_collection.find({
            "role": "admin",
            "is_active": True,
            "last_login": {"$exists": True}  # Only admins who have logged in
        }).to_list(1000)
        
        admin_hospital_ids = [admin.get("hospital_id") for admin in active_admins if admin.get("hospital_id")]
        
        # Get hospitals belonging to these admins + pinned CarePlus
        hospitals = await hospitals_collection.find({
            "$or": [
                {"_id": {"$in": [ObjectId(hid) for hid in admin_hospital_ids]}},
                {"is_pinned": True}  # Always include CarePlus demo hospital
            ]
        }).sort([("is_pinned", -1), ("name", 1)]).to_list(100)
        
        logger.info(f"Found {len(hospitals)} hospitals for doctor signup (from {len(active_admins)} active admins)")
        
        return [serialize_doc(h) for h in hospitals]
        
    except Exception as e:
        logger.error(f"Get public hospitals error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch hospitals")
@app.post("/api/v1/auth/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED, tags=["Authentication"])
async def signup(user_data: UserSignup):
    """Register a new user (Patient, Doctor, or Admin)"""
    try:
        # Log incoming signup request
        logger.info(f"Signup attempt - Role: {user_data.role}, Email: {user_data.email}")
        
        existing_user = await users_collection.find_one({"email": user_data.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        user_dict = user_data.dict(exclude_unset=True)
        user_dict["password"] = hash_password(user_data.password)
        user_dict["created_at"] = datetime.utcnow()
        user_dict["is_active"] = True

        # ========== ADMIN SIGNUP ==========
        if user_data.role == "admin":
            if not user_data.hospital_name or not user_data.hospital_address:
                raise HTTPException(status_code=400, detail="Hospital name and address required for admins")
            
            # Check if hospital already exists
            existing_hospital = await hospitals_collection.find_one({
                "name": user_data.hospital_name
            })
            
            if existing_hospital:
                hospital_id = str(existing_hospital["_id"])
                logger.info(f"Admin joining existing hospital: {user_data.hospital_name}")
            else:
                # Create new hospital (admin_id will be set after user creation)
                hospital_data = {
                    "name": user_data.hospital_name,
                    "address": user_data.hospital_address,
                    "phone": user_data.phone if user_data.phone else "",
                    "email": user_data.email,
                    "is_pinned": False,
                    "is_demo": False,
                    "is_active": True,
                    "created_at": datetime.utcnow(),
                    "services": ["Emergency Care", "General Medicine", "Surgery"],
                    "rating": 4.5,
                    "total_ratings": 0
                }
                
                hospital_result = await hospitals_collection.insert_one(hospital_data)
                hospital_id = str(hospital_result.inserted_id)
                logger.info(f"✅ Hospital created: {user_data.hospital_name} (ID: {hospital_id})")
            
            user_dict["hospital_id"] = hospital_id
            user_dict["hospital_name"] = user_data.hospital_name

        # ========== DOCTOR SIGNUP ==========
        elif user_data.role == "doctor":
            logger.info(f"Doctor signup - Hospital ID: {user_data.hospital_id}")
            
            if not user_data.hospital_id:
                raise HTTPException(status_code=400, detail="Hospital ID required for doctors")
            
            if not user_data.specialization:
                raise HTTPException(status_code=400, detail="Specialization required for doctors")
            
            if not user_data.license_number:
                raise HTTPException(status_code=400, detail="License number required for doctors")
            
            # Validate hospital exists
            try:
                hospital = await hospitals_collection.find_one({"_id": ObjectId(user_data.hospital_id)})
                if not hospital:
                    raise HTTPException(status_code=404, detail="Hospital not found")
                
                logger.info(f"Hospital found: {hospital.get('name')}")
                
                # Check if hospital has an admin who logged in (unless it's pinned demo hospital)
                if not hospital.get("is_pinned"):
                    admin = await users_collection.find_one({
                        "hospital_id": user_data.hospital_id,
                        "role": "admin",
                        "is_active": True,
                        "last_login": {"$exists": True}
                    })
                    
                    if not admin:
                        raise HTTPException(
                            status_code=400, 
                            detail="This hospital's admin must log in first before doctors can join"
                        )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Hospital validation error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid hospital ID: {str(e)}")

        # ========== PATIENT SIGNUP ==========
        elif user_data.role == "patient":
            if not user_data.phone_number:
                raise HTTPException(status_code=400, detail="Phone number required for patients")
            if not user_data.address:
                raise HTTPException(status_code=400, detail="Address required for patients")
            if not user_data.date_of_birth:
                raise HTTPException(status_code=400, detail="Date of birth required for patients")

        # Insert user
        result = await users_collection.insert_one(user_dict)
        user_id = str(result.inserted_id)
        
        # ✅ CRITICAL FIX: Update hospital with admin_id AFTER user creation
        if user_data.role == "admin" and user_dict.get("hospital_id"):
            # Set last_login immediately so hospital appears in public listings
            await users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"last_login": datetime.utcnow()}}
            )
            
            await hospitals_collection.update_one(
                {"_id": ObjectId(user_dict["hospital_id"])},
                {"$set": {"admin_id": user_id}}
            )
            logger.info(f"✅ Hospital {user_dict['hospital_id']} linked to admin {user_id} with last_login set")

        # Create JWT token
        token = create_jwt_token(user_id, user_data.role, user_data.email)

        logger.info(f"✅ {user_data.role.upper()} registered successfully: {user_data.email}")

        return TokenResponse(
            access_token=token,
            token_type="bearer",
            user_id=user_id,
            role=user_data.role,
            email=user_data.email
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Signup error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")
@app.post("/api/v1/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(credentials: UserLogin):
    """Login with email and password"""
    try:
        user = await users_collection.find_one({"email": credentials.email})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        if not user.get("is_active", True):
            raise HTTPException(status_code=403, detail="Account is deactivated")

        if not verify_password(credentials.password, user["password"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        user_id = str(user["_id"])
        token = create_jwt_token(user_id, user["role"], user["email"])

        # ✅ IMPORTANT: Set last_login for admins to make their hospitals visible
        await users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"last_login": datetime.utcnow()}}
        )

        logger.info(f"✅ User logged in: {credentials.email} (Role: {user['role']})")

        return TokenResponse(
            access_token=token,
            token_type="bearer",
            user_id=user_id,
            role=user["role"],
            email=user["email"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")
@app.get("/api/v1/auth/me", response_model=UserResponse, tags=["Authentication"])
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return serialize_doc(current_user)

@app.post("/api/v1/auth/refresh", response_model=TokenResponse, tags=["Authentication"])
async def refresh_token(current_user: dict = Depends(get_current_user)):
    """Refresh JWT token"""
    new_token = create_jwt_token(
        str(current_user["_id"]),
        current_user["role"],
        current_user["email"]
    )
    
    return TokenResponse(
        access_token=new_token,
        token_type="bearer",
        user_id=str(current_user["_id"]),
        role=current_user["role"],
        email=current_user["email"]
    )

# ==================== PATIENT ENDPOINTS ====================

@app.get("/api/v1/patient/profile", tags=["Patient"])
async def get_patient_profile(current_user: dict = Depends(require_role("patient"))):
    """Get patient profile details"""
    return serialize_doc(current_user)

@app.put("/api/v1/patient/profile", tags=["Patient"])
async def update_patient_profile(
    profile: PatientProfile,
    current_user: dict = Depends(require_role("patient"))
):
    """Update patient profile information"""
    try:
        update_data = profile.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()

        await users_collection.update_one(
            {"_id": ObjectId(current_user["_id"])},
            {"$set": update_data}
        )

        logger.info(f"Patient profile updated: {current_user['email']}")
        return {"message": "Profile updated successfully", "updated_fields": list(update_data.keys())}

    except Exception as e:
        logger.error(f"Profile update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Profile update failed")

@app.post("/api/v1/patient/prescriptions/upload", status_code=status.HTTP_201_CREATED, tags=["Patient"])
async def upload_prescription(
    file: UploadFile = File(...),
    doctor_name: Optional[str] = None,
    date_prescribed: Optional[str] = None,
    notes: Optional[str] = None,
    current_user: dict = Depends(require_role("patient"))
):
    """Upload a prescription to Cloudinary (PDF or Image)"""
    try:
        # Validate file type
        allowed_types = ["application/pdf", "image/jpeg", "image/png", "image/jpg"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only PDF and images (JPG, PNG) allowed"
            )
        
        # DEBUG: Log file details
        logger.info(f"=== UPLOAD DEBUG ===")
        logger.info(f"Filename: {file.filename}")
        logger.info(f"Content Type: {file.content_type}")
        
        # Determine correct resource type for Cloudinary
        if file.content_type.startswith('image/'):
            resource_type = "image"
            logger.info(f"Using resource_type: image")
        elif file.content_type == "application/pdf":
            resource_type = "raw"
            logger.info(f"Using resource_type: raw")
        else:
            resource_type = "auto"
            logger.info(f"Using resource_type: auto")
        
        # Upload to Cloudinary
        cloudinary_data = await upload_to_cloudinary(
            file=file,
            folder=f"prescriptions/{current_user['id']}",
            resource_type=resource_type
        )
        
        logger.info(f"Cloudinary URL: {cloudinary_data['url']}")
        logger.info(f"Cloudinary format: {cloudinary_data['format']}")
        logger.info(f"==================")
        
        # Create prescription document
        prescription_data = {
            "patient_id": current_user["id"],
            "patient_name": current_user["full_name"],
            
            # Cloudinary data
            "file_url": cloudinary_data["url"],
            "cloudinary_public_id": cloudinary_data["public_id"],
            "file_name": file.filename,
            "file_type": file.content_type,
            "file_size": cloudinary_data["bytes"],
            "file_format": cloudinary_data["format"],
            
            # Metadata
            "doctor_name": doctor_name,
            "date_prescribed": date_prescribed,
            "notes": notes,
            "uploaded_at": datetime.utcnow(),
            
            # Processing status
            "ocr_processed": False,
            "ai_processed": False,
            "summary": None,
            "medications": [],
            "extracted_text": None
        }
        
        result = await prescriptions_collection.insert_one(prescription_data)
        prescription_data["_id"] = str(result.inserted_id)
        
        logger.info(f"Prescription uploaded to Cloudinary by {current_user['email']}: {file.filename}")
        
        return serialize_doc(prescription_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prescription upload error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prescription upload failed")


@app.get("/api/v1/patient/prescriptions", tags=["Patient"])
async def get_patient_prescriptions(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(require_role("patient"))
):
    """Get all prescriptions for the current patient"""
    try:
        prescriptions = await prescriptions_collection.find(
            {"patient_id": current_user["id"]}
        ).sort("uploaded_at", -1).skip(skip).limit(limit).to_list(limit)

        return [serialize_doc(p) for p in prescriptions]

    except Exception as e:
        logger.error(f"Get prescriptions error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch prescriptions")

@app.get("/api/v1/patient/prescriptions/{prescription_id}",tags=["Patient"])
async def get_prescription_detail(
    prescription_id: str,
    current_user: dict = Depends(require_role("patient"))
):
    """Get detailed prescription information"""
    try:
        prescription = await prescriptions_collection.find_one({
            "_id": ObjectId(prescription_id),
            "patient_id": current_user["id"]
        })
        
        if not prescription:
            raise HTTPException(status_code=404, detail="Prescription not found")

        return serialize_doc(prescription)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get prescription detail error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch prescription")


@app.post("/api/v1/patient/prescriptions/{prescription_id}/extract", tags=["Patient"])
async def extract_prescription_text(
    prescription_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(require_role("patient"))
):
    """Extract text from Cloudinary-hosted prescription using Gemini AI Vision"""
    try:
        prescription = await prescriptions_collection.find_one({
            "_id": ObjectId(prescription_id),
            "patient_id": current_user["id"]
        })
        
        if not prescription:
            raise HTTPException(status_code=404, detail="Prescription not found")
        
        if not gemini_model:
            raise HTTPException(
                status_code=503, 
                detail="AI service not configured. Please contact administrator."
            )
        
        # Check if it's a PDF
        if prescription["file_type"] == "application/pdf":
            extracted_text = "PDF text extraction requires conversion to image. Please re-upload as JPG/PNG for AI extraction."
        else:
            # Extract text from Cloudinary URL
            extracted_text = await extract_text_from_cloudinary_image(prescription["file_url"])
        
        # Update prescription with extracted text
        await prescriptions_collection.update_one(
            {"_id": ObjectId(prescription_id)},
            {"$set": {
                "extracted_text": extracted_text,
                "ocr_processed": True,
                "extracted_at": datetime.utcnow()
            }}
        )
        
        logger.info(f"Text extracted from Cloudinary prescription {prescription_id}")
        
        return {
            "message": "Text extracted successfully",
            "extracted_text": extracted_text,
            "prescription_id": prescription_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")

@app.post("/api/v1/patient/prescriptions/{prescription_id}/summarize", tags=["Patient"])
async def summarize_prescription(
    prescription_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(require_role("patient"))
):
    """Summarize prescription and extract medications using Gemini AI"""
    try:
        prescription = await prescriptions_collection.find_one({
            "_id": ObjectId(prescription_id),
            "patient_id": current_user["id"]
        })
        
        if not prescription:
            raise HTTPException(status_code=404, detail="Prescription not found")

        if not gemini_model:
            raise HTTPException(status_code=503, detail="AI service not configured. Please contact administrator.")

        # Extract text first if not already done
        if not prescription.get("extracted_text"):
            extracted_text = await extract_text_from_prescription(
                prescription["file_path"],
                prescription["file_type"]
            )
            await prescriptions_collection.update_one(
                {"_id": ObjectId(prescription_id)},
                {"$set": {"extracted_text": extracted_text, "ocr_processed": True}}
            )
        else:
            extracted_text = prescription["extracted_text"]

        # Summarize with AI
        ai_result = await summarize_prescription_with_ai(extracted_text)

        # Update prescription with AI results
        await prescriptions_collection.update_one(
            {"_id": ObjectId(prescription_id)},
            {"$set": {
                "ai_processed": True,
                "summary": ai_result["summary"],
                "medications": ai_result["medications"],
                "processed_at": datetime.utcnow()
            }}
        )

        # Create medication reminders
        for med in ai_result["medications"]:
            if med.get("times"):
                reminder_data = {
                    "patient_id": current_user["id"],
                    "prescription_id": prescription_id,
                    "medication_name": med["name"],
                    "dosage": med["dosage"],
                    "frequency": med["frequency"],
                    "times": med["times"],
                    "start_date": datetime.utcnow().date().isoformat(),
                    "duration_days": int(med["duration"].split()[0]) if med.get("duration") and "days" in med["duration"].lower() else 30,
                    "instructions": med.get("instructions", ""),
                    "active": True,
                    "created_at": datetime.utcnow()
                }
                await medications_collection.insert_one(reminder_data)

        logger.info(f"Prescription {prescription_id} summarized and medications extracted")

        return {
            "message": "Prescription summarized successfully",
            "summary": ai_result["summary"],
            "medications": ai_result["medications"],
            "reminders_created": len([m for m in ai_result["medications"] if m.get("times")])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to summarize prescription: {str(e)}")
@app.delete("/api/v1/patient/prescriptions/{prescription_id}", tags=["Patient"])
async def delete_prescription(
    prescription_id: str,
    current_user: dict = Depends(require_role("patient"))
):
    """Delete prescription and remove from Cloudinary"""
    try:
        prescription = await prescriptions_collection.find_one({
            "_id": ObjectId(prescription_id),
            "patient_id": current_user["id"]
        })
        
        if not prescription:
            raise HTTPException(status_code=404, detail="Prescription not found")
        
        # Delete from Cloudinary
        if prescription.get("cloudinary_public_id"):
            resource_type = "raw" if prescription["file_type"] == "application/pdf" else "image"
            deletion_success = await delete_from_cloudinary(
                prescription["cloudinary_public_id"],
                resource_type=resource_type
            )
            if deletion_success:
                logger.info(f"Deleted from Cloudinary: {prescription['cloudinary_public_id']}")
        
        # Delete from database
        await prescriptions_collection.delete_one({"_id": ObjectId(prescription_id)})
        
        # Delete associated medication reminders
        await medications_collection.delete_many({"prescription_id": prescription_id})
        
        logger.info(f"Prescription {prescription_id} deleted completely")
        
        return {"message": "Prescription deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete prescription error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete prescription")
@app.post("/api/v1/patient/prescriptions/{prescription_id}/chat", tags=["Patient"])
async def chat_about_prescription(
    prescription_id: str,
    prompt_request: AIPromptRequest,
    current_user: dict = Depends(require_role("patient"))
):
    """Chat with AI about prescription details"""
    try:
        prescription = await prescriptions_collection.find_one({
            "_id": ObjectId(prescription_id),
            "patient_id": current_user["id"]
        })
        
        if not prescription:
            raise HTTPException(status_code=404, detail="Prescription not found")

        if not gemini_model:
            raise HTTPException(status_code=503, detail="AI chat service not configured")

        if not prescription.get("ai_processed"):
            raise HTTPException(
                status_code=400, 
                detail="Please summarize the prescription first before chatting"
            )

        # Get AI response
        ai_response = await chat_with_prescription_ai(prescription, prompt_request.prompt)

        # Save chat message to database
        chat_message = {
            "prescription_id": prescription_id,
            "patient_id": current_user["id"],
            "user_message": prompt_request.prompt,
            "ai_response": ai_response,
            "created_at": datetime.utcnow()
        }
        await db["prescription_chats"].insert_one(chat_message)

        logger.info(f"AI chat for prescription {prescription_id}")

        return {
            "message": "AI response generated",
            "user_message": prompt_request.prompt,
            "ai_response": ai_response
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI chat error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get AI response")

@app.get("/api/v1/patient/prescriptions/{prescription_id}/chat-history", tags=["Patient"])
async def get_prescription_chat_history(
    prescription_id: str,
    current_user: dict = Depends(require_role("patient"))
):
    """Get chat history for a prescription"""
    try:
        prescription = await prescriptions_collection.find_one({
            "_id": ObjectId(prescription_id),
            "patient_id": current_user["id"]
        })
        
        if not prescription:
            raise HTTPException(status_code=404, detail="Prescription not found")

        chats = await db["prescription_chats"].find({
            "prescription_id": prescription_id,
            "patient_id": current_user["id"]
        }).sort("created_at", 1).to_list(100)

        return [serialize_doc(chat) for chat in chats]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get chat history error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch chat history")


@app.post("/api/v1/patient/hospitals/nearby", tags=["Patient"])
async def get_nearby_hospitals(
    request: NearbyHospitalsRequest,
    current_user: dict = Depends(require_role("patient"))
):
    """Get nearby hospitals using Google Maps API + ALL hospitals in DB"""
    try:
        hospitals = []
        
        # ✅ FIX: Get ALL hospitals from database first
        db_hospitals = await hospitals_collection.find({}).to_list(1000)
        
        for h in db_hospitals:
            hospital_data = serialize_doc(h)
            
            # Calculate distance if location exists
            if h.get("location") and h["location"].get("coordinates"):
                coords = h["location"]["coordinates"]
                # Calculate distance using haversine formula
                from math import radians, cos, sin, asin, sqrt
                
                lon1, lat1 = radians(coords[0]), radians(coords[1])
                lon2, lat2 = radians(request.longitude), radians(request.latitude)
                
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                distance_km = 6371 * c  # Earth radius in km
                
                if distance_km <= request.radius:
                    hospital_data["distance"] = f"{distance_km:.1f} km away"
                    hospitals.append(hospital_data)
            else:
                # Include hospitals without location (like CarePlus)
                hospital_data["distance"] = "Location not available"
                hospitals.append(hospital_data)
        
        # Fetch real hospitals from Google Maps
        if settings.GOOGLE_MAPS_API_KEY:
            real_hospitals = await get_nearby_hospitals_from_google(
                request.latitude,
                request.longitude,
                request.radius
            )
            hospitals.extend(real_hospitals)
        
        # Sort: pinned first, then by distance
        hospitals.sort(key=lambda x: (
            not x.get("is_pinned", False),
            float(x.get("distance", "999").split()[0]) if "km" in x.get("distance", "") else 999
        ))
        
        logger.info(f"Found {len(hospitals)} hospitals")
        
        return {
            "hospitals": hospitals,
            "total": len(hospitals),
            "center": {
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            "radius_km": request.radius
        }
        
    except Exception as e:
        logger.error(f"Get nearby hospitals error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch nearby hospitals")
# ============================================
# STEP 1: Add this NEW endpoint to get hospitals for booking
# ============================================

@app.get("/api/v1/patient/hospitals/for-booking", tags=["Patient"])
async def get_hospitals_for_booking(
    current_user: dict = Depends(require_role("patient"))
):
    """Get all active hospitals where patient can book appointments"""
    try:
        # Get all hospitals with active admins
        active_admins = await users_collection.find({
            "role": "admin",
            "is_active": True,
            "last_login": {"$exists": True}
        }).to_list(1000)
        
        admin_hospital_ids = [admin.get("hospital_id") for admin in active_admins if admin.get("hospital_id")]
        
        # Get hospitals (include CarePlus + admin hospitals)
        hospitals = await hospitals_collection.find({
            "$or": [
                {"_id": {"$in": [ObjectId(hid) for hid in admin_hospital_ids]}},
                {"is_pinned": True}  # Always include CarePlus
            ]
        }).sort([("is_pinned", -1), ("name", 1)]).to_list(100)
        
        # Add admin info to each hospital
        result = []
        for hospital in hospitals:
            hospital_data = serialize_doc(hospital)
            
            # Find hospital admin
            admin = await users_collection.find_one({
                "hospital_id": str(hospital["_id"]),
                "role": "admin",
                "is_active": True
            })
            
            if admin:
                hospital_data["has_admin"] = True
                hospital_data["admin_name"] = admin.get("full_name")
                hospital_data["admin_email"] = admin.get("email")
            else:
                hospital_data["has_admin"] = False
            
            # Count available doctors
            doctors_count = await users_collection.count_documents({
                "hospital_id": str(hospital["_id"]),
                "role": "doctor",
                "is_active": True
            })
            hospital_data["doctors_count"] = doctors_count
            
            result.append(hospital_data)
        
        logger.info(f"Found {len(result)} hospitals for booking")
        return result
        
    except Exception as e:
        logger.error(f"Get hospitals for booking error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch hospitals")


# ============================================
# STEP 2: REPLACE the book_appointment endpoint
# ============================================



# ============================================
# STEP 3: Add debugging endpoint
# ============================================

@app.get("/api/v1/admin/debug-appointments", tags=["Admin"])
async def debug_admin_appointments(
    current_user: dict = Depends(require_role("admin"))
):
    """Debug: Check why appointments aren't showing"""
    try:
        hospital_id = current_user.get("hospital_id")
        
        # Get admin info
        admin_info = {
            "admin_id": current_user["id"],
            "admin_email": current_user["email"],
            "admin_hospital_id": hospital_id,
            "admin_role": current_user["role"]
        }
        
        if not hospital_id:
            return {
                **admin_info,
                "error": "No hospital_id assigned to admin",
                "fix": "Admin needs to be assigned to a hospital"
            }
        
        # Get hospital details
        hospital = await hospitals_collection.find_one({"_id": ObjectId(hospital_id)})
        
        # Get ALL appointments in database
        all_appointments = await appointments_collection.find({}).to_list(1000)
        
        # Get appointments for THIS hospital
        hospital_appointments = await appointments_collection.find({
            "hospital_id": hospital_id
        }).to_list(1000)
        
        # Get appointments by hospital name (in case ID mismatch)
        appointments_by_name = await appointments_collection.find({
            "hospital_name": hospital.get("name") if hospital else None
        }).to_list(1000) if hospital else []
        
        return {
            **admin_info,
            "hospital": {
                "id": hospital_id,
                "name": hospital.get("name") if hospital else "NOT FOUND",
                "exists": hospital is not None
            },
            "appointments": {
                "total_in_database": len(all_appointments),
                "for_this_hospital_by_id": len(hospital_appointments),
                "for_this_hospital_by_name": len(appointments_by_name)
            },
            "sample_appointments": [
                {
                    "id": str(a["_id"]),
                    "patient": a.get("patient_name"),
                    "hospital_id": a.get("hospital_id"),
                    "hospital_name": a.get("hospital_name"),
                    "status": a.get("status"),
                    "matches_admin_hospital": a.get("hospital_id") == hospital_id
                }
                for a in all_appointments[:5]
            ],
            "diagnosis": {
                "has_hospital": hospital is not None,
                "has_appointments_for_hospital": len(hospital_appointments) > 0,
                "possible_issue": "Hospital ID mismatch" if len(all_appointments) > 0 and len(hospital_appointments) == 0 else "No appointments created yet"
            }
        }
        
    except Exception as e:
        logger.error(f"Debug error: {str(e)}")
        return {"error": str(e)}
@app.get("/api/v1/patient/appointments", response_model=List[AppointmentResponse], tags=["Patient"])
async def get_patient_appointments(
    status_filter: Optional[str] = Query(None, description="Filter by status: pending, confirmed, completed, cancelled"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(require_role("patient"))
):
    """Get all appointments for the current patient"""
    try:
        query = {"patient_id": current_user["id"]}
        
        if status_filter:
            query["status"] = status_filter

        appointments = await appointments_collection.find(query)\
            .sort("created_at", -1)\
            .skip(skip)\
            .limit(limit)\
            .to_list(limit)

        return [serialize_doc(a) for a in appointments]

    except Exception as e:
        logger.error(f"Get patient appointments error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch appointments")

@app.post("/api/v1/patient/appointments", response_model=AppointmentResponse, status_code=status.HTTP_201_CREATED, tags=["Patient"])
async def book_appointment(
    appointment: AppointmentFormData,
    current_user: dict = Depends(require_role("patient"))
):
    """Book appointment at selected hospital"""
    try:
        # ✅ CRITICAL: Appointment MUST have hospital_id
        if not hasattr(appointment, 'hospital_id') or not appointment.hospital_id:
            # Fallback to CarePlus if not specified
            hospital = await hospitals_collection.find_one({"is_pinned": True})
            if not hospital:
                raise HTTPException(status_code=404, detail="No hospitals available")
            hospital_id = str(hospital["_id"])
            hospital_name = hospital["name"]
            logger.warning(f"No hospital_id provided, defaulting to CarePlus: {hospital_id}")
        else:
            # Use the specified hospital
            hospital_id = appointment.hospital_id
            hospital = await hospitals_collection.find_one({"_id": ObjectId(hospital_id)})
            if not hospital:
                raise HTTPException(status_code=404, detail="Hospital not found")
            hospital_name = hospital["name"]

        # ✅ Verify hospital has an admin
        admin = await users_collection.find_one({
            "hospital_id": hospital_id,
            "role": "admin",
            "is_active": True
        })
        
        if not admin:
            raise HTTPException(
                status_code=400, 
                detail=f"Hospital {hospital_name} has no active admin. Please choose another hospital."
            )

        # Create appointment
        appointment_data = {
            "patient_id": current_user["id"],
            "patient_name": current_user["full_name"],
            "patient_phone": current_user.get("phone_number"),
            "patient_email": current_user["email"],
            "hospital_id": hospital_id,  # ✅ CORRECT hospital
            "hospital_name": hospital_name,
            
            # Form details
            "symptoms": appointment.symptoms,
            "preferred_date": appointment.preferred_date,
            "preferred_time": appointment.preferred_time,
            "appointment_type": appointment.appointment_type,
            "reason_for_visit": appointment.reason_for_visit,
            "medical_history": appointment.medical_history,
            "current_medications": appointment.current_medications,
            "allergies": appointment.allergies,
            "insurance_info": appointment.insurance_info,
            
            "status": "pending",
            "doctor_id": None,
            "doctor_name": None,
            "notes": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        result = await appointments_collection.insert_one(appointment_data)
        appointment_data["_id"] = str(result.inserted_id)

        # Notify admin
        await create_notification(
            user_id=str(admin["_id"]),
            notification_type="new_appointment_request",
            title="New Appointment Request",
            message=f"New appointment from {current_user['full_name']} - {appointment.appointment_type}",
            data=serialize_doc(appointment_data)
        )
        
        # Create chat message
        chat_msg = {
            "sender_id": current_user["id"],
            "sender_name": current_user["full_name"],
            "sender_role": "patient",
            "receiver_id": str(admin["_id"]),
            "receiver_name": admin["full_name"],
            "receiver_role": "admin",
            "message": f"Hello, I've requested an appointment for {appointment.preferred_date} at {appointment.preferred_time}. Reason: {appointment.reason_for_visit}",
            "message_type": "appointment_request",
            "appointment_id": str(result.inserted_id),
            "read": False,
            "created_at": datetime.utcnow()
        }
        await chats_collection.insert_one(chat_msg)

        logger.info(f"✅ Appointment booked by {current_user['email']} at {hospital_name} (ID: {hospital_id})")
        logger.info(f"✅ Admin notified: {admin['email']} (ID: {admin['_id']})")

        return serialize_doc(appointment_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Book appointment error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to book appointment: {str(e)}")

@app.delete("/api/v1/patient/appointments/{appointment_id}", tags=["Patient"])
async def cancel_appointment(
    appointment_id: str,
    reason: Optional[str] = None,
    current_user: dict = Depends(require_role("patient"))
):
    """Cancel an appointment"""
    try:
        appointment = await appointments_collection.find_one({
            "_id": ObjectId(appointment_id),
            "patient_id": current_user["id"]
        })
        
        if not appointment:
            raise HTTPException(status_code=404, detail="Appointment not found")

        if appointment["status"] in ["completed", "cancelled"]:
            raise HTTPException(status_code=400, detail="Cannot cancel this appointment")

        await appointments_collection.update_one(
            {"_id": ObjectId(appointment_id)},
            {"$set": {
                "status": "cancelled",
                "cancellation_reason": reason,
                "cancelled_at": datetime.utcnow(),
                "cancelled_by": "patient"
            }}
        )

        if appointment.get("doctor_id"):
            await create_notification(
                user_id=appointment["doctor_id"],
                notification_type="appointment_cancelled",
                title="Appointment Cancelled",
                message=f"Appointment with {current_user['full_name']} has been cancelled",
                data={"appointment_id": appointment_id, "reason": reason}
            )

        admin = await users_collection.find_one({
            "hospital_id": appointment["hospital_id"],
            "role": "admin"
        })
        
        if admin:
            await create_notification(
                user_id=str(admin["_id"]),
                notification_type="appointment_cancelled",
                title="Appointment Cancelled by Patient",
                message=f"{current_user['full_name']} cancelled their appointment",
                data={"appointment_id": appointment_id, "reason": reason}
            )

        logger.info(f"Appointment {appointment_id} cancelled by patient")

        return {"message": "Appointment cancelled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cancel appointment error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cancel appointment")

# ==================== ADMIN ENDPOINTS ====================

@app.post("/api/v1/admin/hospitals", response_model=HospitalResponse, status_code=status.HTTP_201_CREATED, tags=["Admin"])
async def create_hospital(
    hospital: HospitalCreate,
    current_user: dict = Depends(require_role("admin"))
):
    """Create a new hospital"""
    try:
        hospital_data = hospital.dict()
        hospital_data["admin_id"] = current_user["id"]
        hospital_data["is_pinned"] = False
        hospital_data["is_demo"] = False
        hospital_data["created_at"] = datetime.utcnow()

        result = await hospitals_collection.insert_one(hospital_data)
        hospital_id = str(result.inserted_id)

        if not current_user.get("hospital_id"):
            await users_collection.update_one(
                {"_id": ObjectId(current_user["_id"])},
                {"$set": {"hospital_id": hospital_id}}
            )

        hospital_data["_id"] = hospital_id
        logger.info(f"Hospital created by admin {current_user['email']}: {hospital['name']}")

        return serialize_doc(hospital_data)

    except Exception as e:
        logger.error(f"Create hospital error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create hospital")

@app.get("/api/v1/admin/hospitals", response_model=List[HospitalResponse], tags=["Admin"])
async def get_admin_hospitals(
    current_user: dict = Depends(require_role("admin"))
):
    """Get all hospitals managed by the admin"""
    try:
        # ✅ FIXED: Query by EITHER admin_id OR _id matching admin's hospital_id
        query = {
            "$or": [
                {"admin_id": current_user["id"]},
                {"_id": ObjectId(current_user["hospital_id"])} if current_user.get("hospital_id") else {}
            ]
        }
        
        hospitals = await hospitals_collection.find(query).to_list(100)

        return [serialize_doc(h) for h in hospitals]

    except Exception as e:
        logger.error(f"Get admin hospitals error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch hospitals")
@app.put("/api/v1/admin/hospitals/{hospital_id}", tags=["Admin"])
async def update_hospital(
    hospital_id: str,
    hospital_update: HospitalUpdate,
    current_user: dict = Depends(require_role("admin"))
):
    """Update hospital information"""
    try:
        hospital = await hospitals_collection.find_one({
            "_id": ObjectId(hospital_id),
            "admin_id": current_user["id"]
        })
        
        if not hospital:
            raise HTTPException(status_code=404, detail="Hospital not found")

        update_data = hospital_update.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()

        await hospitals_collection.update_one(
            {"_id": ObjectId(hospital_id)},
            {"$set": update_data}
        )

        logger.info(f"Hospital {hospital_id} updated by admin")

        return {"message": "Hospital updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update hospital error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update hospital")

@app.delete("/api/v1/admin/hospitals/{hospital_id}", tags=["Admin"])
async def delete_hospital(
    hospital_id: str,
    current_user: dict = Depends(require_role("admin"))
):
    """Delete a hospital"""
    try:
        hospital = await hospitals_collection.find_one({
            "_id": ObjectId(hospital_id),
            "admin_id": current_user["id"]
        })
        
        if not hospital:
            raise HTTPException(status_code=404, detail="Hospital not found")
        
        # Check if hospital is pinned (demo hospital)
        if hospital.get("is_pinned") or hospital.get("is_demo"):
            raise HTTPException(status_code=403, detail="Cannot delete demo hospital")
        
        # Check if hospital has active doctors
        active_doctors = await users_collection.count_documents({
            "hospital_id": hospital_id,
            "role": "doctor",
            "is_active": True
        })
        
        if active_doctors > 0:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot delete hospital with {active_doctors} active doctor(s)"
            )
        
        # Check if hospital has pending/confirmed appointments
        active_appointments = await appointments_collection.count_documents({
            "hospital_id": hospital_id,
            "status": {"$in": ["pending", "confirmed"]}
        })
        
        if active_appointments > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete hospital with {active_appointments} active appointment(s)"
            )
        
        # Delete hospital
        await hospitals_collection.delete_one({"_id": ObjectId(hospital_id)})
        
        # Update admin's hospital_id if this was their assigned hospital
        if current_user.get("hospital_id") == hospital_id:
            await users_collection.update_one(
                {"_id": ObjectId(current_user["_id"])},
                {"$unset": {"hospital_id": ""}}
            )
        
        logger.info(f"Hospital {hospital_id} deleted by admin {current_user['email']}")
        
        return {"message": "Hospital deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete hospital error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete hospital")
@app.get("/api/v1/admin/appointments", tags=["Admin"])
async def get_admin_appointments(
    status_filter: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=200),
    current_user: dict = Depends(require_role("admin"))
):
    """Get all appointments for admin's hospital"""
    try:
        hospital_id = current_user.get("hospital_id")
        if not hospital_id:
            return []

        query = {"hospital_id": hospital_id}
        if status_filter and status_filter != 'all':
            query["status"] = status_filter

        appointments = await appointments_collection.find(query)\
            .sort("created_at", -1)\
            .skip(skip)\
            .limit(limit)\
            .to_list(limit)

        logger.info(f"Admin {current_user['email']} fetched {len(appointments)} appointments")

        return [serialize_doc(a) for a in appointments]

    except Exception as e:
        logger.error(f"Get admin appointments error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch appointments")
@app.get("/api/v1/admin/appointments/{appointment_id}", response_model=AppointmentResponse, tags=["Admin"])
async def get_appointment_detail(
    appointment_id: str,
    current_user: dict = Depends(require_role("admin"))
):
    """Get detailed appointment information"""
    try:
        # First, try to find appointment with hospital filter
        appointment = await appointments_collection.find_one({
            "_id": ObjectId(appointment_id),
            "hospital_id": current_user.get("hospital_id")
        })
        
        # If not found, check if appointment exists at all (for debugging)
        if not appointment:
            # Try finding without hospital filter
            appointment_any = await appointments_collection.find_one({
                "_id": ObjectId(appointment_id)
            })
            
            if appointment_any:
                # Appointment exists but belongs to different hospital
                admin_hospital = await hospitals_collection.find_one({
                    "_id": ObjectId(current_user.get("hospital_id"))
                })
                
                apt_hospital = await hospitals_collection.find_one({
                    "_id": ObjectId(appointment_any.get("hospital_id"))
                })
                
                logger.warning(
                    f"Hospital mismatch for appointment {appointment_id}: "
                    f"Admin hospital: {admin_hospital.get('name') if admin_hospital else 'Unknown'} "
                    f"({current_user.get('hospital_id')}), "
                    f"Appointment hospital: {apt_hospital.get('name') if apt_hospital else 'Unknown'} "
                    f"({appointment_any.get('hospital_id')})"
                )
                
                # For now, allow access but log it
                # In production, you might want to raise an error
                appointment = appointment_any
            else:
                raise HTTPException(status_code=404, detail="Appointment not found")

        return serialize_doc(appointment)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get appointment detail error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch appointment details")

@app.get("/api/v1/doctor/appointments", response_model=List[AppointmentResponse], tags=["Doctor"])
async def get_doctor_appointments(
    status_filter: Optional[str] = Query(None),
    date_filter: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),  # ← Maximum is 100!
    current_user: dict = Depends(require_role("doctor"))
):
    """Get all appointments for admin's hospital (CarePlus)"""
    try:
        hospital_id = current_user.get("hospital_id")
        if not hospital_id:
            return []

        query = {"hospital_id": hospital_id}
        if status_filter:
            query["status"] = status_filter

        appointments = await appointments_collection.find(query)\
            .sort("created_at", -1)\
            .skip(skip)\
            .limit(limit)\
            .to_list(limit)

        return [serialize_doc(a) for a in appointments]

    except Exception as e:
        logger.error(f"Get admin appointments error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch appointments")

@app.put("/api/v1/admin/appointments/{appointment_id}/assign", tags=["Admin"])
async def assign_doctor_to_appointment(
    appointment_id: str,
    assignment: DoctorAssignment,
    current_user: dict = Depends(require_role("admin"))
):
    """Assign CarePlus doctor to appointment and notify patient"""
    try:
        appointment = await appointments_collection.find_one({
            "_id": ObjectId(appointment_id),
            "hospital_id": current_user.get("hospital_id")
        })
        
        if not appointment:
            raise HTTPException(status_code=404, detail="Appointment not found")

        doctor = await users_collection.find_one({
            "_id": ObjectId(assignment.doctor_id),
            "role": "doctor",
            "hospital_id": current_user.get("hospital_id")
        })
        
        if not doctor:
            raise HTTPException(status_code=404, detail="Doctor not found in your hospital")

        update_data = {
            "doctor_id": assignment.doctor_id,
            "doctor_name": doctor["full_name"],
            "status": "confirmed",
            "scheduled_date": assignment.scheduled_date or appointment.get("preferred_date"),
            "scheduled_time": assignment.scheduled_time or appointment.get("preferred_time"),
            "updated_at": datetime.utcnow()
        }
        
        if assignment.notes:
            update_data["admin_notes"] = assignment.notes

        await appointments_collection.update_one(
            {"_id": ObjectId(appointment_id)},
            {"$set": update_data}
        )

        # Notify doctor
        await create_notification(
            user_id=assignment.doctor_id,
            notification_type="appointment_assigned",
            title="New Patient Assigned",
            message=f"You have been assigned to patient {appointment['patient_name']} on {update_data['scheduled_date']} at {update_data['scheduled_time']}",
            data=serialize_doc(appointment)
        )

        # Notify patient
        await create_notification(
            user_id=appointment["patient_id"],
            notification_type="appointment_confirmed",
            title="Appointment Confirmed",
            message=f"Your appointment has been confirmed with Dr. {doctor['full_name']} on {update_data['scheduled_date']} at {update_data['scheduled_time']}",
            data=serialize_doc(appointment)
        )
        
        # Send chat to patient from admin
        chat_to_patient = {
            "sender_id": current_user["id"],
            "sender_name": current_user["full_name"],
            "sender_role": "admin",
            "receiver_id": appointment["patient_id"],
            "receiver_name": appointment["patient_name"],
            "receiver_role": "patient",
            "message": f"Good news! Your appointment has been scheduled with Dr. {doctor['full_name']} ({doctor.get('specialization', 'General Physician')}) on {update_data['scheduled_date']} at {update_data['scheduled_time']}. Please arrive 15 minutes early.",
            "message_type": "appointment_confirmation",
            "appointment_id": appointment_id,
            "read": False,
            "created_at": datetime.utcnow()
        }
        await chats_collection.insert_one(chat_to_patient)
        
        # Send notification to patient via WebSocket
        await manager.send_personal_message({
            "type": "chat_message",
            "data": serialize_doc(chat_to_patient)
        }, appointment["patient_id"])

        logger.info(f"Doctor {assignment.doctor_id} assigned to appointment {appointment_id}")

        return {
            "message": "Doctor assigned and patient notified successfully",
            "doctor_name": doctor["full_name"],
            "scheduled_date": update_data["scheduled_date"],
            "scheduled_time": update_data["scheduled_time"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Assign doctor error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to assign doctor")
@app.delete("/api/v1/admin/appointments/{appointment_id}", tags=["Admin"])
async def cancel_appointment_by_admin(
    appointment_id: str,
    reason: Optional[str] = None,
    current_user: dict = Depends(require_role("admin"))
):
    """Cancel an appointment (Admin)"""
    try:
        appointment = await appointments_collection.find_one({
            "_id": ObjectId(appointment_id),
            "hospital_id": current_user.get("hospital_id")
        })
        
        if not appointment:
            raise HTTPException(status_code=404, detail="Appointment not found")

        if appointment["status"] in ["completed", "cancelled"]:
            raise HTTPException(status_code=400, detail="Cannot cancel this appointment")

        await appointments_collection.update_one(
            {"_id": ObjectId(appointment_id)},
            {"$set": {
                "status": "cancelled",
                "cancellation_reason": reason,
                "cancelled_at": datetime.utcnow(),
                "cancelled_by": "admin"
            }}
        )

        # Notify patient
        await create_notification(
            user_id=appointment["patient_id"],
            notification_type="appointment_cancelled",
            title="Appointment Cancelled",
            message=f"Your appointment has been cancelled by the hospital. Reason: {reason or 'Not specified'}",
            data={"appointment_id": appointment_id, "reason": reason}
        )

        # Notify doctor if assigned
        if appointment.get("doctor_id"):
            await create_notification(
                user_id=appointment["doctor_id"],
                notification_type="appointment_cancelled",
                title="Appointment Cancelled",
                message=f"Appointment with {appointment['patient_name']} has been cancelled by admin",
                data={"appointment_id": appointment_id, "reason": reason}
            )

        logger.info(f"Appointment {appointment_id} cancelled by admin")

        return {"message": "Appointment cancelled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cancel appointment error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cancel appointment")
@app.put("/api/v1/admin/appointments/{appointment_id}/reschedule", tags=["Admin"])
async def reschedule_appointment(
    appointment_id: str,
    new_date: str,
    new_time: str,
    reason: Optional[str] = None,
    current_user: dict = Depends(require_role("admin"))
):
    """Reschedule an appointment"""
    try:
        appointment = await appointments_collection.find_one({
            "_id": ObjectId(appointment_id),
            "hospital_id": current_user.get("hospital_id")
        })
        
        if not appointment:
            raise HTTPException(status_code=404, detail="Appointment not found")

        await appointments_collection.update_one(
            {"_id": ObjectId(appointment_id)},
            {"$set": {
                "scheduled_date": new_date,
                "scheduled_time": new_time,
                "reschedule_reason": reason,
                "rescheduled_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }}
        )

        await create_notification(
            user_id=appointment["patient_id"],
            notification_type="appointment_rescheduled",
            title="Appointment Rescheduled",
            message=f"Your appointment has been rescheduled to {new_date} at {new_time}",
            data={"appointment_id": appointment_id, "new_date": new_date, "new_time": new_time, "reason": reason}
        )

        if appointment.get("doctor_id"):
            await create_notification(
                user_id=appointment["doctor_id"],
                notification_type="appointment_rescheduled",
                title="Appointment Rescheduled",
                message=f"Appointment with {appointment['patient_name']} rescheduled to {new_date} at {new_time}",
                data={"appointment_id": appointment_id, "new_date": new_date, "new_time": new_time}
            )

        logger.info(f"Appointment {appointment_id} rescheduled")

        return {"message": "Appointment rescheduled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reschedule appointment error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to reschedule appointment")

@app.get("/api/v1/admin/doctors", tags=["Admin"])
async def get_hospital_doctors(
    current_user: dict = Depends(require_role("admin"))
):
    """Get all doctors in CarePlus Hospital"""
    try:
        hospital_id = current_user.get("hospital_id")
        if not hospital_id:
            return []

        doctors = await users_collection.find({
            "hospital_id": hospital_id,
            "role": "doctor",
            "is_active": True
        }).to_list(100)

        return [serialize_doc(d) for d in doctors]

    except Exception as e:
        logger.error(f"Get hospital doctors error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch doctors")

@app.get("/api/v1/patient/dashboard/stats", tags=["Patient"])
async def get_patient_dashboard_stats(
    current_user: dict = Depends(require_role("patient"))
):
    """Get dashboard statistics for patient"""
    try:
        # Get counts
        total_appointments = await appointments_collection.count_documents({
            "patient_id": current_user["id"]
        })
        
        upcoming_appointments = await appointments_collection.count_documents({
            "patient_id": current_user["id"],
            "status": {"$in": ["pending", "confirmed"]},
            "scheduled_date": {"$gte": datetime.utcnow().date().isoformat()}
        })
        
        total_prescriptions = await prescriptions_collection.count_documents({
            "patient_id": current_user["id"]
        })
        
        active_medications = await medications_collection.count_documents({
            "patient_id": current_user["id"],
            "active": True
        })
        
        recent_appointments = await appointments_collection.find({
            "patient_id": current_user["id"]
        }).sort("created_at", -1).limit(5).to_list(5)
        
        return {
            "total_appointments": total_appointments,
            "upcoming_appointments": upcoming_appointments,
            "total_prescriptions": total_prescriptions,
            "active_medications": active_medications,
            "recent_appointments": [serialize_doc(a) for a in recent_appointments]
        }
    except Exception as e:
        logger.error(f"Get patient dashboard stats error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch dashboard stats")
@app.get("/api/v1/admin/dashboard/stats", tags=["Admin"])
async def get_admin_dashboard_stats(
    current_user: dict = Depends(require_role("admin"))
):
    """Get dashboard statistics for admin"""
    try:
        hospital_id = current_user.get("hospital_id")
        if not hospital_id:
            return {"error": "No hospital assigned"}

        appointments = await appointments_collection.find({"hospital_id": hospital_id}).to_list(1000)
        
        stats = {
            "total_appointments": len(appointments),
            "pending_appointments": len([a for a in appointments if a["status"] == "pending"]),
            "confirmed_appointments": len([a for a in appointments if a["status"] == "confirmed"]),
            "completed_appointments": len([a for a in appointments if a["status"] == "completed"]),
            "cancelled_appointments": len([a for a in appointments if a["status"] == "cancelled"]),
            "total_doctors": await users_collection.count_documents({"hospital_id": hospital_id, "role": "doctor"}),
            "today_appointments": len([a for a in appointments if a.get("scheduled_date") == datetime.utcnow().date().isoformat()])
        }

        return stats

    except Exception as e:
        logger.error(f"Get dashboard stats error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch dashboard stats")

# ==================== DOCTOR ENDPOINTS ====================

@app.get("/api/v1/doctor/profile", tags=["Doctor"])
async def get_doctor_profile(
    current_user: dict = Depends(require_role("doctor"))
):
    """Get doctor profile"""
    return serialize_doc(current_user)

@app.put("/api/v1/doctor/profile", tags=["Doctor"])
async def update_doctor_profile(
    profile: DoctorProfile,
    current_user: dict = Depends(require_role("doctor"))
):
    """Update doctor profile"""
    try:
        update_data = profile.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()

        await users_collection.update_one(
            {"_id": ObjectId(current_user["_id"])},
            {"$set": update_data}
        )

        logger.info(f"Doctor profile updated: {current_user['email']}")

        return {"message": "Profile updated successfully"}

    except Exception as e:
        logger.error(f"Doctor profile update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update profile")

@app.get("/api/v1/doctor/appointments", response_model=List[AppointmentResponse], tags=["Doctor"])
async def get_doctor_appointments(
    status_filter: Optional[str] = Query(None),
    date_filter: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: dict = Depends(require_role("doctor"))
):
    """Get all appointments assigned to the doctor"""
    try:
        query = {"doctor_id": current_user["id"]}
        
        if status_filter:
            query["status"] = status_filter
        
        if date_filter:
            query["scheduled_date"] = date_filter

        appointments = await appointments_collection.find(query)\
            .sort("scheduled_date", 1)\
            .skip(skip)\
            .limit(limit)\
            .to_list(limit)

        return [serialize_doc(a) for a in appointments]

    except Exception as e:
        logger.error(f"Get doctor appointments error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch appointments")

@app.put("/api/v1/doctor/appointments/{appointment_id}/status", tags=["Doctor"])
async def update_appointment_status(
    appointment_id: str,
    new_status: str = Query(..., regex="^(confirmed|in_progress|completed)$"),
    notes: Optional[str] = None,
    current_user: dict = Depends(require_role("doctor"))
):
    """Update appointment status"""
    try:
        appointment = await appointments_collection.find_one({
            "_id": ObjectId(appointment_id),
            "doctor_id": current_user["id"]
        })
        
        if not appointment:
            raise HTTPException(status_code=404, detail="Appointment not found")

        update_data = {
            "status": new_status,
            "updated_at": datetime.utcnow()
        }
        
        if notes:
            update_data["doctor_notes"] = notes
        
        if new_status == "completed":
            update_data["completed_at"] = datetime.utcnow()

        await appointments_collection.update_one(
            {"_id": ObjectId(appointment_id)},
            {"$set": update_data}
        )

        await create_notification(
            user_id=appointment["patient_id"],
            notification_type="appointment_status_update",
            title=f"Appointment {new_status.replace('_', ' ').title()}",
            message=f"Your appointment status has been updated to {new_status}",
            data={"appointment_id": appointment_id, "status": new_status}
        )

        logger.info(f"Appointment {appointment_id} status updated to {new_status}")

        return {"message": "Appointment status updated successfully", "status": new_status}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update appointment status error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update appointment status")

@app.get("/api/v1/doctor/patient/{patient_id}/details", tags=["Doctor"])
async def get_patient_health_details(
    patient_id: str,
    current_user: dict = Depends(require_role("doctor"))
):
    """Get comprehensive patient health details"""
    try:
        has_appointment = await appointments_collection.find_one({
            "patient_id": patient_id,
            "doctor_id": current_user["id"]
        })
        
        if not has_appointment:
            raise HTTPException(status_code=403, detail="Access denied to this patient's records")

        patient = await users_collection.find_one({
            "_id": ObjectId(patient_id),
            "role": "patient"
        })
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        prescriptions = await prescriptions_collection.find(
            {"patient_id": patient_id}
        ).sort("uploaded_at", -1).limit(10).to_list(10)

        vitals = await vitals_collection.find(
            {"patient_id": patient_id}
        ).sort("recorded_at", -1).limit(30).to_list(30)

        appointment_history = await appointments_collection.find(
            {"patient_id": patient_id}
        ).sort("created_at", -1).limit(10).to_list(10)

        medications = await medications_collection.find(
            {"patient_id": patient_id, "active": True}
        ).to_list(50)

        return {
            "patient": serialize_doc(patient),
            "prescriptions": [serialize_doc(p) for p in prescriptions],
            "vitals": [serialize_doc(v) for v in vitals],
            "appointment_history": [serialize_doc(a) for a in appointment_history],
            "active_medications": [serialize_doc(m) for m in medications]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get patient details error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch patient details")

@app.get("/api/v1/doctor/dashboard/stats", tags=["Doctor"])
async def get_doctor_dashboard_stats(
    current_user: dict = Depends(require_role("doctor"))
):
    """Get dashboard statistics for doctor"""
    try:
        appointments = await appointments_collection.find({
            "doctor_id": current_user["id"]
        }).to_list(1000)

        today = datetime.utcnow().date().isoformat()

        stats = {
            "total_appointments": len(appointments),
            "today_appointments": len([a for a in appointments if a.get("scheduled_date") == today]),
            "pending_appointments": len([a for a in appointments if a["status"] == "pending"]),
            "completed_appointments": len([a for a in appointments if a["status"] == "completed"]),
            "total_patients": len(set(a["patient_id"] for a in appointments))
        }

        return stats

    except Exception as e:
        logger.error(f"Get doctor dashboard stats error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch dashboard stats")

# ==================== HEALTH VITALS ENDPOINTS ====================

@app.post("/api/v1/vitals", tags=["Health Vitals"])
async def create_health_vitals(
    vitals: HealthVitalsCreate,
    current_user: dict = Depends(require_role("patient"))
):
    """Record health vitals"""
    try:
        vitals_data = vitals.dict()
        vitals_data["patient_id"] = current_user["id"]
        vitals_data["recorded_at"] = datetime.utcnow()

        result = await vitals_collection.insert_one(vitals_data)
        vitals_data["_id"] = str(result.inserted_id)

        logger.info(f"Vitals recorded for patient {current_user['email']}")

        return serialize_doc(vitals_data)

    except Exception as e:
        logger.error(f"Create vitals error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to record vitals")

@app.get("/api/v1/vitals", tags=["Health Vitals"])
async def get_health_vitals(
    days: int = Query(30, ge=1, le=365, description="Number of days to retrieve"),
    current_user: dict = Depends(require_role("patient"))
):
    """Get health vitals history"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        vitals = await vitals_collection.find({
            "patient_id": current_user["id"],
            "recorded_at": {"$gte": start_date}
        }).sort("recorded_at", -1).to_list(1000)

        return [serialize_doc(v) for v in vitals]

    except Exception as e:
        logger.error(f"Get vitals error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch vitals")

@app.get("/api/v1/vitals/latest", tags=["Health Vitals"])
async def get_latest_vitals(
    current_user: dict = Depends(require_role("patient"))
):
    """Get most recent vital signs"""
    try:
        latest_vital = await vitals_collection.find_one(
            {"patient_id": current_user["id"]},
            sort=[("recorded_at", -1)]
        )

        if not latest_vital:
            raise HTTPException(status_code=404, detail="No vitals found")

        return serialize_doc(latest_vital)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get latest vitals error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch latest vitals")

# Add to backend main.py - OPTIONAL DELETE VITALS ENDPOINT

@app.delete("/api/v1/vitals/{vital_id}", tags=["Health Vitals"])
async def delete_health_vital(
    vital_id: str,
    current_user: dict = Depends(require_role("patient"))
):
    """Delete a health vital record"""
    try:
        vital = await vitals_collection.find_one({
            "_id": ObjectId(vital_id),
            "patient_id": current_user["id"]
        })
        
        if not vital:
            raise HTTPException(status_code=404, detail="Vital record not found")

        await vitals_collection.delete_one({"_id": ObjectId(vital_id)})

        logger.info(f"Vital record {vital_id} deleted by patient {current_user['email']}")

        return {"message": "Vital record deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete vital error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete vital record")
# ==================== MEDICATION REMINDERS ====================

@app.post("/api/v1/medications/reminders", tags=["Medications"])
async def create_medication_reminder(
    reminder: MedicationReminderCreate,
    current_user: dict = Depends(require_role("patient"))
):
    """Create a medication reminder"""
    try:
        reminder_data = reminder.dict()
        reminder_data["patient_id"] = current_user["id"]
        reminder_data["created_at"] = datetime.utcnow()
        reminder_data["active"] = True

        result = await medications_collection.insert_one(reminder_data)
        reminder_data["_id"] = str(result.inserted_id)

        logger.info(f"Medication reminder created for {current_user['email']}")

        return serialize_doc(reminder_data)

    except Exception as e:
        logger.error(f"Create medication reminder error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create reminder")

@app.get("/api/v1/medications/reminders", tags=["Medications"])
async def get_medication_reminders(
    active_only: bool = Query(True),
    current_user: dict = Depends(require_role("patient"))
):
    """Get all medication reminders"""
    try:
        query = {"patient_id": current_user["id"]}
        
        if active_only:
            query["active"] = True

        reminders = await medications_collection.find(query).to_list(100)

        return [serialize_doc(r) for r in reminders]

    except Exception as e:
        logger.error(f"Get medication reminders error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch reminders")

@app.put("/api/v1/medications/reminders/{reminder_id}", tags=["Medications"])
async def update_medication_reminder(
    reminder_id: str,
    reminder_update: MedicationReminderCreate,
    current_user: dict = Depends(require_role("patient"))
):
    """Update a medication reminder"""
    try:
        reminder = await medications_collection.find_one({
            "_id": ObjectId(reminder_id),
            "patient_id": current_user["id"]
        })
        
        if not reminder:
            raise HTTPException(status_code=404, detail="Reminder not found")

        update_data = reminder_update.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()

        await medications_collection.update_one(
            {"_id": ObjectId(reminder_id)},
            {"$set": update_data}
        )

        return {"message": "Reminder updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update medication reminder error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update reminder")

@app.delete("/api/v1/medications/reminders/{reminder_id}", tags=["Medications"])
async def delete_medication_reminder(
    reminder_id: str,
    current_user: dict = Depends(require_role("patient"))
):
    """Deactivate a medication reminder"""
    try:
        reminder = await medications_collection.find_one({
            "_id": ObjectId(reminder_id),
            "patient_id": current_user["id"]
        })
        
        if not reminder:
            raise HTTPException(status_code=404, detail="Reminder not found")

        await medications_collection.update_one(
            {"_id": ObjectId(reminder_id)},
            {"$set": {"active": False, "deactivated_at": datetime.utcnow()}}
        )

        return {"message": "Reminder deactivated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete medication reminder error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to deactivate reminder")

# ==================== ENHANCED MEDICATION ENDPOINTS ====================

@app.post("/api/v1/medications/reminders/{reminder_id}/taken", tags=["Medications"])
async def mark_medication_taken(
    reminder_id: str,
    taken_at: Optional[str] = None,
    current_user: dict = Depends(require_role("patient"))
):
    """Mark a medication as taken"""
    try:
        reminder = await medications_collection.find_one({
            "_id": ObjectId(reminder_id),
            "patient_id": current_user["id"]
        })
        
        if not reminder:
            raise HTTPException(status_code=404, detail="Reminder not found")

        log_entry = {
            "reminder_id": reminder_id,
            "patient_id": current_user["id"],
            "medication_name": reminder["medication_name"],
            "dosage": reminder["dosage"],
            "taken_at": taken_at or datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow()
        }
        
        await db["medication_logs"].insert_one(log_entry)

        logger.info(f"Medication {reminder_id} marked as taken by {current_user['email']}")

        return {"message": "Medication marked as taken", "log": serialize_doc(log_entry)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mark medication taken error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to mark medication as taken")

@app.get("/api/v1/medications/logs", tags=["Medications"])
async def get_medication_logs(
    reminder_id: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365),
    current_user: dict = Depends(require_role("patient"))
):
    """Get medication intake history"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        query = {
            "patient_id": current_user["id"],
            "created_at": {"$gte": start_date}
        }
        
        if reminder_id:
            query["reminder_id"] = reminder_id
        
        logs = await db["medication_logs"].find(query)\
            .sort("created_at", -1)\
            .to_list(1000)

        return [serialize_doc(log) for log in logs]

    except Exception as e:
        logger.error(f"Get medication logs error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch medication logs")

@app.get("/api/v1/medications/today", tags=["Medications"])
async def get_todays_medications(
    current_user: dict = Depends(require_role("patient"))
):
    """Get today's medication schedule"""
    try:
        from datetime import date
        
        reminders = await medications_collection.find({
            "patient_id": current_user["id"],
            "active": True
        }).to_list(100)
        
        today_start = datetime.combine(date.today(), datetime.min.time())
        today_end = datetime.combine(date.today(), datetime.max.time())
        
        logs = await db["medication_logs"].find({
            "patient_id": current_user["id"],
            "created_at": {"$gte": today_start, "$lte": today_end}
        }).to_list(1000)
        
        schedule = []
        for reminder in reminders:
            for time in reminder.get("times", []):
                taken = any(
                    log["reminder_id"] == str(reminder["_id"]) and
                    log.get("taken_at", "").startswith(datetime.now().strftime("%Y-%m-%d"))
                    for log in logs
                )
                
                schedule.append({
                    "reminder_id": str(reminder["_id"]),
                    "medication_name": reminder["medication_name"],
                    "dosage": reminder["dosage"],
                    "time": time,
                    "taken": taken,
                    "instructions": reminder.get("instructions", "")
                })
        
        schedule.sort(key=lambda x: x["time"])
        
        return {
            "date": date.today().isoformat(),
            "total_medications": len(schedule),
            "taken": len([s for s in schedule if s["taken"]]),
            "pending": len([s for s in schedule if not s["taken"]]),
            "schedule": schedule
        }

    except Exception as e:
        logger.error(f"Get today's medications error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch today's medications")

@app.get("/api/v1/medications/adherence", tags=["Medications"])
async def get_medication_adherence(
    days: int = Query(30, ge=7, le=90),
    current_user: dict = Depends(require_role("patient"))
):
    """Get medication adherence statistics"""
    try:
        from datetime import date, timedelta
        
        start_date = date.today() - timedelta(days=days)
        
        reminders = await medications_collection.find({
            "patient_id": current_user["id"],
            "active": True
        }).to_list(100)
        
        total_expected = 0
        for reminder in reminders:
            times_per_day = len(reminder.get("times", []))
            duration_days = min(days, reminder.get("duration_days", days))
            total_expected += times_per_day * duration_days
        
        start_datetime = datetime.combine(start_date, datetime.min.time())
        logs = await db["medication_logs"].find({
            "patient_id": current_user["id"],
            "created_at": {"$gte": start_datetime}
        }).to_list(10000)
        
        total_taken = len(logs)
        adherence_rate = (total_taken / total_expected * 100) if total_expected > 0 else 0
        
        return {
            "period_days": days,
            "total_expected": total_expected,
            "total_taken": total_taken,
            "missed": total_expected - total_taken,
            "adherence_rate": round(adherence_rate, 1),
            "active_medications": len(reminders)
        }

    except Exception as e:
        logger.error(f"Get adherence stats error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch adherence statistics")
# ==================== CHAT ENDPOINTS ====================

@app.post("/api/v1/chat/send", tags=["Chat"])
async def send_chat_message(
    message: ChatMessageCreate,
    current_user: dict = Depends(get_current_user)
):
    """Send a chat message between patient and admin/doctor"""
    try:
        receiver = await users_collection.find_one({"_id": ObjectId(message.receiver_id)})
        if not receiver:
            raise HTTPException(status_code=404, detail="Receiver not found")

        message_data = {
            "sender_id": current_user["id"],
            "sender_name": current_user["full_name"],
            "sender_role": current_user["role"],
            "receiver_id": message.receiver_id,
            "receiver_name": receiver["full_name"],
            "receiver_role": receiver["role"],
            "message": message.message,
            "message_type": message.message_type if hasattr(message, 'message_type') else "text",
            "read": False,
            "created_at": datetime.utcnow()
        }

        result = await chats_collection.insert_one(message_data)
        message_data["_id"] = str(result.inserted_id)

        # Send via WebSocket
        await manager.send_personal_message({
            "type": "chat_message",
            "data": serialize_doc(message_data)
        }, message.receiver_id)

        logger.info(f"Message sent from {current_user['email']} to {receiver['email']}")

        return serialize_doc(message_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Send message error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send message")

@app.get("/api/v1/chat/conversations", tags=["Chat"])
async def get_conversations(
    current_user: dict = Depends(get_current_user)
):
    """Get all conversations for the current user"""
    try:
        messages = await chats_collection.find({
            "$or": [
                {"sender_id": current_user["id"]},
                {"receiver_id": current_user["id"]}
            ]
        }).sort("created_at", -1).to_list(1000)

        conversations = {}
        for msg in messages:
            other_user_id = msg["receiver_id"] if msg["sender_id"] == current_user["id"] else msg["sender_id"]
            
            if other_user_id not in conversations:
                conversations[other_user_id] = {
                    "user_id": other_user_id,
                    "user_name": msg["receiver_name"] if msg["sender_id"] == current_user["id"] else msg["sender_name"],
                    "user_role": msg["receiver_role"] if msg["sender_id"] == current_user["id"] else msg["sender_role"],
                    "last_message": msg["message"],
                    "last_message_time": msg["created_at"],
                    "unread_count": 0
                }
            
            if msg["receiver_id"] == current_user["id"] and not msg.get("read"):
                conversations[other_user_id]["unread_count"] += 1

        return list(conversations.values())

    except Exception as e:
        logger.error(f"Get conversations error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch conversations")

@app.get("/api/v1/chat/{user_id}/messages", tags=["Chat"])
async def get_chat_messages(
    user_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """Get chat messages between current user and another user"""
    try:
        messages = await chats_collection.find({
            "$or": [
                {"sender_id": current_user["id"], "receiver_id": user_id},
                {"sender_id": user_id, "receiver_id": current_user["id"]}
            ]
        }).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)

        await chats_collection.update_many(
            {"sender_id": user_id, "receiver_id": current_user["id"], "read": False},
            {"$set": {"read": True, "read_at": datetime.utcnow()}}
        )

        return [serialize_doc(m) for m in reversed(messages)]

    except Exception as e:
        logger.error(f"Get chat messages error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch messages")

@app.put("/api/v1/chat/messages/{message_id}/read", tags=["Chat"])
async def mark_message_as_read(
    message_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Mark a message as read"""
    try:
        message = await chats_collection.find_one({
            "_id": ObjectId(message_id),
            "receiver_id": current_user["id"]
        })
        
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        await chats_collection.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": {"read": True, "read_at": datetime.utcnow()}}
        )

        return {"message": "Message marked as read"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mark message as read error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update message")

@app.get("/api/v1/chat/unread/count", tags=["Chat"])
async def get_unread_messages_count(
    current_user: dict = Depends(get_current_user)
):
    """Get count of unread messages"""
    try:
        count = await chats_collection.count_documents({
            "receiver_id": current_user["id"],
            "read": False
        })

        return {"unread_count": count}

    except Exception as e:
        logger.error(f"Get unread messages count error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get unread count")


# ==================== ENHANCED CHAT ENDPOINT ====================

@app.get("/api/v1/chat/available-users", tags=["Chat"])
async def get_available_chat_users(
    current_user: dict = Depends(get_current_user)
):
    """Get list of users available to chat with"""
    try:
        available_users = []
        
        if current_user["role"] == "patient":
            appointments = await appointments_collection.find({
                "patient_id": current_user["id"]
            }).to_list(100)
            
            hospital_ids = list(set([apt.get("hospital_id") for apt in appointments if apt.get("hospital_id")]))
            doctor_ids = list(set([apt.get("doctor_id") for apt in appointments if apt.get("doctor_id")]))
            
            if hospital_ids:
                admins = await users_collection.find({
                    "role": "admin",
                    "hospital_id": {"$in": hospital_ids},
                    "is_active": True
                }).to_list(50)
                
                for admin in admins:
                    hospital = await hospitals_collection.find_one({"_id": ObjectId(admin.get("hospital_id"))})
                    available_users.append({
                        "id": str(admin["_id"]),
                        "name": admin["full_name"],
                        "role": "admin",
                        "email": admin["email"],
                        "hospital_name": hospital.get("name", "Unknown Hospital") if hospital else "Unknown Hospital",
                        "online": False
                    })
            
            if doctor_ids:
                doctors = await users_collection.find({
                    "_id": {"$in": [ObjectId(did) for did in doctor_ids if did]},
                    "role": "doctor",
                    "is_active": True
                }).to_list(50)
                
                for doctor in doctors:
                    available_users.append({
                        "id": str(doctor["_id"]),
                        "name": doctor["full_name"],
                        "role": "doctor",
                        "email": doctor["email"],
                        "specialization": doctor.get("specialization", "General Physician"),
                        "online": False
                    })
        
        elif current_user["role"] == "doctor":
            appointments = await appointments_collection.find({
                "doctor_id": current_user["id"]
            }).to_list(100)
            
            patient_ids = list(set([apt.get("patient_id") for apt in appointments if apt.get("patient_id")]))
            
            if patient_ids:
                patients = await users_collection.find({
                    "_id": {"$in": [ObjectId(pid) for pid in patient_ids]},
                    "role": "patient",
                    "is_active": True
                }).to_list(100)
                
                for patient in patients:
                    available_users.append({
                        "id": str(patient["_id"]),
                        "name": patient["full_name"],
                        "role": "patient",
                        "email": patient["email"],
                        "phone": patient.get("phone_number"),
                        "online": False
                    })
            
            if current_user.get("hospital_id"):
                admin = await users_collection.find_one({
                    "hospital_id": current_user["hospital_id"],
                    "role": "admin",
                    "is_active": True
                })
                
                if admin:
                    available_users.append({
                        "id": str(admin["_id"]),
                        "name": admin["full_name"],
                        "role": "admin",
                        "email": admin["email"],
                        "hospital_name": current_user.get("hospital_name", "Hospital"),
                        "online": False
                    })
        
        elif current_user["role"] == "admin":
            appointments = await appointments_collection.find({
                "hospital_id": current_user.get("hospital_id")
            }).to_list(1000)
            
            patient_ids = list(set([apt.get("patient_id") for apt in appointments if apt.get("patient_id")]))
            
            if patient_ids:
                patients = await users_collection.find({
                    "_id": {"$in": [ObjectId(pid) for pid in patient_ids]},
                    "role": "patient",
                    "is_active": True
                }).to_list(100)
                
                for patient in patients:
                    available_users.append({
                        "id": str(patient["_id"]),
                        "name": patient["full_name"],
                        "role": "patient",
                        "email": patient["email"],
                        "phone": patient.get("phone_number"),
                        "online": False
                    })
            
            doctors = await users_collection.find({
                "hospital_id": current_user.get("hospital_id"),
                "role": "doctor",
                "is_active": True
            }).to_list(100)
            
            for doctor in doctors:
                available_users.append({
                    "id": str(doctor["_id"]),
                    "name": doctor["full_name"],
                    "role": "doctor",
                    "email": doctor["email"],
                    "specialization": doctor.get("specialization", "General Physician"),
                    "online": False
                })
        
        seen_ids = set()
        unique_users = []
        for user in available_users:
            if user["id"] not in seen_ids:
                seen_ids.add(user["id"])
                unique_users.append(user)
        
        logger.info(f"Found {len(unique_users)} available chat users for {current_user['email']}")
        
        return unique_users

    except Exception as e:
        logger.error(f"Get available chat users error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch available users")
# ==================== NOTIFICATIONS ENDPOINTS ====================

@app.get("/api/v1/notifications", response_model=List[NotificationResponse], tags=["Notifications"])
async def get_notifications(
    unread_only: bool = Query(False),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """Get all notifications for the current user"""
    try:
        query = {"user_id": current_user["id"]}
        
        if unread_only:
            query["read"] = False

        notifications = await notifications_collection.find(query)\
            .sort("created_at", -1)\
            .skip(skip)\
            .limit(limit)\
            .to_list(limit)

        return [serialize_doc(n) for n in notifications]

    except Exception as e:
        logger.error(f"Get notifications error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch notifications")

@app.put("/api/v1/notifications/{notification_id}/read", tags=["Notifications"])
async def mark_notification_as_read(
    notification_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Mark a notification as read"""
    try:
        notification = await notifications_collection.find_one({
            "_id": ObjectId(notification_id),
            "user_id": current_user["id"]
        })
        
        if not notification:
            raise HTTPException(status_code=404, detail="Notification not found")

        await notifications_collection.update_one(
            {"_id": ObjectId(notification_id)},
            {"$set": {"read": True, "read_at": datetime.utcnow()}}
        )

        return {"message": "Notification marked as read"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mark notification as read error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update notification")

@app.put("/api/v1/notifications/mark-all-read", tags=["Notifications"])
async def mark_all_notifications_as_read(
    current_user: dict = Depends(get_current_user)
):
    """Mark all notifications as read"""
    try:
        result = await notifications_collection.update_many(
            {"user_id": current_user["id"], "read": False},
            {"$set": {"read": True, "read_at": datetime.utcnow()}}
        )

        return {"message": f"{result.modified_count} notifications marked as read"}

    except Exception as e:
        logger.error(f"Mark all notifications as read error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update notifications")

@app.delete("/api/v1/notifications/{notification_id}", tags=["Notifications"])
async def delete_notification(
    notification_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a notification"""
    try:
        notification = await notifications_collection.find_one({
            "_id": ObjectId(notification_id),
            "user_id": current_user["id"]
        })
        
        if not notification:
            raise HTTPException(status_code=404, detail="Notification not found")

        await notifications_collection.delete_one({"_id": ObjectId(notification_id)})

        return {"message": "Notification deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete notification error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete notification")

@app.get("/api/v1/notifications/unread/count", tags=["Notifications"])
async def get_unread_notifications_count(
    current_user: dict = Depends(get_current_user)
):
    """Get count of unread notifications"""
    try:
        count = await notifications_collection.count_documents({
            "user_id": current_user["id"],
            "read": False
        })

        return {"unread_count": count}

    except Exception as e:
        logger.error(f"Get unread count error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get unread count")

# ==================== WEBSOCKET ENDPOINTS ====================

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time communication"""
    try:
        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            await websocket.close(code=1008, reason="User not found")
            return

        await manager.connect(websocket, user_id, user["role"])
        
        try:
            while True:
                data = await websocket.receive_json()
                
                message_type = data.get("type")
                
                if message_type == "chat_message":
                    receiver_id = data.get("receiver_id")
                    message_content = data.get("message")
                    
                    receiver = await users_collection.find_one({"_id": ObjectId(receiver_id)})
                    
                    if receiver:
                        message_data = {
                            "sender_id": user_id,
                            "sender_name": user["full_name"],
                            "sender_role": user["role"],
                            "receiver_id": receiver_id,
                            "receiver_name": receiver["full_name"],
                            "receiver_role": receiver["role"],
                            "message": message_content,
                            "read": False,
                            "created_at": datetime.utcnow()
                        }
                        
                        result = await chats_collection.insert_one(message_data)
                        message_data["_id"] = str(result.inserted_id)
                        
                        await manager.send_personal_message({
                            "type": "chat_message",
                            "data": serialize_doc(message_data)
                        }, receiver_id)
                        
                        await manager.send_personal_message({
                            "type": "message_sent",
                            "data": serialize_doc(message_data)
                        }, user_id)
                
                elif message_type == "typing":
                    receiver_id = data.get("receiver_id")
                    await manager.send_personal_message({
                        "type": "typing",
                        "data": {
                            "user_id": user_id,
                            "user_name": user["full_name"],
                            "is_typing": data.get("is_typing", True)
                        }
                    }, receiver_id)
                
                elif message_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    
        except WebSocketDisconnect:
            manager.disconnect(user_id)
            logger.info(f"WebSocket disconnected: {user_id}")
        except Exception as e:
            logger.error(f"WebSocket error for user {user_id}: {str(e)}")
            manager.disconnect(user_id)
            
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
        await websocket.close(code=1011, reason="Internal server error")

# ==================== FILE DOWNLOAD ENDPOINTS ====================

@app.get("/api/v1/prescriptions/{prescription_id}/download", tags=["Files"])
async def download_prescription_file(
    prescription_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get Cloudinary URL for prescription file"""
    try:
        prescription = await prescriptions_collection.find_one({"_id": ObjectId(prescription_id)})
        
        if not prescription:
            raise HTTPException(status_code=404, detail="Prescription not found")
        
        # Check access permissions
        if current_user["role"] == "patient":
            if prescription["patient_id"] != current_user["id"]:
                raise HTTPException(status_code=403, detail="Access denied")
        elif current_user["role"] == "doctor":
            has_access = await appointments_collection.find_one({
                "patient_id": prescription["patient_id"],
                "doctor_id": current_user["id"]
            })
            if not has_access:
                raise HTTPException(status_code=403, detail="Access denied")
        elif current_user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Return Cloudinary URL for download
        return {
            "download_url": prescription["file_url"],
            "file_name": prescription["file_name"],
            "file_type": prescription["file_type"],
            "file_size": prescription["file_size"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download prescription error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get download URL")
# ==================== QR CODE ENDPOINTS ====================
# Add this endpoint to your main.py (around line 2000, near other endpoints)

# Add this endpoint to your main.py (around line 2000, near other endpoints)

@app.get("/api/v1/public/patient/{patient_id}/health-card", tags=["Public"])
async def get_patient_health_card_public(patient_id: str):
    """
    PUBLIC ENDPOINT - Get patient health information via QR code scan
    No authentication required for emergency access
    Includes: Patient info, vitals, medications, appointments, AND prescriptions
    """
    try:
        # Get patient information
        patient = await users_collection.find_one({
            "_id": ObjectId(patient_id),
            "role": "patient"
        })
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        # Get latest vitals
        latest_vital = await vitals_collection.find_one(
            {"patient_id": patient_id},
            sort=[("recorded_at", -1)]
        )

        # Get active medications
        active_medications = await medications_collection.find({
            "patient_id": patient_id,
            "active": True
        }).to_list(100)

        # ✅ Get prescriptions (last 10)
        prescriptions = await prescriptions_collection.find({
            "patient_id": patient_id
        }).sort("uploaded_at", -1).limit(10).to_list(10)
        
        # Process prescriptions to include essential info
        prescription_list = []
        for presc in prescriptions:
            presc_data = {
                "id": str(presc["_id"]),
                "file_name": presc.get("file_name"),
                "file_url": presc.get("file_url"),  # Cloudinary URL
                "file_type": presc.get("file_type"),
                "doctor_name": presc.get("doctor_name"),
                "date_prescribed": presc.get("date_prescribed"),
                "uploaded_at": presc.get("uploaded_at").isoformat() if presc.get("uploaded_at") else None,
                "summary": presc.get("summary"),
                "medications": presc.get("medications", []),
                "ai_processed": presc.get("ai_processed", False),
                "notes": presc.get("notes")
            }
            prescription_list.append(presc_data)

        # Get appointment history (last 10)
        appointment_history = await appointments_collection.find({
            "patient_id": patient_id
        }).sort("created_at", -1).limit(10).to_list(10)
        
        # Enrich appointments with hospital details
        enriched_appointments = []
        for apt in appointment_history:
            apt_data = serialize_doc(apt)
            
            if apt.get("hospital_id"):
                hospital = await hospitals_collection.find_one({
                    "_id": ObjectId(apt["hospital_id"])
                })
                if hospital:
                    apt_data["hospital_details"] = {
                        "name": hospital.get("name"),
                        "address": hospital.get("address"),
                        "phone": hospital.get("phone"),
                        "is_google_hospital": hospital.get("is_real", False)
                    }
            
            enriched_appointments.append(apt_data)

        # Count Google hospital visits
        google_hospital_count = sum(
            1 for apt in enriched_appointments 
            if apt.get("hospital_details", {}).get("is_google_hospital")
        )

        # Log access for security audit
        logger.info(f"🔍 Public QR scan access for patient {patient_id} - {len(prescription_list)} prescriptions accessed")

        return {
            "patient": {
                "name": patient.get("full_name"),
                "email": patient.get("email"),
                "phone": patient.get("phone_number"),
                "blood_group": patient.get("blood_group"),
                "date_of_birth": patient.get("date_of_birth"),
                "gender": patient.get("gender"),
                "address": patient.get("address"),
                "emergency_contact": patient.get("emergency_contact")
            },
            "latest_vitals": serialize_doc(latest_vital) if latest_vital else None,
            "active_medications": [serialize_doc(m) for m in active_medications],
            "prescriptions": prescription_list,  # ✅ Added prescriptions
            "appointment_history": enriched_appointments,
            "summary": {
                "total_appointments": len(enriched_appointments),
                "google_hospital_visits": google_hospital_count,
                "active_medications": len(active_medications),
                "total_prescriptions": len(prescription_list),  # ✅ Added count
                "last_vital_check": latest_vital.get("recorded_at") if latest_vital else None
            },
            "access_info": {
                "accessed_at": datetime.utcnow().isoformat(),
                "access_type": "qr_scan",
                "security_notice": "This access has been logged for patient safety"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Public health card access error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve health information")
@app.get("/api/v1/patient/qr-code", tags=["Patient"])
async def generate_patient_qr_code(
    current_user: dict = Depends(require_role("patient"))
):
    """Generate QR code data for patient"""
    try:
        qr_data = {
            "patient_id": current_user["id"],
            "patient_name": current_user["full_name"],
            "patient_email": current_user["email"],
            "blood_group": current_user.get("blood_group"),
            "emergency_contact": current_user.get("emergency_contact"),
            "date_of_birth": current_user.get("date_of_birth"),
            "phone_number": current_user.get("phone_number"),
            "generated_at": datetime.utcnow().isoformat()
        }

        return qr_data

    except Exception as e:
        logger.error(f"Generate QR code error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate QR code")

# @app.post("/api/v1/doctor/scan-qr", tags=["Doctor"])
# async def scan_patient_qr_code(
#     patient_id: str,
#     current_user: dict = Depends(require_role("doctor"))
# ):
#     """Scan patient QR code and get complete health details"""
#     try:
#         # Get patient information
#         patient = await users_collection.find_one({
#             "_id": ObjectId(patient_id),
#             "role": "patient"
#         })
        
#         if not patient:
#             raise HTTPException(status_code=404, detail="Patient not found")

#         # Get all patient data
#         prescriptions = await prescriptions_collection.find(
#             {"patient_id": patient_id}
#         ).sort("uploaded_at", -1).to_list(100)

#         vitals = await vitals_collection.find(
#             {"patient_id": patient_id}
#         ).sort("recorded_at", -1).to_list(100)

#         appointment_history = await appointments_collection.find(
#             {"patient_id": patient_id}
#         ).sort("created_at", -1).to_list(100)

#         medications = await medications_collection.find(
#             {"patient_id": patient_id, "active": True}
#         ).to_list(100)

#         # Log the access for security
#         logger.info(f"Doctor {current_user['email']} accessed patient {patient_id} via QR scan")

#         return {
#             "patient": serialize_doc(patient),
#             "prescriptions": [serialize_doc(p) for p in prescriptions],
#             "vitals": [serialize_doc(v) for v in vitals],
#             "appointment_history": [serialize_doc(a) for a in appointment_history],
#             "active_medications": [serialize_doc(m) for m in medications]
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Scan QR code error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to scan QR code")
# # ==================== SEARCH & FILTER ENDPOINTS ====================

@app.post("/api/v1/doctor/scan-qr", tags=["Doctor"])
async def scan_patient_qr_code(
    patient_id: str,
    current_user: dict = Depends(require_role("doctor"))
):
    """Scan patient QR code and get complete health details (works with all hospitals)"""
    try:
        # Get patient information
        patient = await users_collection.find_one({
            "_id": ObjectId(patient_id),
            "role": "patient"
        })
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        # Get all patient data
        prescriptions = await prescriptions_collection.find(
            {"patient_id": patient_id}
        ).sort("uploaded_at", -1).to_list(100)

        vitals = await vitals_collection.find(
            {"patient_id": patient_id}
        ).sort("recorded_at", -1).to_list(100)

        # ✅ Get ALL appointments (including those at Google Maps hospitals)
        appointment_history = await appointments_collection.find(
            {"patient_id": patient_id}
        ).sort("created_at", -1).to_list(100)
        
        # ✅ Enrich appointment data with hospital information
        enriched_appointments = []
        for apt in appointment_history:
            apt_data = serialize_doc(apt)
            
            # Try to get hospital details from database
            if apt.get("hospital_id"):
                hospital = await hospitals_collection.find_one({"_id": ObjectId(apt["hospital_id"])})
                if hospital:
                    apt_data["hospital_details"] = {
                        "name": hospital.get("name"),
                        "address": hospital.get("address"),
                        "phone": hospital.get("phone"),
                        "is_google_hospital": hospital.get("is_real", False),
                        "place_id": hospital.get("place_id")
                    }
            
            enriched_appointments.append(apt_data)

        medications = await medications_collection.find(
            {"patient_id": patient_id, "active": True}
        ).to_list(100)

        # Log the access for security
        logger.info(f"Doctor {current_user['email']} accessed patient {patient_id} via QR scan")

        return {
            "patient": serialize_doc(patient),
            "prescriptions": [serialize_doc(p) for p in prescriptions],
            "vitals": [serialize_doc(v) for v in vitals],
            "appointment_history": enriched_appointments,
            "active_medications": [serialize_doc(m) for m in medications],
            "scan_info": {
                "scanned_by": current_user["full_name"],
                "scanned_at": datetime.utcnow().isoformat(),
                "total_appointments": len(enriched_appointments),
                "total_prescriptions": len(prescriptions),
                "total_vitals": len(vitals),
                "active_medications": len(medications)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scan QR code error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to scan QR code")
@app.get("/api/v1/search/patients", tags=["Search"])
async def search_patients(
    query: str = Query(..., min_length=2),
    current_user: dict = Depends(require_role("admin", "doctor"))
):
    """Search patients by name or email"""
    try:
        if current_user["role"] == "doctor":
            patient_ids = await appointments_collection.distinct(
                "patient_id",
                {"doctor_id": current_user["id"]}
            )
            
            patients = await users_collection.find({
                "_id": {"$in": [ObjectId(pid) for pid in patient_ids]},
                "role": "patient",
                "$or": [
                    {"full_name": {"$regex": query, "$options": "i"}},
                    {"email": {"$regex": query, "$options": "i"}}
                ]
            }).limit(20).to_list(20)
        else:
            patients = await users_collection.find({
                "role": "patient",
                "$or": [
                    {"full_name": {"$regex": query, "$options": "i"}},
                    {"email": {"$regex": query, "$options": "i"}}
                ]
            }).limit(20).to_list(20)

        return [serialize_doc(p) for p in patients]

    except Exception as e:
        logger.error(f"Search patients error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search patients")


@app.get("/api/v1/search/doctors", tags=["Search"])
async def search_doctors(
    query: str = Query(..., min_length=2),
    specialization: Optional[str] = None,
    current_user: dict = Depends(require_role("admin", "patient"))
):
    """Search doctors by name, specialization"""
    try:
        search_query = {
            "role": "doctor",
            "is_active": True,
            "$or": [
                {"full_name": {"$regex": query, "$options": "i"}},
                {"specialization": {"$regex": query, "$options": "i"}}
            ]
        }
        
        if specialization:
            search_query["specialization"] = {"$regex": specialization, "$options": "i"}

        if current_user["role"] == "admin":
            search_query["hospital_id"] = current_user.get("hospital_id")

        doctors = await users_collection.find(search_query).limit(20).to_list(20)

        return [serialize_doc(d) for d in doctors]

    except Exception as e:
        logger.error(f"Search doctors error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search doctors")

# REPLACE the existing /api/v1/search/hospitals endpoint with this:
@app.get("/api/v1/search/hospitals", tags=["Search"])
async def search_hospitals(
    query: str = Query("", min_length=0),  # Changed to allow empty string
    current_user: dict = Depends(get_current_user) if False else None  # Make auth optional
):
    """Search hospitals by name or location (public endpoint)"""
    try:
        search_query = {}
        
        if query:
            search_query["$or"] = [
                {"name": {"$regex": query, "$options": "i"}},
                {"address": {"$regex": query, "$options": "i"}}
            ]
        
        hospitals = await hospitals_collection.find(search_query).limit(50).to_list(50)
        
        # Sort pinned hospitals first
        hospitals.sort(key=lambda x: (not x.get("is_pinned", False), x.get("name", "")))
        
        return [serialize_doc(h) for h in hospitals]
        
    except Exception as e:
        logger.error(f"Search hospitals error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search hospitals")
# ==================== ANALYTICS ENDPOINTS ====================
@app.get("/api/v1/public/hospitals", tags=["Public"])
async def get_public_hospitals():
    """
    Get list of all active hospitals (no auth required)
    Used for: Doctor signup, Patient booking
    """
    try:
        # Get all hospitals that have an admin_id OR are pinned/dummy
        hospitals = await hospitals_collection.find({
            "$or": [
                {"admin_id": {"$exists": True}},  # Has an admin
                {"is_pinned": True},              # Featured hospital
                {"is_dummy": True}                # Demo hospital
            ]
        }).sort([
            ("is_pinned", -1),   # Pinned first
            ("is_dummy", -1),    # Then dummy
            ("name", 1)          # Then alphabetically
        ]).to_list(100)
        
        logger.info(f"✅ Public hospitals endpoint: Returning {len(hospitals)} hospitals")
        
        return [serialize_doc(h) for h in hospitals]
        
    except Exception as e:
        logger.error(f"❌ Get public hospitals error: {str(e)}")
        return []

@app.put("/api/v1/admin/hospitals/{hospital_id}/pin", tags=["Admin"])
async def toggle_hospital_pin(
    hospital_id: str,
    pin: bool = True,
    current_user: dict = Depends(require_role("admin"))
):
    """Toggle hospital pin status (featured/not featured)"""
    try:
        hospital = await hospitals_collection.find_one({
            "_id": ObjectId(hospital_id),
            "$or": [
                {"admin_id": current_user["id"]},
                {"_id": ObjectId(current_user.get("hospital_id"))}
            ]
        })
        
        if not hospital:
            raise HTTPException(status_code=404, detail="Hospital not found")
        
        # Update pin status
        await hospitals_collection.update_one(
            {"_id": ObjectId(hospital_id)},
            {"$set": {"is_pinned": pin, "updated_at": datetime.utcnow()}}
        )
        
        action = "pinned" if pin else "unpinned"
        logger.info(f"Hospital {hospital_id} {action} by admin {current_user['email']}")
        
        return {
            "message": f"Hospital {action} successfully",
            "hospital_id": hospital_id,
            "is_pinned": pin
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Toggle pin error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update pin status")
@app.get("/api/v1/analytics/patient-health-trends", tags=["Analytics"])
async def get_patient_health_trends(
    days: int = Query(30, ge=7, le=365),
    current_user: dict = Depends(require_role("patient"))
):
    """Get health trends for patient"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        vitals = await vitals_collection.find({
            "patient_id": current_user["id"],
            "recorded_at": {"$gte": start_date}
        }).sort("recorded_at", 1).to_list(1000)

        trends = {
            "heart_rate": [],
            "blood_pressure_systolic": [],
            "blood_pressure_diastolic": [],
            "steps": [],
            "sleep_hours": []
        }

        for vital in vitals:
            date = vital["recorded_at"].strftime("%Y-%m-%d")
            
            if vital.get("heart_rate"):
                trends["heart_rate"].append({"date": date, "value": vital["heart_rate"]})
            if vital.get("blood_pressure_systolic"):
                trends["blood_pressure_systolic"].append({"date": date, "value": vital["blood_pressure_systolic"]})
            if vital.get("blood_pressure_diastolic"):
                trends["blood_pressure_diastolic"].append({"date": date, "value": vital["blood_pressure_diastolic"]})
            if vital.get("steps"):
                trends["steps"].append({"date": date, "value": vital["steps"]})
            if vital.get("sleep_hours"):
                trends["sleep_hours"].append({"date": date, "value": vital["sleep_hours"]})

        return trends

    except Exception as e:
        logger.error(f"Get health trends error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch health trends")

# Add this to your main.py if it doesn't exist

@app.get("/api/v1/analytics/admin-reports", tags=["Analytics"])
async def get_admin_analytics_reports(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    current_user: dict = Depends(require_role("admin"))
):
    """Get analytics reports for admin"""
    try:
        hospital_id = current_user.get("hospital_id")
        if not hospital_id:
            return {
                "error": "No hospital assigned",
                "total_appointments": 0,
                "by_status": {},
                "by_doctor": {}
            }

        # Build date filter
        date_filter = {}
        if start_date:
            try:
                date_filter["$gte"] = datetime.fromisoformat(start_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format")
        if end_date:
            try:
                # Add one day to include the end date
                end_datetime = datetime.fromisoformat(end_date)
                end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
                date_filter["$lte"] = end_datetime
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format")

        # Build query
        query = {"hospital_id": hospital_id}
        if date_filter:
            query["created_at"] = date_filter

        # Fetch appointments
        appointments = await appointments_collection.find(query).to_list(10000)

        # Calculate statistics
        total = len(appointments)
        by_status = {}
        by_doctor = {}
        
        for apt in appointments:
            # Count by status
            status = apt.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1
            
            # Count by doctor (only if doctor assigned)
            if apt.get("doctor_name"):
                doctor = apt["doctor_name"]
                # Remove "Dr." prefix if exists
                doctor = doctor.replace("Dr. ", "").strip()
                by_doctor[doctor] = by_doctor.get(doctor, 0) + 1

        logger.info(f"Admin analytics: {total} appointments, {len(by_doctor)} doctors")

        return {
            "total_appointments": total,
            "by_status": by_status,
            "by_doctor": by_doctor,
            "period": {
                "start": start_date,
                "end": end_date
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get admin analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch analytics")
# ==================== ADMIN PROFILE ENDPOINTS ====================

@app.get("/api/v1/admin/profile", tags=["Admin"])
async def get_admin_profile(
    current_user: dict = Depends(require_role("admin"))
):
    """Get admin profile details"""
    return serialize_doc(current_user)

@app.put("/api/v1/admin/profile", tags=["Admin"])
async def update_admin_profile(
    profile: AdminProfile,
    current_user: dict = Depends(require_role("admin"))
):
    """Update admin profile information"""
    try:
        update_data = profile.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()

        await users_collection.update_one(
            {"_id": ObjectId(current_user["_id"])},
            {"$set": update_data}
        )

        logger.info(f"Admin profile updated: {current_user['email']}")
        return {"message": "Profile updated successfully", "updated_fields": list(update_data.keys())}

    except Exception as e:
        logger.error(f"Admin profile update error: {str(e)}")
        raise HTTPException(status_code=500, detail="Profile update failed")

@app.put("/api/v1/admin/change-password", tags=["Admin"])
async def admin_change_password(
    current_password: str,
    new_password: str,
    current_user: dict = Depends(require_role("admin"))
):
    """Change admin password"""
    try:
        # Verify current password
        user = await users_collection.find_one({"_id": ObjectId(current_user["_id"])})
        
        if not verify_password(current_password, user["password"]):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        
        # Validate new password
        if len(new_password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        
        # Update password
        hashed_new_password = hash_password(new_password)
        
        await users_collection.update_one(
            {"_id": ObjectId(current_user["_id"])},
            {"$set": {
                "password": hashed_new_password,
                "password_changed_at": datetime.utcnow()
            }}
        )
        
        logger.info(f"Admin password changed: {current_user['email']}")
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin change password error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to change password")

# ==================== HEALTH CHECK ====================

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    try:
        await users_collection.find_one()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected",
            "version": "2.0.0",
            "ai_configured": gemini_model is not None,
            "google_maps_configured": settings.GOOGLE_MAPS_API_KEY is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "database": "disconnected",
                "error": str(e)
            }
        )

@app.get("/", tags=["System"])
async def root():
    """Root endpoint"""
    return {
        "message": "Digital Health Card API with AI Integration",
        "version": "2.0.0",
        "docs": "/api/docs",
        "health": "/health",
        "features": [
            "AI-powered prescription extraction",
            "Real-time hospital tracking",
            "Smart appointment booking",
            "Patient-Admin-Doctor chat"
        ],
        "description": "CarePlus Hospital Demo System"
    }

@app.get("/api/v1/system/info", tags=["System"])
async def get_system_info():
    """Get system information"""
    try:
        total_users = await users_collection.count_documents({})
        total_hospitals = await hospitals_collection.count_documents({})
        total_appointments = await appointments_collection.count_documents({})
        total_prescriptions = await prescriptions_collection.count_documents({})

        return {
            "system": "Digital Health Card",
            "version": "2.0.0",
            "statistics": {
                "total_users": total_users,
                "total_hospitals": total_hospitals,
                "total_appointments": total_appointments,
                "total_prescriptions": total_prescriptions
            },
            "ai_features": {
                "prescription_extraction": gemini_model is not None,
                "ai_chat": gemini_model is not None,
                "hospital_tracking": settings.GOOGLE_MAPS_API_KEY is not None
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Get system info error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch system info")

# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ==================== STARTUP SCRIPT ====================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("🏥 Digital Health Card API with AI Integration")
    print("=" * 70)
    print(f"📝 API Documentation: http://localhost:8000/api/docs")
    print(f"📊 ReDoc Documentation: http://localhost:8000/api/redoc")
    print(f"❤️  Health Check: http://localhost:8000/health")
    print("=" * 70)
    print("\n🔐 CAREPLUS HOSPITAL DEMO CREDENTIALS:")
    print("   Admin:   admin@careplus.com / CareAdmin@123")
    print("   Doctor:  doctor@careplus.com / CareDoc@123")
    print("   Patient: patient@demo.com / Patient@123")
    print("=" * 70)
    print("\n✨ NEW FEATURES:")
    print("   🤖 AI Prescription Extraction with Gemini")
    print("   💬 AI Chat for Prescription Questions")
    print("   🗺️  Real-world Hospital Tracking with Google Maps")
    print("   📱 Patient-Admin-Doctor Chat System")
    print("   🏥 Pinned CarePlus Hospital Demo")
    print("=" * 70)
    
    if not settings.GEMINI_API_KEY:
        print("\n⚠️  WARNING: GEMINI_API_KEY not configured")
        print("   Set GEMINI_API_KEY in .env for AI features")
    
    if not settings.GOOGLE_MAPS_API_KEY:
        print("\n⚠️  WARNING: GOOGLE_MAPS_API_KEY not configured")
        print("   Set GOOGLE_MAPS_API_KEY in .env for hospital tracking")
    
    print("=" * 70 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )