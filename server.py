import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import smtplib
import sqlite3
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager, contextmanager
from datetime import date, datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Literal

from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field, ValidationError, field_validator

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(BASE_DIR / "farm_scans.sqlite3")))
MAX_IMAGE_BYTES = 10 * 1024 * 1024
SESSION_COOKIE = "agri_session"
SESSION_DAYS = 30
SUPPORTED_IMAGES = {
    "image/jpeg": b"\xff\xd8\xff",
    "image/png": b"\x89PNG\r\n\x1a\n",
}
VALID_LANGUAGES = {"English", "Hindi", "Marathi", "Telugu", "Tamil"}
MARKETPLACE_SEED_PRODUCTS = (
    ("basmati-rice", "Basmati Rice", "grains", 85, "per kg", 500, "🍚", "Ravi Kumar"),
    ("organic-wheat", "Organic Wheat", "grains", 35, "per kg", 500, "🌾", "Sunita Devi"),
    ("fresh-tomatoes", "Fresh Tomatoes", "vegetables", 40, "per kg", 500, "🍅", "Mohan Lal"),
    ("green-spinach", "Green Spinach", "vegetables", 25, "per bunch", 300, "🥬", "Priya Patil"),
    ("alphonso-mango", "Alphonso Mango", "fruits", 150, "per dozen", 200, "🥭", "Suresh Naik"),
    ("black-chana", "Black Chana", "pulses", 95, "per kg", 300, "🫘", "Anita More"),
    ("pomegranate", "Pomegranate", "fruits", 120, "per kg", 200, "🍎", "Ramesh Bhosale"),
    ("toor-dal", "Toor Dal", "pulses", 130, "per kg", 250, "🟡", "Vandana Shinde"),
)

def normalize_language(value: str | None) -> str:
    normalized = (value or "English").strip()
    if normalized in VALID_LANGUAGES:
        return normalized
    return "English"


@contextmanager
def database_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
    return f"pbkdf2_sha256$310000${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = stored_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations)
        ).hex()
        return hmac.compare_digest(candidate, digest_hex)
    except (ValueError, TypeError):
        return False


def create_session(user_id: int, remember: bool) -> tuple[str, datetime]:
    token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS if remember else 1)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with database_connection() as connection:
        connection.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, user_id, expires_at.isoformat()),
        )
    return token, expires_at


def current_user(request: Request) -> dict[str, object] | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    with database_connection() as connection:
        row = connection.execute(
            "SELECT users.id, users.name, users.email, COALESCE(users.preferred_language, 'English'), "
            "COALESCE(users.role, 'farmer') FROM sessions "
            "JOIN users ON users.id = sessions.user_id "
            "WHERE sessions.token_hash = ? AND sessions.expires_at > ?",
            (token_hash, now),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "name": row[1],
        "email": row[2],
        "preferred_language": row[3],
        "role": row[4],
    }


def require_user(request: Request) -> dict[str, object]:
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please sign in to continue.")
    return user


def require_role(request: Request, *roles: str) -> dict[str, object]:
    user = require_user(request)
    if user["role"] not in roles:
        raise HTTPException(status_code=403, detail="Your account role cannot access this feature.")
    return user


def set_session_cookie(response: Response, token: str, expires_at: datetime, remember: bool) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=int((expires_at - datetime.now(timezone.utc)).total_seconds()),
        expires=expires_at,
        httponly=True,
        samesite="lax",
        secure=os.getenv("COOKIE_SECURE", "false").strip().lower() == "true",
    )


def send_password_reset_email(email: str, otp_code: str, language: str | None) -> bool:
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    if not smtp_host:
        return False
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
    sender = os.getenv("SMTP_FROM", smtp_user or "noreply@agrismart.local")
    message = EmailMessage()
    message["Subject"] = "AgriSmart password reset code"
    message["From"] = sender
    message["To"] = email
    translated_label = normalize_language(language)
    message.set_content(
        f"Your AgriSmart OTP is {otp_code}. It expires in 30 minutes. "
        f"Preferred language: {translated_label}."
    )
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            if os.getenv("SMTP_USE_TLS", "true").lower() == "true":
                server.starttls()
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.send_message(message)
        return True
    except Exception:
        return False


class Diagnosis(BaseModel):
    disease: str
    cropDetected: str
    confidence: int = Field(ge=0, le=100)
    severity: Literal["High", "Medium", "Low", "Healthy"]
    urgency: str
    description: str
    symptoms: list[str]
    medicines: list[str]
    prevention: list[str]
    spreadRisk: int = Field(ge=0, le=100)
    yieldLoss: int = Field(ge=0, le=100)
    soilImpact: int = Field(ge=0, le=100)
    treatmentCostPerAcre: float = Field(ge=0)
    lossWithoutTreatment: float = Field(ge=0)
    savingsWithTreatment: float = Field(ge=0)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=10)


class WeatherRequest(BaseModel):
    city: str = Field(min_length=1, max_length=100)


class SignupRequest(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: str = Field(min_length=5, max_length=160)
    password: str = Field(min_length=8, max_length=128)
    language: str = "English"
    role: Literal["farmer", "customer"] = "farmer"

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Name cannot be blank.")
        return value.strip()

    @field_validator("email")
    @classmethod
    def clean_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            raise ValueError("Enter a valid email address.")
        return value

    @field_validator("language")
    @classmethod
    def clean_language(cls, value: str) -> str:
        return normalize_language(value)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=160)
    password: str = Field(min_length=1, max_length=128)
    remember: bool = False
    language: str | None = None

    @field_validator("email")
    @classmethod
    def clean_email(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("language")
    @classmethod
    def clean_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_language(value)


class ForgotPasswordRequest(BaseModel):
    email: str = Field(min_length=5, max_length=160)
    language: str | None = None

    @field_validator("email")
    @classmethod
    def clean_email(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("language")
    @classmethod
    def clean_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_language(value)


class ResetPasswordRequest(BaseModel):
    token: str | None = None
    otp: str | None = None
    email: str | None = None
    password: str = Field(min_length=8, max_length=128)


class FarmFieldCreate(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    crop: str = Field(min_length=2, max_length=60)
    season: Literal["Kharif", "Rabi", "Zaid", "Perennial", "Other"] = "Other"
    area_acres: float = Field(gt=0, le=10000)
    location: str = Field(min_length=2, max_length=120)
    status: str = Field(min_length=2, max_length=40)


class FarmProfileUpdate(BaseModel):
    location: str = Field(default="", max_length=120)
    primary_crop: str = Field(default="", max_length=60)
    requirements: str = Field(default="", max_length=500)
    is_in_need: bool = False
    farm_size_acres: float | None = Field(default=None, gt=0, le=1_000_000)
    crops_grown: str = Field(default="", max_length=300)
    preferred_contact: Literal["in_app", "email", "phone"] = "in_app"
    preferred_language: Literal["English", "Hindi", "Marathi", "Telugu", "Tamil"] = "English"

    @field_validator("location", "primary_crop", "requirements")
    @classmethod
    def clean_profile_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("crops_grown")
    @classmethod
    def clean_crops_grown(cls, value: str) -> str:
        crops = [crop.strip() for crop in value.split(",") if crop.strip()]
        return ", ".join(dict.fromkeys(crops))


class FarmerNeedStatusUpdate(BaseModel):
    status: Literal["active", "paused", "fulfilled"]


class FarmerNeedResponseCreate(BaseModel):
    message: str = Field(min_length=5, max_length=1000)

    @field_validator("message")
    @classmethod
    def clean_message(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 5:
            raise ValueError("Please write a little more in your response.")
        return value


class FarmCropCreate(BaseModel):
    crop_name: str = Field(min_length=2, max_length=60)
    season: Literal["Kharif", "Rabi", "Zaid", "Perennial", "Other"] = "Other"
    variety: str = Field(default="", max_length=80)
    sowing_date: date
    expected_harvest_date: date | None = None
    quantity_kg: float = Field(ge=0, le=1_000_000_000)
    expenses: float = Field(ge=0, le=1_000_000_000)
    sales_income: float = Field(ge=0, le=1_000_000_000)
    health: Literal["Healthy", "Watch", "At Risk"] = "Healthy"
    fertilizer_notes: str = Field(default="", max_length=1000)
    pesticide_notes: str = Field(default="", max_length=1000)


class FarmTaskCreate(BaseModel):
    field_id: int | None = None
    title: str = Field(min_length=2, max_length=120)
    due_date: date | None = None
    priority: Literal["Low", "Medium", "High"] = "Medium"
    status: Literal["Pending", "In Progress", "Done"] = "Pending"
    notes: str | None = None


class FarmTaskStatusUpdate(BaseModel):
    status: Literal["Pending", "In Progress", "Done"]


class AdminUserRoleUpdate(BaseModel):
    role: Literal["farmer", "customer", "admin"]


class AdminProductStatusUpdate(BaseModel):
    active: bool


class AdminOrderStatusUpdate(BaseModel):
    status: Literal["confirmed", "completed", "cancelled"]


class PriceObservationCreate(BaseModel):
    crop: Literal["cotton", "soybean", "wheat"]
    market: str = Field(min_length=2, max_length=100)
    price_per_quintal: float = Field(gt=0, le=10000000)
    reported_on: date
    source: str = Field(min_length=2, max_length=160)

    @field_validator("market", "source")
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("This field cannot be blank.")
        return value.strip()


class MarketplaceProductCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    category: Literal["grains", "vegetables", "fruits", "pulses"]
    price: float = Field(gt=0, le=10_000_000)
    unit: str = Field(min_length=2, max_length=30)
    quantity: int = Field(gt=0, le=1_000_000)
    emoji: str = Field(default="🌾", max_length=8)


class MarketplaceOrderLine(BaseModel):
    product_id: int = Field(gt=0)
    quantity: int = Field(gt=0, le=10_000)


class MarketplaceOrderCreate(BaseModel):
    items: list[MarketplaceOrderLine] = Field(min_length=1, max_length=30)


@asynccontextmanager
async def lifespan(_: FastAPI):
    with database_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                preferred_language TEXT NOT NULL DEFAULT 'English',
                role TEXT NOT NULL DEFAULT 'farmer',
                created_at TEXT NOT NULL
            )
            """
        )
        columns = connection.execute("PRAGMA table_info(users)").fetchall()
        has_language = any(column[1] == "preferred_language" for column in columns)
        if not has_language:
            connection.execute(
                "ALTER TABLE users ADD COLUMN preferred_language TEXT NOT NULL DEFAULT 'English'"
            )
        has_role = any(column[1] == "role" for column in columns)
        if not has_role:
            connection.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'farmer'")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS farmer_profiles (
                user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                location TEXT NOT NULL DEFAULT '',
                primary_crop TEXT NOT NULL DEFAULT '',
                requirements TEXT NOT NULL DEFAULT '',
                is_in_need INTEGER NOT NULL DEFAULT 0,
                farm_size_acres REAL,
                crops_grown TEXT NOT NULL DEFAULT '',
                preferred_contact TEXT NOT NULL DEFAULT 'in_app',
                need_status TEXT NOT NULL DEFAULT 'paused'
            )
            """
        )
        profile_columns = {
            column[1] for column in connection.execute("PRAGMA table_info(farmer_profiles)").fetchall()
        }
        for column_name, definition in (
            ("farm_size_acres", "REAL"),
            ("crops_grown", "TEXT NOT NULL DEFAULT ''"),
            ("preferred_contact", "TEXT NOT NULL DEFAULT 'in_app'"),
            ("need_status", "TEXT NOT NULL DEFAULT 'paused'"),
        ):
            if column_name not in profile_columns:
                connection.execute(f"ALTER TABLE farmer_profiles ADD COLUMN {column_name} {definition}")
        connection.execute(
            "UPDATE farmer_profiles SET need_status = 'active' "
            "WHERE is_in_need = 1 AND need_status = 'paused'"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS farmer_need_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farmer_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                responder_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                read_at TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_farmer_need_responses_farmer "
            "ON farmer_need_responses (farmer_id, created_at DESC)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS password_resets (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TEXT NOT NULL,
                used INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS password_otps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                otp_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS farm_fields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                crop TEXT NOT NULL,
                season TEXT NOT NULL DEFAULT 'Other',
                area_acres REAL NOT NULL CHECK (area_acres > 0),
                location TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        field_columns = connection.execute("PRAGMA table_info(farm_fields)").fetchall()
        if not any(column[1] == "season" for column in field_columns):
            connection.execute(
                "ALTER TABLE farm_fields ADD COLUMN season TEXT NOT NULL DEFAULT 'Other'"
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS farm_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                field_id INTEGER REFERENCES farm_fields(id) ON DELETE SET NULL,
                title TEXT NOT NULL,
                due_date TEXT,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS farm_crops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                crop_name TEXT NOT NULL,
                season TEXT NOT NULL DEFAULT 'Other',
                variety TEXT NOT NULL DEFAULT '',
                sowing_date TEXT NOT NULL,
                expected_harvest_date TEXT,
                quantity_kg REAL NOT NULL DEFAULT 0,
                expenses REAL NOT NULL DEFAULT 0,
                sales_income REAL NOT NULL DEFAULT 0,
                health TEXT NOT NULL DEFAULT 'Healthy',
                fertilizer_notes TEXT NOT NULL DEFAULT '',
                pesticide_notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        crop_columns = connection.execute("PRAGMA table_info(farm_crops)").fetchall()
        if not any(column[1] == "season" for column in crop_columns):
            connection.execute(
                "ALTER TABLE farm_crops ADD COLUMN season TEXT NOT NULL DEFAULT 'Other'"
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                image BLOB NOT NULL,
                media_type TEXT NOT NULL,
                diagnosis TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS market_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crop TEXT NOT NULL,
                market TEXT NOT NULL,
                price_per_quintal REAL NOT NULL CHECK (price_per_quintal > 0),
                reported_on TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_prices_crop_reported_on "
            "ON market_prices (crop, reported_on, id)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS marketplace_products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seed_key TEXT UNIQUE,
                seller_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                seller_name TEXT NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL CHECK (price > 0),
                unit TEXT NOT NULL,
                quantity INTEGER NOT NULL CHECK (quantity >= 0),
                emoji TEXT NOT NULL DEFAULT '🌾',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
            """
        )
        product_columns = connection.execute("PRAGMA table_info(marketplace_products)").fetchall()
        if not any(column[1] == "is_active" for column in product_columns):
            connection.execute(
                "ALTER TABLE marketplace_products ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1"
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS marketplace_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                buyer_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                total REAL NOT NULL CHECK (total >= 0),
                status TEXT NOT NULL DEFAULT 'placed',
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS marketplace_order_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL REFERENCES marketplace_orders(id) ON DELETE CASCADE,
                product_id INTEGER REFERENCES marketplace_products(id) ON DELETE SET NULL,
                product_name TEXT NOT NULL,
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                unit_price REAL NOT NULL CHECK (unit_price > 0)
            )
            """
        )
        for seed_key, name, category, price, unit, quantity, emoji, seller_name in MARKETPLACE_SEED_PRODUCTS:
            connection.execute(
                """
                INSERT OR IGNORE INTO marketplace_products
                    (seed_key, seller_name, name, category, price, unit, quantity, emoji, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    seed_key,
                    seller_name,
                    name,
                    category,
                    price,
                    unit,
                    quantity,
                    emoji,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        admin_email = os.getenv("ADMIN_EMAIL", "").strip().lower()
        admin_password = os.getenv("ADMIN_PASSWORD", "")
        if admin_email and admin_password:
            existing_admin = connection.execute(
                "SELECT id FROM users WHERE email = ?", (admin_email,)
            ).fetchone()
            if existing_admin:
                connection.execute("UPDATE users SET role = 'admin' WHERE id = ?", (existing_admin[0],))
            else:
                connection.execute(
                    "INSERT INTO users (name, email, password_hash, preferred_language, role, created_at) "
                    "VALUES (?, ?, ?, 'English', 'admin', ?)",
                    (
                        os.getenv("ADMIN_NAME", "AgriSmart Admin"),
                        admin_email,
                        hash_password(admin_password),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
    yield


app = FastAPI(title="AgriSmart API", lifespan=lifespan)


def get_ai_credentials() -> tuple[str, str]:
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        return "gemini", gemini_key
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        return "anthropic", anthropic_key
    raise HTTPException(
        status_code=503,
        detail=(
            "AI is not configured. Add GEMINI_API_KEY or ANTHROPIC_API_KEY "
            "to the local .env file and restart the server."
        ),
    )


def gemini_contents(messages: list[dict[str, object]]) -> list[types.Content]:
    contents: list[types.Content] = []
    for message in messages:
        raw_content = message.get("content", "")
        blocks = raw_content if isinstance(raw_content, list) else [{"type": "text", "text": raw_content}]
        parts: list[types.Part] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(types.Part.from_text(text=str(block.get("text", ""))))
            elif block.get("type") == "image":
                source = block.get("source")
                if isinstance(source, dict) and source.get("type") == "base64":
                    parts.append(
                        types.Part.from_bytes(
                            data=base64.b64decode(str(source.get("data", ""))),
                            mime_type=str(source.get("media_type", "image/jpeg")),
                        )
                    )
        if parts:
            role = "model" if message.get("role") == "assistant" else "user"
            contents.append(types.Content(role=role, parts=parts))
    return contents


async def ask_ai(
    *,
    messages: list[dict[str, object]],
    max_tokens: int,
    system: str | None = None,
) -> str:
    provider, api_key = get_ai_credentials()
    if provider == "gemini":
        client = genai.Client(api_key=api_key)
        try:
            response = await client.aio.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
                contents=gemini_contents(messages),
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    system_instruction=system or None,
                ),
            )
            return response.text or ""
        except APIError as exc:
            raise HTTPException(status_code=502, detail="The Gemini service rejected the request.") from exc
        finally:
            await client.aio.aclose()

    client = AsyncAnthropic(api_key=api_key)
    try:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=system or "",
            messages=messages,
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
    except APIConnectionError as exc:
        raise HTTPException(status_code=502, detail="Could not connect to the AI service.") from exc
    except APIStatusError as exc:
        raise HTTPException(status_code=502, detail="The AI service rejected the request.") from exc
    finally:
        await client.close()


def parse_json_response(text: str) -> dict[str, object]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="The AI service returned an invalid report.") from exc
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="The AI service returned an invalid report.")
    return result


def weather_code_to_label(code: int) -> tuple[str, str]:
    mapping = {
        0: ("Clear", "☀️"),
        1: ("Mostly Clear", "🌤️"),
        2: ("Partly Cloudy", "⛅"),
        3: ("Cloudy", "☁️"),
        45: ("Foggy", "🌫️"),
        48: ("Foggy", "🌫️"),
        51: ("Light Drizzle", "🌦️"),
        53: ("Drizzle", "🌦️"),
        55: ("Heavy Drizzle", "🌧️"),
        56: ("Freezing Drizzle", "🌧️"),
        57: ("Freezing Drizzle", "🌧️"),
        61: ("Light Rain", "🌦️"),
        63: ("Rain", "🌧️"),
        65: ("Heavy Rain", "⛈️"),
        66: ("Freezing Rain", "🌧️"),
        67: ("Freezing Rain", "🌧️"),
        71: ("Light Snow", "🌨️"),
        73: ("Snow", "❄️"),
        75: ("Heavy Snow", "❄️"),
        77: ("Snow Grains", "❄️"),
        80: ("Rain Showers", "🌦️"),
        81: ("Heavy Rain Showers", "🌧️"),
        82: ("Violent Rain Showers", "⛈️"),
        85: ("Snow Showers", "🌨️"),
        86: ("Heavy Snow Showers", "🌨️"),
        95: ("Thunderstorm", "⛈️"),
        96: ("Thunderstorm", "⛈️"),
        99: ("Thunderstorm", "⛈️"),
    }
    return mapping.get(code, ("Weather", "🌤️"))


def get_live_weather(city: str) -> dict[str, object] | None:
    try:
        geocode_query = urllib.parse.quote(city)
        geocode_url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            f"name={geocode_query}&count=1&language=en&format=json"
        )
        with urllib.request.urlopen(geocode_url, timeout=10) as response:
            geocode_payload = json.loads(response.read().decode("utf-8"))
        results = geocode_payload.get("results") or []
        if not results:
            return None
        location = results[0]
        lat = location.get("latitude")
        lon = location.get("longitude")
        if lat is None or lon is None:
            return None
        forecast_url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code"
            "&timezone=auto"
        )
        with urllib.request.urlopen(forecast_url, timeout=10) as response:
            forecast_payload = json.loads(response.read().decode("utf-8"))
        current = forecast_payload.get("current") or {}
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        precipitation = current.get("precipitation")
        weather_code = int(current.get("weather_code", 0) or 0)
        condition, icon = weather_code_to_label(weather_code)
        risk = "low"
        risk_title = "Low disease pressure"
        details = "Live weather data indicates a moderate or manageable risk."
        if precipitation is not None and precipitation > 4 or humidity is not None and humidity > 85:
            risk = "high"
            risk_title = "High disease risk"
            details = "Recent rainfall and humidity are favourable for fungal disease spread."
        elif precipitation is not None and precipitation > 1 or humidity is not None and humidity > 70:
            risk = "medium"
            risk_title = "Moderate disease risk"
            details = "Humidity and moisture are rising; inspect the crop canopy for early symptoms."
        if temp is not None and temp > 35:
            risk = "high" if risk == "low" else risk
            risk_title = "Heat stress warning" if risk == "low" else risk_title
            details = "Heat and dry stress may elevate crop stress; schedule irrigation and monitor leaf wilting."
        return {
            "city": str(location.get("name") or city),
            "temp": temp,
            "feels": temp,
            "humidity": humidity,
            "wind": wind,
            "condition": condition,
            "icon": icon,
            "rainfall": precipitation,
            "uvIndex": 0,
            "riskLevel": risk,
            "riskTitle": risk_title,
            "riskDetails": details + " This is current live weather data from Open-Meteo; it is not a field sensor reading.",
        }
    except Exception:
        return None


def get_seven_day_forecast(city: str) -> dict[str, object]:
    geocode_url = (
        "https://geocoding-api.open-meteo.com/v1/search?"
        f"name={urllib.parse.quote(city)}&count=10&language=en&format=json&countryCode=IN"
    )
    try:
        with urllib.request.urlopen(geocode_url, timeout=8) as response:
            geocode_payload = json.loads(response.read().decode("utf-8"))
        results = geocode_payload.get("results") or []
        if not results:
            raise HTTPException(status_code=404, detail="Village or town not found. Try a nearby town name.")
        location = next(
            (result for result in results if str(result.get("name", "")).casefold() == city.casefold()),
            results[0],
        )
        latitude = location.get("latitude")
        longitude = location.get("longitude")
        if latitude is None or longitude is None:
            raise HTTPException(status_code=502, detail="Weather location lookup returned incomplete data.")

        forecast_url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={latitude}&longitude={longitude}&forecast_days=7&timezone=auto"
            "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
            "precipitation_probability_max,wind_speed_10m_max"
        )
        with urllib.request.urlopen(forecast_url, timeout=8) as response:
            forecast_payload = json.loads(response.read().decode("utf-8"))
    except HTTPException:
        raise
    except (OSError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="The 7-day forecast is temporarily unavailable. Please try again later.",
        ) from exc

    daily = forecast_payload.get("daily") or {}
    dates = daily.get("time") or []
    forecast_days = []
    for index, forecast_date in enumerate(dates[:7]):
        try:
            code = int(daily["weather_code"][index])
            high = float(daily["temperature_2m_max"][index])
            low = float(daily["temperature_2m_min"][index])
            rainfall = float(daily["precipitation_sum"][index])
            rain_chance = float(daily["precipitation_probability_max"][index])
            wind = float(daily["wind_speed_10m_max"][index])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail="The weather provider returned incomplete forecast data.") from exc
        condition, icon = weather_code_to_label(code)
        forecast_days.append(
            {
                "date": forecast_date,
                "condition": condition,
                "icon": icon,
                "high_c": round(high, 1),
                "low_c": round(low, 1),
                "rain_mm": round(rainfall, 1),
                "rain_chance": round(rain_chance),
                "wind_kmh": round(wind, 1),
                "weather_code": code,
            }
        )
    if len(forecast_days) != 7:
        raise HTTPException(status_code=502, detail="The weather provider did not return seven forecast days.")

    alerts: list[dict[str, str]] = []
    for forecast in forecast_days:
        if forecast["rain_mm"] >= 64.5:
            alerts.append(
                {
                    "level": "warning",
                    "date": str(forecast["date"]),
                    "title": "Heavy rain possible",
                    "detail": f"{forecast['rain_mm']} mm is forecast. Check local advisories and drainage.",
                }
            )
        if forecast["weather_code"] in {95, 96, 99}:
            alerts.append(
                {
                    "level": "warning",
                    "date": str(forecast["date"]),
                    "title": "Thunderstorm forecast",
                    "detail": "Storm forecasts do not confirm hail; follow official local warnings.",
                }
            )
        if forecast["low_c"] <= 5:
            alerts.append(
                {
                    "level": "warning",
                    "date": str(forecast["date"]),
                    "title": "Cold conditions possible",
                    "detail": f"Minimum temperature may reach {forecast['low_c']}°C. Check crop-specific local guidance.",
                }
            )
    if all(forecast["rain_mm"] < 1 for forecast in forecast_days):
        alerts.append(
            {
                "level": "notice",
                "date": str(forecast_days[0]["date"]),
                "title": "Little or no rain forecast",
                "detail": "No meaningful rain is forecast in this 7-day outlook; check soil moisture and your local irrigation plan.",
            }
        )

    return {
        "location": {
            "name": str(location.get("name") or city),
            "admin1": str(location.get("admin1") or ""),
        },
        "source": "Open-Meteo",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "days": forecast_days,
        "alerts": alerts,
    }


def image_signature_matches(media_type: str, image_bytes: bytes) -> bool:
    if media_type == "image/webp":
        return image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP"
    signature = SUPPORTED_IMAGES.get(media_type)
    return signature is not None and image_bytes.startswith(signature)


def scan_to_dict(row: tuple[object, ...]) -> dict[str, object]:
    scan_id, created_at, media_type, diagnosis_json = row
    timestamp = datetime.fromisoformat(str(created_at)).astimezone()
    return {
        "id": scan_id,
        "date": timestamp.strftime("%d/%m/%Y"),
        "time": timestamp.strftime("%H:%M"),
        "image": f"/api/scans/{scan_id}/image",
        "mediaType": media_type,
        "result": json.loads(str(diagnosis_json)),
    }


def market_price_summary(crop: str) -> dict[str, object]:
    with database_connection() as connection:
        rows = connection.execute(
            "SELECT id, market, price_per_quintal, reported_on, source "
            "FROM market_prices WHERE crop = ? ORDER BY reported_on, id",
            (crop,),
        ).fetchall()

    observations = [
        {
            "id": row[0],
            "market": row[1],
            "price_per_quintal": row[2],
            "reported_on": row[3],
            "source": row[4],
        }
        for row in rows
    ]
    if not observations:
        return {
            "crop": crop,
            "market": None,
            "observations": [],
            "latest_price": None,
            "previous_price": None,
            "forecast_30d": None,
            "trend": None,
            "source": None,
        }

    latest = observations[-1]
    same_market = [item for item in observations if item["market"] == latest["market"]]
    market_observations = [
        item for item in same_market if item["reported_on"] <= latest["reported_on"]
    ]
    previous = next(
        (
            item
            for item in reversed(market_observations[:-1])
            if item["reported_on"] < latest["reported_on"]
        ),
        None,
    )
    forecast = None
    trend = None
    if previous:
        days_between = (date.fromisoformat(latest["reported_on"]) - date.fromisoformat(previous["reported_on"])).days
        daily_change = (latest["price_per_quintal"] - previous["price_per_quintal"]) / days_between
        forecast = round(max(0, latest["price_per_quintal"] + daily_change * 30), 2)
        change_ratio = (latest["price_per_quintal"] - previous["price_per_quintal"]) / previous["price_per_quintal"]
        trend = "up" if change_ratio > 0.02 else "down" if change_ratio < -0.02 else "stable"

    return {
        "crop": crop,
        "market": latest["market"],
        "observations": market_observations,
        "latest_price": latest["price_per_quintal"],
        "previous_price": previous["price_per_quintal"] if previous else None,
        "forecast_30d": forecast,
        "trend": trend,
        "source": latest["source"],
    }


def live_market_alerts() -> list[dict[str, object]]:
    alerts: list[dict[str, object]] = []
    for crop in ("cotton", "soybean", "wheat"):
        summary = market_price_summary(crop)
        price = summary["latest_price"]
        previous = summary["previous_price"]
        if price is None:
            alerts.append(
                {
                    "crop": crop,
                    "market": None,
                    "price_per_quintal": None,
                    "change_percent": None,
                    "trend": None,
                    "notification": f"No {crop} observations yet. Add a dated market observation to see price changes.",
                    "source": None,
                    "reported_on": None,
                    "last_updated": None,
                }
            )
            continue

        if previous is None:
            change_percent = None
            status = None
            notification = f"One {crop} observation is available. Add another dated observation to calculate a price trend."
        else:
            change_percent = round(((price - previous) / previous) * 100, 2)
            if abs(change_percent) >= 6:
                status = "up" if change_percent > 0 else "down"
                notification = (
                    f"{crop.title()} reported prices are {'rising sharply' if status == 'up' else 'falling sharply'} at {change_percent:+.2f}%."
                    f" {'Consider holding stock for higher returns.' if status == 'up' else 'Review sales timing and protect margin.'}"
                )
            elif abs(change_percent) >= 2:
                status = "up" if change_percent > 0 else "down"
                notification = (
                    f"{crop.title()} reported prices are moving {('upward' if status == 'up' else 'downward')} at {change_percent:+.2f}%. "
                    "Use the report dates and source when making selling decisions."
                )
            else:
                status = "stable"
                notification = (
                    f"{crop.title()} reported price is ₹{int(price):,} per quintal. "
                    "There is not yet a significant change between the latest reports."
                )

        alerts.append(
            {
                "crop": crop,
                "market": summary.get("market"),
                "price_per_quintal": round(float(price), 2),
                "change_percent": change_percent,
                "trend": status,
                "notification": notification,
                "source": summary.get("source"),
                "reported_on": summary["observations"][-1]["reported_on"],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
        )

    return alerts


@app.get("/api/health")
def health() -> dict[str, object]:
    provider = "gemini" if os.getenv("GEMINI_API_KEY", "").strip() else "anthropic" if os.getenv("ANTHROPIC_API_KEY", "").strip() else None
    return {"status": "ok", "aiConfigured": provider is not None, "aiProvider": provider}


@app.post("/api/auth/signup")
def signup(request: SignupRequest) -> Response:
    role = request.role
    with database_connection() as connection:
        existing = connection.execute("SELECT id FROM users WHERE email = ?", (request.email,)).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="An account with this email already exists.")
        cursor = connection.execute(
            "INSERT INTO users (name, email, password_hash, preferred_language, role, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                request.name,
                request.email,
                hash_password(request.password),
                normalize_language(request.language),
                role,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        user_id = cursor.lastrowid
    token, expires_at = create_session(user_id, True)
    response = Response(
        content=json.dumps(
            {
                "message": "Account created",
                "user": {
                    "name": request.name,
                    "email": request.email,
                    "preferred_language": normalize_language(request.language),
                    "role": role,
                },
            }
        ),
        media_type="application/json",
    )
    set_session_cookie(response, token, expires_at, True)
    return response


@app.post("/api/auth/login")
def login(request: LoginRequest) -> Response:
    with database_connection() as connection:
        row = connection.execute(
            "SELECT id, name, email, password_hash, COALESCE(preferred_language, 'English'), "
            "COALESCE(role, 'farmer') FROM users WHERE email = ?",
            (request.email,),
        ).fetchone()
    if not row or not verify_password(request.password, row[3]):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    preferred_language = normalize_language(request.language or row[4])
    if preferred_language != row[4]:
        with database_connection() as connection:
            connection.execute(
                "UPDATE users SET preferred_language = ? WHERE id = ?",
                (preferred_language, row[0]),
            )
    token, expires_at = create_session(row[0], request.remember)
    response = Response(
        content=json.dumps(
            {
                "message": "Signed in",
                "user": {
                    "name": row[1],
                    "email": row[2],
                    "preferred_language": preferred_language,
                    "role": row[5],
                },
            }
        ),
        media_type="application/json",
    )
    set_session_cookie(response, token, expires_at, request.remember)
    return response


@app.get("/api/auth/me")
def me(request: Request) -> dict[str, object]:
    return {"user": require_user(request)}


@app.get("/api/farm/profile")
def get_farm_profile(request: Request) -> dict[str, object]:
    user = require_role(request, "farmer", "admin")
    with database_connection() as connection:
        row = connection.execute(
            "SELECT location, primary_crop, requirements, is_in_need, farm_size_acres, "
            "crops_grown, preferred_contact, need_status "
            "FROM farmer_profiles WHERE user_id = ?",
            (user["id"],),
        ).fetchone()
    if row is None:
        return {
            "location": "",
            "primary_crop": "",
            "requirements": "",
            "is_in_need": False,
            "farm_size_acres": None,
            "crops_grown": "",
            "preferred_contact": "in_app",
            "need_status": "paused",
            "preferred_language": user["preferred_language"],
        }
    return {
        "location": row[0],
        "primary_crop": row[1],
        "requirements": row[2],
        "is_in_need": bool(row[3]),
        "farm_size_acres": row[4],
        "crops_grown": row[5],
        "preferred_contact": row[6],
        "need_status": row[7],
        "preferred_language": user["preferred_language"],
    }


@app.put("/api/farm/profile")
def update_farm_profile(request: Request, payload: FarmProfileUpdate) -> dict[str, object]:
    user = require_role(request, "farmer", "admin")
    profile = payload.model_dump()
    if profile["is_in_need"] and not profile["requirements"]:
        raise HTTPException(status_code=422, detail="Add your requirements before publishing a need.")
    with database_connection() as connection:
        connection.execute(
            """
            INSERT INTO farmer_profiles (
                user_id, location, primary_crop, requirements, is_in_need, farm_size_acres,
                crops_grown, preferred_contact, need_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                location = excluded.location,
                primary_crop = excluded.primary_crop,
                requirements = excluded.requirements,
                is_in_need = excluded.is_in_need,
                farm_size_acres = excluded.farm_size_acres,
                crops_grown = excluded.crops_grown,
                preferred_contact = excluded.preferred_contact,
                need_status = CASE
                    WHEN excluded.is_in_need = 1 THEN 'active'
                    WHEN farmer_profiles.need_status = 'fulfilled' THEN 'fulfilled'
                    ELSE 'paused'
                END
            """,
            (
                user["id"],
                profile["location"],
                profile["primary_crop"],
                profile["requirements"],
                int(profile["is_in_need"]),
                profile["farm_size_acres"],
                profile["crops_grown"],
                profile["preferred_contact"],
                "active" if profile["is_in_need"] else "paused",
            ),
        )
        connection.execute(
            "UPDATE users SET preferred_language = ? WHERE id = ?",
            (profile["preferred_language"], user["id"]),
        )
        status_row = connection.execute(
            "SELECT need_status FROM farmer_profiles WHERE user_id = ?",
            (user["id"],),
        ).fetchone()
    return {**profile, "need_status": status_row[0]}


@app.patch("/api/farm/profile/need-status")
def update_farmer_need_status(
    request: Request, payload: FarmerNeedStatusUpdate
) -> dict[str, str]:
    user = require_role(request, "farmer", "admin")
    with database_connection() as connection:
        profile = connection.execute(
            "SELECT requirements FROM farmer_profiles WHERE user_id = ?",
            (user["id"],),
        ).fetchone()
        if payload.status == "active" and (profile is None or not profile[0].strip()):
            raise HTTPException(status_code=422, detail="Add your requirements before publishing a need.")
        connection.execute(
            """
            INSERT INTO farmer_profiles (user_id, is_in_need, need_status)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                is_in_need = excluded.is_in_need,
                need_status = excluded.need_status
            """,
            (user["id"], int(payload.status == "active"), payload.status),
        )
    return {"status": payload.status}


@app.get("/api/farm/needs")
def list_farmer_needs(request: Request) -> list[dict[str, object]]:
    user = require_user(request)
    with database_connection() as connection:
        rows = connection.execute(
            """
            SELECT p.user_id, u.name, p.location, p.primary_crop, p.crops_grown,
                   p.farm_size_acres, p.requirements, p.preferred_contact
            FROM farmer_profiles p
            JOIN users u ON u.id = p.user_id
            WHERE p.need_status = 'active' AND p.is_in_need = 1
              AND length(trim(p.requirements)) > 0
            ORDER BY p.user_id = ? DESC, u.name COLLATE NOCASE
            """,
            (user["id"],),
        ).fetchall()
    return [
        {
            "farmer_id": row[0],
            "farmer_name": row[1],
            "location": row[2],
            "primary_crop": row[3],
            "crops_grown": row[4],
            "farm_size_acres": row[5],
            "requirements": row[6],
            "preferred_contact": row[7],
            "is_own_need": row[0] == user["id"],
        }
        for row in rows
    ]


@app.post("/api/farm/needs/{farmer_id}/responses")
def create_farmer_need_response(
    farmer_id: int, request: Request, payload: FarmerNeedResponseCreate
) -> dict[str, object]:
    responder = require_user(request)
    if farmer_id == responder["id"]:
        raise HTTPException(status_code=400, detail="You cannot respond to your own need.")
    now = datetime.now(timezone.utc).isoformat()
    with database_connection() as connection:
        active_need = connection.execute(
            "SELECT 1 FROM farmer_profiles WHERE user_id = ? AND need_status = 'active' "
            "AND is_in_need = 1 AND length(trim(requirements)) > 0",
            (farmer_id,),
        ).fetchone()
        if active_need is None:
            raise HTTPException(status_code=404, detail="This farmer need is no longer available.")
        cursor = connection.execute(
            """
            INSERT INTO farmer_need_responses (farmer_id, responder_id, message, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (farmer_id, responder["id"], payload.message, now),
        )
    return {"id": cursor.lastrowid, "created_at": now, "message": payload.message}


@app.get("/api/farm/needs/responses")
def list_farmer_need_responses(request: Request) -> dict[str, object]:
    farmer = require_role(request, "farmer", "admin")
    with database_connection() as connection:
        unread_count = connection.execute(
            "SELECT COUNT(*) FROM farmer_need_responses WHERE farmer_id = ? AND read_at IS NULL",
            (farmer["id"],),
        ).fetchone()[0]
        rows = connection.execute(
            """
            SELECT r.id, u.name, r.message, r.created_at, r.read_at
            FROM farmer_need_responses r
            JOIN users u ON u.id = r.responder_id
            WHERE r.farmer_id = ?
            ORDER BY r.created_at DESC, r.id DESC
            LIMIT 50
            """,
            (farmer["id"],),
        ).fetchall()
    return {
        "unread_count": unread_count,
        "responses": [
            {
                "id": row[0],
                "responder_name": row[1],
                "message": row[2],
                "created_at": row[3],
                "read": row[4] is not None,
            }
            for row in rows
        ],
    }


@app.patch("/api/farm/needs/responses/read")
def mark_farmer_need_responses_read(request: Request) -> dict[str, int]:
    farmer = require_role(request, "farmer", "admin")
    with database_connection() as connection:
        updated = connection.execute(
            "UPDATE farmer_need_responses SET read_at = ? WHERE farmer_id = ? AND read_at IS NULL",
            (datetime.now(timezone.utc).isoformat(), farmer["id"]),
        )
    return {"marked_read": updated.rowcount}


@app.get("/api/admin/overview")
def admin_overview(request: Request) -> dict[str, object]:
    require_role(request, "admin")
    with database_connection() as connection:
        users = connection.execute("SELECT role, COUNT(*) FROM users GROUP BY role").fetchall()
        fields = connection.execute("SELECT COUNT(*) FROM farm_fields").fetchone()[0]
        scans = connection.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        observations = connection.execute("SELECT COUNT(*) FROM market_prices").fetchone()[0]
        active_listings = connection.execute(
            "SELECT COUNT(*) FROM marketplace_products WHERE is_active = 1 AND quantity > 0"
        ).fetchone()[0]
        open_orders = connection.execute(
            "SELECT COUNT(*) FROM marketplace_orders WHERE status IN ('placed', 'confirmed')"
        ).fetchone()[0]
    return {
        "users_by_role": {row[0]: row[1] for row in users},
        "farm_fields": fields,
        "crop_scans": scans,
        "market_observations": observations,
        "active_listings": active_listings,
        "open_orders": open_orders,
    }


@app.get("/api/admin/users")
def admin_users(request: Request) -> list[dict[str, object]]:
    require_role(request, "admin")
    with database_connection() as connection:
        rows = connection.execute(
            "SELECT id, name, email, role, created_at FROM users ORDER BY created_at DESC, id DESC"
        ).fetchall()
    return [
        {"id": row[0], "name": row[1], "email": row[2], "role": row[3], "created_at": row[4]}
        for row in rows
    ]


@app.patch("/api/admin/users/{user_id}/role")
def update_admin_user_role(
    user_id: int, payload: AdminUserRoleUpdate, request: Request
) -> dict[str, object]:
    admin = require_role(request, "admin")
    if user_id == admin["id"] and payload.role != "admin":
        raise HTTPException(status_code=400, detail="You cannot change your own administrator role.")
    with database_connection() as connection:
        user = connection.execute("SELECT id, role FROM users WHERE id = ?", (user_id,)).fetchone()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found.")
        if user[1] == "admin" and payload.role != "admin":
            admin_count = connection.execute(
                "SELECT COUNT(*) FROM users WHERE role = 'admin'"
            ).fetchone()[0]
            if admin_count <= 1:
                raise HTTPException(
                    status_code=409,
                    detail="The last administrator cannot be demoted.",
                )
        connection.execute("UPDATE users SET role = ? WHERE id = ?", (payload.role, user_id))
    return {"id": user_id, "role": payload.role}


@app.get("/api/admin/products")
def admin_products(request: Request) -> list[dict[str, object]]:
    require_role(request, "admin")
    with database_connection() as connection:
        rows = connection.execute(
            """
            SELECT p.id, p.name, p.category, p.price, p.unit, p.quantity, p.is_active,
                   COALESCE(u.name, p.seller_name), p.created_at
            FROM marketplace_products p
            LEFT JOIN users u ON u.id = p.seller_id
            ORDER BY p.created_at DESC, p.id DESC
            """
        ).fetchall()
    return [
        {
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "price": row[3],
            "unit": row[4],
            "quantity": row[5],
            "active": bool(row[6]),
            "seller": row[7],
            "created_at": row[8],
        }
        for row in rows
    ]


@app.patch("/api/admin/products/{product_id}")
def update_admin_product_status(
    product_id: int, payload: AdminProductStatusUpdate, request: Request
) -> dict[str, object]:
    require_role(request, "admin")
    with database_connection() as connection:
        updated = connection.execute(
            "UPDATE marketplace_products SET is_active = ? WHERE id = ?",
            (int(payload.active), product_id),
        )
        if updated.rowcount != 1:
            raise HTTPException(status_code=404, detail="Marketplace listing not found.")
    return {"id": product_id, "active": payload.active}


@app.get("/api/admin/orders")
def admin_orders(request: Request) -> list[dict[str, object]]:
    require_role(request, "admin")
    with database_connection() as connection:
        rows = connection.execute(
            """
            SELECT o.id, o.total, o.status, o.created_at, u.name, u.email
            FROM marketplace_orders o
            JOIN users u ON u.id = o.buyer_id
            ORDER BY o.created_at DESC, o.id DESC
            """
        ).fetchall()
        orders = []
        for row in rows:
            items = connection.execute(
                "SELECT product_name, quantity, unit_price FROM marketplace_order_items "
                "WHERE order_id = ? ORDER BY id",
                (row[0],),
            ).fetchall()
            orders.append(
                {
                    "id": row[0],
                    "total": row[1],
                    "status": row[2],
                    "created_at": row[3],
                    "buyer_name": row[4],
                    "buyer_email": row[5],
                    "items": [
                        {"name": item[0], "quantity": item[1], "unit_price": item[2]}
                        for item in items
                    ],
                }
            )
    return orders


@app.patch("/api/admin/orders/{order_id}")
def update_admin_order_status(
    order_id: int, payload: AdminOrderStatusUpdate, request: Request
) -> dict[str, object]:
    require_role(request, "admin")
    with database_connection() as connection:
        order = connection.execute(
            "SELECT status FROM marketplace_orders WHERE id = ?", (order_id,)
        ).fetchone()
        if order is None:
            raise HTTPException(status_code=404, detail="Order not found.")
        current_status = order[0]
        allowed_transitions = {
            "placed": {"confirmed", "cancelled"},
            "confirmed": {"completed", "cancelled"},
            "completed": set(),
            "cancelled": set(),
        }
        if payload.status not in allowed_transitions.get(current_status, set()):
            raise HTTPException(
                status_code=409,
                detail=f"Order cannot move from {current_status} to {payload.status}.",
            )
        if payload.status == "cancelled":
            items = connection.execute(
                "SELECT product_id, quantity FROM marketplace_order_items WHERE order_id = ?",
                (order_id,),
            ).fetchall()
            for product_id, quantity in items:
                if product_id is not None:
                    connection.execute(
                        "UPDATE marketplace_products SET quantity = quantity + ? WHERE id = ?",
                        (quantity, product_id),
                    )
        connection.execute(
            "UPDATE marketplace_orders SET status = ? WHERE id = ?",
            (payload.status, order_id),
        )
    return {"id": order_id, "status": payload.status}


@app.post("/api/auth/logout")
def logout(request: Request) -> Response:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        with database_connection() as connection:
            connection.execute(
                "DELETE FROM sessions WHERE token_hash = ?",
                (hashlib.sha256(token.encode()).hexdigest(),),
            )
    response = Response(content=json.dumps({"message": "Signed out"}), media_type="application/json")
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.post("/api/auth/forgot-password")
def forgot_password(request: ForgotPasswordRequest) -> dict[str, object]:
    reset_url = None
    otp_code = None
    with database_connection() as connection:
        row = connection.execute(
            "SELECT id, COALESCE(preferred_language, 'English') FROM users WHERE email = ?",
            (request.email,),
        ).fetchone()
        if row:
            token = secrets.token_urlsafe(48)
            expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)
            connection.execute(
                "INSERT OR REPLACE INTO password_resets (token_hash, user_id, expires_at, used) VALUES (?, ?, ?, 0)",
                (hashlib.sha256(token.encode()).hexdigest(), row[0], expires_at.isoformat()),
            )
            otp_code = f"{secrets.randbelow(900_000) + 100_000:06d}"
            connection.execute(
                "INSERT INTO password_otps (user_id, otp_hash, expires_at, used, created_at) VALUES (?, ?, ?, 0, ?)",
                (row[0], hashlib.sha256(otp_code.encode()).hexdigest(), expires_at.isoformat(), datetime.now(timezone.utc).isoformat()),
            )
            if os.getenv("APP_ENV", "development").lower() != "production":
                reset_url = f"/login.html?reset_token={token}"

            preferred_language = normalize_language(request.language or row[1])
            sent = send_password_reset_email(request.email, otp_code, preferred_language)
            if not sent:
                print(f"Development OTP for {request.email}: {otp_code}")

    result: dict[str, object] = {
        "message": "If an account exists, a reset code has been prepared and sent to the email address.",
        "reset_url": reset_url,
        "otp_code": otp_code,
        "expires_in_minutes": 30,
    }
    return result


@app.post("/api/auth/reset-password")
def reset_password(request: ResetPasswordRequest) -> dict[str, str]:
    now = datetime.now(timezone.utc).isoformat()
    if request.token:
        token_hash = hashlib.sha256(request.token.encode()).hexdigest()
        with database_connection() as connection:
            row = connection.execute(
                "SELECT user_id FROM password_resets WHERE token_hash = ? AND expires_at > ? AND used = 0",
                (token_hash, now),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=400, detail="This reset link is invalid or expired.")
            user_id = row[0]
            connection.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(request.password), user_id))
            connection.execute("UPDATE password_resets SET used = 1 WHERE token_hash = ?", (token_hash,))
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        return {"message": "Password updated. You can now sign in."}

    if request.email and request.otp:
        with database_connection() as connection:
            user = connection.execute(
                "SELECT id FROM users WHERE email = ?",
                (request.email.strip().lower(),),
            ).fetchone()
            if not user:
                raise HTTPException(status_code=400, detail="We could not find an account with that email.")
            row = connection.execute(
                "SELECT id, otp_hash, expires_at FROM password_otps WHERE user_id = ? AND used = 0 AND expires_at > ? ORDER BY created_at DESC LIMIT 1",
                (user[0], now),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=400, detail="This OTP is invalid or expired.")
            if hashlib.sha256(request.otp.strip().encode()).hexdigest() != row[1]:
                raise HTTPException(status_code=400, detail="This OTP is invalid or expired.")
            connection.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(request.password), user[0]))
            connection.execute("UPDATE password_otps SET used = 1 WHERE id = ?", (row[0],))
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (user[0],))
        return {"message": "Password updated. You can now sign in."}

    raise HTTPException(status_code=400, detail="Enter a valid reset token or OTP.")


@app.get("/api/market-prices")
def get_market_prices(
    request: Request,
    crop: Literal["cotton", "soybean", "wheat"] | None = None,
) -> list[dict[str, object]]:
    require_user(request)
    crops = [crop] if crop else ["cotton", "soybean", "wheat"]
    return [market_price_summary(crop_name) for crop_name in crops]


@app.get("/api/market-live")
def get_live_market_data(request: Request) -> list[dict[str, object]]:
    require_user(request)
    return live_market_alerts()


@app.get("/api/marketplace/products")
def list_marketplace_products(request: Request) -> list[dict[str, object]]:
    require_user(request)
    with database_connection() as connection:
        rows = connection.execute(
            """
            SELECT p.id, p.name, p.category, p.price, p.unit, p.quantity, p.emoji,
                   COALESCE(u.name, p.seller_name)
            FROM marketplace_products p
            LEFT JOIN users u ON u.id = p.seller_id
            WHERE p.quantity > 0 AND p.is_active = 1
            ORDER BY p.created_at DESC, p.id
            """
        ).fetchall()
    return [
        {
            "id": row[0],
            "name": row[1],
            "category": row[2],
            "price": row[3],
            "unit": row[4],
            "quantity": row[5],
            "emoji": row[6],
            "farmer": row[7],
        }
        for row in rows
    ]


@app.post("/api/marketplace/products")
def create_marketplace_product(
    request: Request, product: MarketplaceProductCreate
) -> dict[str, object]:
    user = require_role(request, "farmer", "admin")
    with database_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO marketplace_products
                (seller_id, seller_name, name, category, price, unit, quantity, emoji, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                user["name"],
                product.name.strip(),
                product.category,
                product.price,
                product.unit.strip(),
                product.quantity,
                product.emoji,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return {"id": cursor.lastrowid, **product.model_dump(), "farmer": user["name"]}


@app.post("/api/marketplace/orders")
def create_marketplace_order(
    request: Request, order: MarketplaceOrderCreate
) -> dict[str, object]:
    buyer = require_user(request)
    quantities: dict[int, int] = {}
    for item in order.items:
        if item.product_id in quantities:
            raise HTTPException(status_code=422, detail="Each product can appear only once per order.")
        quantities[item.product_id] = item.quantity

    now = datetime.now(timezone.utc).isoformat()
    with database_connection() as connection:
        order_items: list[tuple[int, str, int, float]] = []
        total = 0.0
        for product_id, quantity in quantities.items():
            row = connection.execute(
                "SELECT name, price, quantity FROM marketplace_products WHERE id = ? AND is_active = 1",
                (product_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="A marketplace product was not found.")
            if quantity > row[2]:
                raise HTTPException(status_code=409, detail=f"Not enough stock for {row[0]}.")
            updated = connection.execute(
                "UPDATE marketplace_products SET quantity = quantity - ? WHERE id = ? AND quantity >= ?",
                (quantity, product_id, quantity),
            )
            if updated.rowcount != 1:
                raise HTTPException(status_code=409, detail=f"Not enough stock for {row[0]}.")
            unit_price = float(row[1])
            total += unit_price * quantity
            order_items.append((product_id, str(row[0]), quantity, unit_price))

        cursor = connection.execute(
            "INSERT INTO marketplace_orders (buyer_id, total, status, created_at) VALUES (?, ?, 'placed', ?)",
            (buyer["id"], round(total, 2), now),
        )
        connection.executemany(
            """
            INSERT INTO marketplace_order_items (order_id, product_id, product_name, quantity, unit_price)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(cursor.lastrowid, *item) for item in order_items],
        )
    return {"id": cursor.lastrowid, "total": round(total, 2), "status": "placed"}


@app.get("/api/marketplace/orders")
def list_marketplace_orders(request: Request) -> list[dict[str, object]]:
    buyer = require_user(request)
    with database_connection() as connection:
        orders = connection.execute(
            "SELECT id, total, status, created_at FROM marketplace_orders "
            "WHERE buyer_id = ? ORDER BY created_at DESC, id DESC",
            (buyer["id"],),
        ).fetchall()
        results = []
        for order_id, total, status, created_at in orders:
            items = connection.execute(
                "SELECT product_name, quantity, unit_price FROM marketplace_order_items WHERE order_id = ?",
                (order_id,),
            ).fetchall()
            results.append(
                {
                    "id": order_id,
                    "total": total,
                    "status": status,
                    "created_at": created_at,
                    "items": [
                        {"name": item[0], "quantity": item[1], "unit_price": item[2]}
                        for item in items
                    ],
                }
            )
    return results


@app.post("/api/market-prices")
def add_market_price(request: Request, observation: PriceObservationCreate) -> dict[str, object]:
    require_role(request, "farmer", "admin")
    with database_connection() as connection:
        connection.execute(
            "INSERT INTO market_prices "
            "(crop, market, price_per_quintal, reported_on, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                observation.crop,
                observation.market,
                observation.price_per_quintal,
                observation.reported_on.isoformat(),
                observation.source,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return market_price_summary(observation.crop)


@app.delete("/api/market-prices/{observation_id}")
def delete_market_price(request: Request, observation_id: int) -> dict[str, bool]:
    require_role(request, "farmer", "admin")
    with database_connection() as connection:
        cursor = connection.execute("DELETE FROM market_prices WHERE id = ?", (observation_id,))
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Price observation not found.")
    return {"deleted": True}


@app.get("/api/farm/summary")
def farm_summary(request: Request) -> dict[str, list[dict[str, object]]]:
    user = require_role(request, "farmer", "admin")
    with database_connection() as connection:
        fields = connection.execute(
            """
            SELECT id, name, crop, season, area_acres, location, status, created_at
            FROM farm_fields
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user["id"],),
        ).fetchall()
        tasks = connection.execute(
            """
            SELECT id, field_id, title, due_date, priority, status, notes, created_at
            FROM farm_tasks
            WHERE user_id = ?
            ORDER BY due_date IS NULL, due_date, created_at DESC
            """,
            (user["id"],),
        ).fetchall()

    field_payload = [
        {
            "id": row[0],
            "name": row[1],
            "crop": row[2],
            "season": row[3],
            "area_acres": float(row[4]),
            "location": row[5],
            "status": row[6],
            "created_at": row[7],
        }
        for row in fields
    ]
    task_payload = [
        {
            "id": row[0],
            "field_id": row[1],
            "title": row[2],
            "due_date": row[3],
            "priority": row[4],
            "status": row[5],
            "notes": row[6],
            "created_at": row[7],
        }
        for row in tasks
    ]
    return {"fields": field_payload, "tasks": task_payload}


@app.get("/api/farm/fields")
def list_farm_fields(request: Request) -> list[dict[str, object]]:
    user = require_role(request, "farmer", "admin")
    with database_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, name, crop, season, area_acres, location, status, created_at
            FROM farm_fields
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user["id"],),
        ).fetchall()
    return [
        {
            "id": row[0],
            "name": row[1],
            "crop": row[2],
            "season": row[3],
            "area_acres": float(row[4]),
            "location": row[5],
            "status": row[6],
            "created_at": row[7],
        }
        for row in rows
    ]


@app.post("/api/farm/fields")
def create_farm_field(request: Request, payload: FarmFieldCreate) -> dict[str, object]:
    user = require_role(request, "farmer", "admin")
    with database_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO farm_fields (user_id, name, crop, season, area_acres, location, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                payload.name.strip(),
                payload.crop.strip(),
                payload.season,
                payload.area_acres,
                payload.location.strip(),
                payload.status.strip(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        field_id = cursor.lastrowid
    return {
        "id": field_id,
        "name": payload.name.strip(),
        "crop": payload.crop.strip(),
        "season": payload.season,
        "area_acres": payload.area_acres,
        "location": payload.location.strip(),
        "status": payload.status.strip(),
    }


@app.get("/api/farm/tasks")
def list_farm_tasks(request: Request) -> list[dict[str, object]]:
    user = require_role(request, "farmer", "admin")
    with database_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, field_id, title, due_date, priority, status, notes, created_at
            FROM farm_tasks
            WHERE user_id = ?
            ORDER BY due_date IS NULL, due_date, created_at DESC
            """,
            (user["id"],),
        ).fetchall()
    return [
        {
            "id": row[0],
            "field_id": row[1],
            "title": row[2],
            "due_date": row[3],
            "priority": row[4],
            "status": row[5],
            "notes": row[6],
            "created_at": row[7],
        }
        for row in rows
    ]


@app.get("/api/farm/reminders")
def farm_reminders(request: Request) -> list[dict[str, object]]:
    user = require_role(request, "farmer", "admin")
    today = date.today()
    reminder_window = today + timedelta(days=14)
    reminders: list[dict[str, object]] = []
    with database_connection() as connection:
        tasks = connection.execute(
            """
            SELECT id, title, due_date, priority
            FROM farm_tasks
            WHERE user_id = ? AND status != 'Done' AND due_date IS NOT NULL
                AND due_date <= ?
            ORDER BY due_date, id
            """,
            (user["id"], reminder_window.isoformat()),
        ).fetchall()
        crops = connection.execute(
            """
            SELECT id, crop_name, expected_harvest_date
            FROM farm_crops
            WHERE user_id = ? AND expected_harvest_date IS NOT NULL
                AND expected_harvest_date <= ?
            ORDER BY expected_harvest_date, id
            """,
            (user["id"], reminder_window.isoformat()),
        ).fetchall()

    for task_id, title, due_date, priority in tasks:
        days_until = (date.fromisoformat(due_date) - today).days
        reminders.append(
            {
                "id": f"task-{task_id}",
                "kind": "task",
                "record_id": task_id,
                "title": title,
                "due_date": due_date,
                "days_until": days_until,
                "priority": priority,
                "message": "Overdue" if days_until < 0 else "Due today" if days_until == 0 else f"Due in {days_until} days",
            }
        )
    for crop_id, crop_name, harvest_date in crops:
        days_until = (date.fromisoformat(harvest_date) - today).days
        reminders.append(
            {
                "id": f"harvest-{crop_id}",
                "kind": "harvest",
                "record_id": crop_id,
                "title": f"{crop_name} harvest",
                "due_date": harvest_date,
                "days_until": days_until,
                "priority": "High" if days_until <= 3 else "Medium",
                "message": "Harvest date passed" if days_until < 0 else "Harvest expected today" if days_until == 0 else f"Expected in {days_until} days",
            }
        )
    return sorted(reminders, key=lambda reminder: (reminder["due_date"], reminder["kind"]))


@app.patch("/api/farm/tasks/{task_id}")
def update_farm_task_status(
    task_id: int, payload: FarmTaskStatusUpdate, request: Request
) -> dict[str, object]:
    user = require_role(request, "farmer", "admin")
    with database_connection() as connection:
        updated = connection.execute(
            "UPDATE farm_tasks SET status = ? WHERE id = ? AND user_id = ?",
            (payload.status, task_id, user["id"]),
        )
        if updated.rowcount != 1:
            raise HTTPException(status_code=404, detail="Farm task not found.")
    return {"id": task_id, "status": payload.status}


@app.get("/api/farm/calendar")
def farm_calendar(request: Request) -> dict[str, object]:
    user = require_role(request, "farmer", "admin")
    month = datetime.now(timezone.utc).month
    if month in {6, 7, 8, 9}:
        season = "Kharif"
        reminder = "Monitor monsoon drainage and check standing water after heavy rain."
    elif month in {10, 11, 12, 1, 2}:
        season = "Rabi"
        reminder = "Keep irrigation intervals steady and watch for cool-weather disease pressure."
    else:
        season = "Zaid"
        reminder = "Schedule irrigation during heat stress and apply mulching to reduce evaporation."

    with database_connection() as connection:
        rows = connection.execute(
            "SELECT id, name, crop, area_acres, location, status, season "
            "FROM farm_fields WHERE user_id = ? ORDER BY created_at DESC",
            (user["id"],),
        ).fetchall()

    season_reminders = {
        "Kharif": "Monitor monsoon drainage and check standing water after heavy rain.",
        "Rabi": "Keep irrigation intervals steady and watch for cool-weather disease pressure.",
        "Zaid": "Schedule irrigation during heat stress and apply mulching to reduce evaporation.",
        "Perennial": "Monitor crop growth regularly and adjust irrigation and care to its current growth stage.",
    }
    recommendations = []
    for row in rows:
        field_season = row[6] if row[6] != "Other" else season
        field_reminder = season_reminders.get(field_season, reminder)
        recommendations.append(
            {
                "field_name": row[1],
                "crop": row[2],
                "season": field_season,
                "advice": f"{field_reminder} Current field status: {row[5]}.",
                "irrigation_hint": "Check soil moisture before dawn and schedule drip irrigation if the field feels dry.",
            }
        )

    if not recommendations:
        recommendations = [{
            "field_name": "No fields yet",
            "crop": "Add a field first",
            "season": season,
            "advice": "Create a field to receive seasonal guidance and irrigation reminders.",
            "irrigation_hint": "Use a simple moisture check before applying water.",
        }]

    return {"season": season, "month": month, "recommendations": recommendations}


@app.get("/api/farm/soil-status")
def soil_status(request: Request) -> list[dict[str, object]]:
    user = require_role(request, "farmer", "admin")
    weather = get_live_weather("Nanded") or {"temp": 30, "humidity": 60, "rainfall": 1, "riskLevel": "medium"}
    with database_connection() as connection:
        rows = connection.execute(
            "SELECT id, name, crop, area_acres, location, status FROM farm_fields WHERE user_id = ? ORDER BY created_at DESC",
            (user["id"],),
        ).fetchall()

    results: list[dict[str, object]] = []
    for row in rows:
        crop_factor = {"Cotton": 0.75, "Soybean": 0.8, "Wheat": 0.68, "Rice": 0.7}.get(row[2], 0.72)
        rainfall = float(weather.get("rainfall") or 0)
        temp = float(weather.get("temp") or 30)
        humidity = float(weather.get("humidity") or 60)
        moisture = max(15, min(95, round(68 - (temp - 25) * 1.2 + (rainfall * 4) - (humidity - 60) * 0.4 + crop_factor * 10, 1)))
        if humidity > 80 or rainfall > 2:
            alert_level = "high"
            recommendation = "Reduce irrigation and improve drainage to avoid waterlogging and fungal disease risk."
        elif moisture < 40:
            alert_level = "high"
            recommendation = "Apply targeted irrigation immediately and check for heat stress in the root zone."
        elif moisture < 55:
            alert_level = "medium"
            recommendation = "Increase watering frequency slightly and monitor root-zone dryness during the next cycle."
        else:
            alert_level = "low"
            recommendation = "Soil moisture is within a safe window. Continue regular field checks."

        results.append(
            {
                "field_id": row[0],
                "field_name": row[1],
                "crop": row[2],
                "location": row[4],
                "status": row[5],
                "moisture_percent": moisture,
                "alert_level": alert_level,
                "recommendation": recommendation,
                "weather_condition": weather.get("condition", "Moderate"),
            }
        )

    if not results:
        return [{
            "field_id": None,
            "field_name": "No fields yet",
            "crop": "Add a field",
            "location": "N/A",
            "status": "No field data",
            "moisture_percent": 0,
            "alert_level": "low",
            "recommendation": "Create a field to begin moisture monitoring.",
            "weather_condition": weather.get("condition", "Moderate"),
        }]

    return results


@app.post("/api/farm/tasks")
def create_farm_task(request: Request, payload: FarmTaskCreate) -> dict[str, object]:
    user = require_role(request, "farmer", "admin")
    if payload.field_id is not None:
        with database_connection() as connection:
            field = connection.execute(
                "SELECT id FROM farm_fields WHERE id = ? AND user_id = ?",
                (payload.field_id, user["id"]),
            ).fetchone()
        if field is None:
            raise HTTPException(status_code=404, detail="Field not found.")
    with database_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO farm_tasks (user_id, field_id, title, due_date, priority, status, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                payload.field_id,
                payload.title.strip(),
                payload.due_date.isoformat() if payload.due_date else None,
                payload.priority,
                payload.status,
                payload.notes.strip() if payload.notes else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        task_id = cursor.lastrowid
    return {
        "id": task_id,
        "field_id": payload.field_id,
        "title": payload.title.strip(),
        "due_date": payload.due_date.isoformat() if payload.due_date else None,
        "priority": payload.priority,
        "status": payload.status,
        "notes": payload.notes.strip() if payload.notes else None,
    }


@app.get("/api/farm/crops")
def list_farm_crops(request: Request) -> list[dict[str, object]]:
    user = require_role(request, "farmer", "admin")
    with database_connection() as connection:
        rows = connection.execute(
            "SELECT id, crop_name, season, variety, sowing_date, expected_harvest_date, quantity_kg, "
            "expenses, sales_income, health, fertilizer_notes, pesticide_notes "
            "FROM farm_crops WHERE user_id = ? ORDER BY expected_harvest_date, created_at DESC",
            (user["id"],),
        ).fetchall()
    return [
        {
            "id": row[0],
            "crop_name": row[1],
            "season": row[2],
            "variety": row[3],
            "sowing_date": row[4],
            "expected_harvest_date": row[5],
            "quantity_kg": row[6],
            "expenses": row[7],
            "sales_income": row[8],
            "health": row[9],
            "fertilizer_notes": row[10],
            "pesticide_notes": row[11],
        }
        for row in rows
    ]


@app.post("/api/farm/crops")
def create_farm_crop(request: Request, payload: FarmCropCreate) -> dict[str, object]:
    user = require_role(request, "farmer", "admin")
    crop = payload.model_dump()
    crop["sowing_date"] = payload.sowing_date.isoformat()
    crop["expected_harvest_date"] = (
        payload.expected_harvest_date.isoformat() if payload.expected_harvest_date else None
    )
    crop["crop_name"] = payload.crop_name.strip()
    crop["variety"] = payload.variety.strip()
    crop["fertilizer_notes"] = payload.fertilizer_notes.strip()
    crop["pesticide_notes"] = payload.pesticide_notes.strip()
    with database_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO farm_crops (user_id, crop_name, season, variety, sowing_date, expected_harvest_date, "
            "quantity_kg, expenses, sales_income, health, fertilizer_notes, pesticide_notes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user["id"],
                crop["crop_name"],
                crop["season"],
                crop["variety"],
                crop["sowing_date"],
                crop["expected_harvest_date"],
                crop["quantity_kg"],
                crop["expenses"],
                crop["sales_income"],
                crop["health"],
                crop["fertilizer_notes"],
                crop["pesticide_notes"],
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return {"id": cursor.lastrowid, **crop}


@app.post("/api/analyze")
async def analyze_crop(
    request: Request,
    image: UploadFile = File(...),
    crop_type: str = Form(default=""),
    farm_size: float = Form(default=1, gt=0, le=10000),
    language: str = Form(default="English"),
) -> dict[str, object]:
    require_user(request)
    get_ai_credentials()
    allowed_languages = {"English", "Hindi", "Marathi", "Telugu", "Tamil"}
    if language not in allowed_languages:
        raise HTTPException(status_code=422, detail="Choose a supported response language.")

    media_type = (image.content_type or "").lower()
    if media_type not in {*SUPPORTED_IMAGES, "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WEBP image.")
    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than the 10 MB limit.")
    if not image_signature_matches(media_type, image_bytes):
        raise HTTPException(status_code=415, detail="The uploaded file is not a supported image.")

    crop_note = (
        f"The farmer says this is a {crop_type[:80]} crop."
        if crop_type
        else "Identify the crop type automatically."
    )
    prompt = f"""You are an agricultural scientist. Analyze the crop image and return only valid JSON, with no markdown or extra text, using exactly these fields:
{{"disease":"Disease name or Healthy Crop","cropDetected":"Crop name","confidence":85,"severity":"High","urgency":"Recommended response timeframe","description":"Short explanation","symptoms":["symptom"],"medicines":["treatment with dosage guidance"],"prevention":["prevention step"],"spreadRisk":75,"yieldLoss":40,"soilImpact":30,"treatmentCostPerAcre":800,"lossWithoutTreatment":5000,"savingsWithTreatment":4200}}
{crop_note}
Farm size: {farm_size} acres. Respond in {language}. Severity must be High, Medium, Low, or Healthy. Risk percentages must be integers from 0 to 100. Costs are estimates in Indian rupees. If this is not a crop image, use disease="Image not recognized" and severity="Low". State uncertainty clearly and do not present a visual estimate as a laboratory-confirmed diagnosis."""

    raw_text = await ask_ai(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=1600,
    )
    try:
        diagnosis = Diagnosis.model_validate(parse_json_response(raw_text))
    except ValidationError as exc:
        raise HTTPException(status_code=502, detail="The AI service returned an incomplete report.") from exc

    result = diagnosis.model_dump()
    result["_farmSize"] = farm_size
    created_at = datetime.now(timezone.utc).isoformat()
    with database_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO scans (created_at, image, media_type, diagnosis) VALUES (?, ?, ?, ?)",
            (created_at, image_bytes, media_type, json.dumps(result)),
        )
        scan_id = cursor.lastrowid

    return {"id": scan_id, "result": result}


@app.get("/api/scans")
def list_scans(request: Request) -> list[dict[str, object]]:
    require_user(request)
    with database_connection() as connection:
        rows = connection.execute(
            "SELECT id, created_at, media_type, diagnosis FROM scans ORDER BY id DESC LIMIT 100"
        ).fetchall()
    return [scan_to_dict(row) for row in rows]


@app.get("/api/scans/{scan_id}/image")
def get_scan_image(request: Request, scan_id: int) -> Response:
    require_user(request)
    with database_connection() as connection:
        row = connection.execute(
            "SELECT image, media_type FROM scans WHERE id = ?", (scan_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Scan image not found.")
    return Response(content=row[0], media_type=row[1], headers={"Cache-Control": "private, max-age=3600"})


@app.delete("/api/scans/{scan_id}")
def delete_scan(request: Request, scan_id: int) -> dict[str, bool]:
    require_user(request)
    with database_connection() as connection:
        cursor = connection.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Scan not found.")
    return {"deleted": True}


@app.post("/api/chat")
async def chat(http_request: Request, request: ChatRequest) -> dict[str, str]:
    require_user(http_request)
    system = """You are KrishiBot, a friendly farming advisor for Indian farmers. Give concise, practical guidance about crops, diseases, integrated pest management, soil, irrigation, and farming seasons. Use simple language. Do not claim certainty when details are missing. Recommend a local agricultural expert before pesticide use when crop or dosage details are uncertain."""
    reply = await ask_ai(
        messages=[message.model_dump() for message in request.messages],
        max_tokens=600,
        system=system,
    )
    return {"reply": reply.strip()}


@app.post("/api/weather-risk")
async def weather_risk(http_request: Request, request: WeatherRequest) -> dict[str, object]:
    require_user(http_request)
    live_data = get_live_weather(request.city)
    if live_data is not None:
        return live_data

    prompt = f"""For {request.city}, India, provide an illustrative weather scenario and crop disease-risk estimate. This is not a live weather lookup. Return only JSON with keys city,temp,feels,humidity,wind,condition,icon,rainfall,uvIndex,riskLevel,riskTitle,riskDetails. Clearly state in riskDetails that this is an AI-generated estimate, not current verified weather."""
    raw_text = await ask_ai(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return parse_json_response(raw_text)


@app.get("/api/weather/forecast")
def weather_forecast(request: Request, city: str) -> dict[str, object]:
    require_user(request)
    city = city.strip()
    if not city or len(city) > 100:
        raise HTTPException(status_code=422, detail="Enter a village or town name up to 100 characters.")
    return get_seven_day_forecast(city)


@app.get("/")
def home(request: Request) -> Response:
    if not current_user(request):
        return RedirectResponse("/login.html")
    return render_dashboard()


@app.get("/login.html")
def login_page() -> FileResponse:
    return FileResponse(BASE_DIR / "login.html", headers={"Cache-Control": "no-store"})


@app.get("/agriculture_projects.html")
def agriculture_projects(request: Request) -> Response:
    if not current_user(request):
        return RedirectResponse("/login.html")
    return render_dashboard()


def render_dashboard() -> HTMLResponse:
    page = (BASE_DIR / "agriculture_projects.html").read_text(encoding="utf-8")
    partial_dir = BASE_DIR / "sections"
    section_names = (
        "admin-overview",
        "weather-forecast",
        "farm-planner",
        "schemes",
        "farm-tools",
        "project-overview",
        "market-prices",
        "irrigation",
        "marketplace",
        "disease-detection",
    )
    start = page.find("<!-- INCLUDE:admin-overview -->")
    end = page.find("<footer>", start)
    if start < 0 or end < 0:
        raise RuntimeError("Dashboard section boundaries are missing.")
    sections = "\n".join(
        (partial_dir / f"{section_name}.html").read_text(encoding="utf-8")
        for section_name in section_names
    )
    page = page[:start] + sections + "\n" + page[end:]
    return HTMLResponse(page)


@app.get("/crop-disease-detector.html")
def crop_disease_detector(request: Request) -> Response:
    if not current_user(request):
        return RedirectResponse("/login.html")
    return RedirectResponse("/#disease-detection")


@app.get("/detector-embed")
def detector_embed(request: Request) -> Response:
    if not current_user(request):
        return RedirectResponse("/login.html")
    return FileResponse(BASE_DIR / "crop-disease-detector.html")
