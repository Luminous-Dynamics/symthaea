# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Mail ML Service

FastAPI service providing machine learning capabilities for:
- Spam detection
- Phishing detection
- Email categorization
- Semantic search embeddings
- Meeting extraction
- Smart replies
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer

# ============================================================================
# Configuration
# ============================================================================

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "3002"))
MODEL_PATH = os.getenv("MODEL_PATH", "/models")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Models
# ============================================================================

class EmailInput(BaseModel):
    """Input model for email analysis."""
    id: str
    from_email: str = Field(alias="from")
    to: list[str]
    subject: str
    body: str
    headers: dict = Field(default_factory=dict)
    attachments: list[dict] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class SpamResult(BaseModel):
    """Result of spam detection."""
    is_spam: bool
    confidence: float
    reasons: list[str]
    spam_score: float


class PhishingResult(BaseModel):
    """Result of phishing detection."""
    is_phishing: bool
    confidence: float
    risk_level: str  # low, medium, high, critical
    indicators: list[dict]
    recommendations: list[str]


class CategoryResult(BaseModel):
    """Result of email categorization."""
    primary_category: str
    categories: list[dict]  # {category, confidence}
    suggested_labels: list[str]


class EmbeddingResult(BaseModel):
    """Result of embedding generation."""
    embedding: list[float]
    model: str
    dimension: int


class MeetingResult(BaseModel):
    """Extracted meeting information."""
    has_meeting: bool
    title: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    location: Optional[str] = None
    attendees: list[str] = Field(default_factory=list)
    confidence: float


class SmartReplyResult(BaseModel):
    """Generated smart reply suggestions."""
    replies: list[dict]  # {text, tone, confidence}


class AnalysisResult(BaseModel):
    """Complete email analysis result."""
    email_id: str
    spam: SpamResult
    phishing: PhishingResult
    category: CategoryResult
    meeting: Optional[MeetingResult] = None
    timestamp: str


# ============================================================================
# ML Service Class
# ============================================================================

class MLService:
    """Machine learning service for email analysis."""

    def __init__(self):
        self.embedding_model: Optional[SentenceTransformer] = None
        self.redis_client: Optional[redis.Redis] = None

        # Spam detection patterns
        self.spam_keywords = [
            "buy now", "limited time", "act now", "winner", "congratulations",
            "free money", "cash prize", "urgent action", "click here",
            "unsubscribe", "no obligation", "risk free", "satisfaction guaranteed",
            "double your", "earn money", "extra income", "work from home",
        ]

        # Phishing indicators
        self.phishing_patterns = {
            "urgent_language": ["urgent", "immediately", "suspended", "verify now", "action required"],
            "credential_requests": ["password", "login", "verify your account", "confirm your identity"],
            "suspicious_urls": ["bit.ly", "tinyurl", "goo.gl"],
            "impersonation": ["paypal", "amazon", "netflix", "bank", "irs", "microsoft"],
        }

        # Category keywords
        self.category_keywords = {
            "work": ["meeting", "project", "deadline", "task", "report", "team"],
            "personal": ["family", "friend", "birthday", "weekend", "vacation"],
            "finance": ["invoice", "payment", "transaction", "receipt", "bank"],
            "social": ["invitation", "event", "party", "celebrate"],
            "newsletter": ["unsubscribe", "newsletter", "digest", "weekly"],
            "promotions": ["sale", "discount", "offer", "promo", "deal"],
        }

    async def initialize(self):
        """Initialize ML models and connections."""
        logger.info("Initializing ML service...")

        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")

        # Connect to Redis
        try:
            self.redis_client = await redis.from_url(REDIS_URL)
            await self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None

        logger.info("ML service initialized")

    async def shutdown(self):
        """Cleanup resources."""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("ML service shutdown complete")

    def detect_spam(self, email: EmailInput) -> SpamResult:
        """Detect if email is spam."""
        reasons = []
        score = 0.0

        text = f"{email.subject} {email.body}".lower()

        # Check for spam keywords
        for keyword in self.spam_keywords:
            if keyword in text:
                score += 0.1
                reasons.append(f"Contains spam keyword: '{keyword}'")

        # Check for excessive caps
        caps_ratio = sum(1 for c in email.subject if c.isupper()) / max(len(email.subject), 1)
        if caps_ratio > 0.5:
            score += 0.2
            reasons.append("Excessive capitalization in subject")

        # Check for suspicious sender patterns
        if email.from_email.count('@') > 1:
            score += 0.3
            reasons.append("Multiple @ symbols in sender address")

        # Check for too many recipients
        if len(email.to) > 20:
            score += 0.2
            reasons.append("Large number of recipients")

        # Check for no-reply sender
        if "noreply" in email.from_email.lower() or "no-reply" in email.from_email.lower():
            score += 0.05
            reasons.append("Sent from no-reply address")

        # Check headers for spam indicators
        if email.headers.get("X-Spam-Flag") == "YES":
            score += 0.5
            reasons.append("Flagged by email server")

        # Normalize score
        score = min(score, 1.0)
        is_spam = score > 0.5

        return SpamResult(
            is_spam=is_spam,
            confidence=abs(score - 0.5) * 2,  # Distance from threshold
            reasons=reasons[:5],  # Top 5 reasons
            spam_score=score,
        )

    def detect_phishing(self, email: EmailInput) -> PhishingResult:
        """Detect phishing attempts."""
        indicators = []
        risk_score = 0.0

        text = f"{email.subject} {email.body}".lower()

        # Check for urgent language
        for pattern in self.phishing_patterns["urgent_language"]:
            if pattern in text:
                risk_score += 0.15
                indicators.append({
                    "type": "urgent_language",
                    "detail": f"Contains urgent language: '{pattern}'",
                    "severity": "medium",
                })

        # Check for credential requests
        for pattern in self.phishing_patterns["credential_requests"]:
            if pattern in text:
                risk_score += 0.25
                indicators.append({
                    "type": "credential_request",
                    "detail": f"Requests credentials: '{pattern}'",
                    "severity": "high",
                })

        # Check for suspicious URLs
        for pattern in self.phishing_patterns["suspicious_urls"]:
            if pattern in text:
                risk_score += 0.2
                indicators.append({
                    "type": "suspicious_url",
                    "detail": f"Contains shortened URL service: '{pattern}'",
                    "severity": "medium",
                })

        # Check for brand impersonation
        sender_domain = email.from_email.split("@")[-1].lower() if "@" in email.from_email else ""
        for brand in self.phishing_patterns["impersonation"]:
            if brand in text and brand not in sender_domain:
                risk_score += 0.3
                indicators.append({
                    "type": "brand_impersonation",
                    "detail": f"Mentions '{brand}' but sent from different domain",
                    "severity": "high",
                })

        # Check for mismatched reply-to
        reply_to = email.headers.get("Reply-To", "")
        if reply_to and reply_to != email.from_email:
            risk_score += 0.2
            indicators.append({
                "type": "mismatched_reply",
                "detail": "Reply-To differs from sender",
                "severity": "medium",
            })

        # Determine risk level
        risk_score = min(risk_score, 1.0)
        if risk_score < 0.2:
            risk_level = "low"
        elif risk_score < 0.5:
            risk_level = "medium"
        elif risk_score < 0.8:
            risk_level = "high"
        else:
            risk_level = "critical"

        # Generate recommendations
        recommendations = []
        if risk_level in ["high", "critical"]:
            recommendations.append("Do not click any links in this email")
            recommendations.append("Do not provide any personal information")
            recommendations.append("Verify the sender through official channels")
        elif risk_level == "medium":
            recommendations.append("Exercise caution with this email")
            recommendations.append("Verify links before clicking")

        return PhishingResult(
            is_phishing=risk_level in ["high", "critical"],
            confidence=abs(risk_score - 0.5) * 2,
            risk_level=risk_level,
            indicators=indicators,
            recommendations=recommendations,
        )

    def categorize_email(self, email: EmailInput) -> CategoryResult:
        """Categorize email into categories."""
        text = f"{email.subject} {email.body}".lower()

        # Calculate scores for each category
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score / len(keywords)

        # Sort by score
        sorted_categories = sorted(
            category_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Default category
        if not sorted_categories:
            primary_category = "inbox"
            categories = [{"category": "inbox", "confidence": 0.5}]
        else:
            primary_category = sorted_categories[0][0]
            categories = [
                {"category": cat, "confidence": round(score, 3)}
                for cat, score in sorted_categories[:3]
            ]

        # Suggest labels based on content
        suggested_labels = []
        if "urgent" in text or "asap" in text:
            suggested_labels.append("urgent")
        if "fyi" in text or "for your information" in text:
            suggested_labels.append("fyi")
        if "action" in text or "todo" in text:
            suggested_labels.append("action-required")
        if any(att.get("filename", "").endswith((".pdf", ".doc", ".docx")) for att in email.attachments):
            suggested_labels.append("has-documents")

        return CategoryResult(
            primary_category=primary_category,
            categories=categories,
            suggested_labels=suggested_labels[:5],
        )

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """Generate semantic embedding for text."""
        if not self.embedding_model:
            raise HTTPException(status_code=503, detail="Embedding model not loaded")

        # Check cache
        cache_key = f"embedding:{hash(text)}"
        if self.redis_client:
            cached = await self.redis_client.get(cache_key)
            if cached:
                embedding = np.frombuffer(cached, dtype=np.float32).tolist()
                return EmbeddingResult(
                    embedding=embedding,
                    model=EMBEDDING_MODEL,
                    dimension=len(embedding),
                )

        # Generate embedding
        embedding = self.embedding_model.encode(text).tolist()

        # Cache result
        if self.redis_client:
            await self.redis_client.setex(
                cache_key,
                3600,  # 1 hour TTL
                np.array(embedding, dtype=np.float32).tobytes()
            )

        return EmbeddingResult(
            embedding=embedding,
            model=EMBEDDING_MODEL,
            dimension=len(embedding),
        )

    def extract_meeting(self, email: EmailInput) -> MeetingResult:
        """Extract meeting information from email."""
        import re

        text = f"{email.subject} {email.body}"
        text_lower = text.lower()

        # Check if this looks like a meeting email
        meeting_indicators = ["meeting", "schedule", "calendar", "invite", "appointment", "call"]
        has_meeting = any(indicator in text_lower for indicator in meeting_indicators)

        if not has_meeting:
            return MeetingResult(has_meeting=False, confidence=0.9)

        # Extract title (usually in subject)
        title = email.subject

        # Extract date patterns
        date_patterns = [
            r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4}\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(tomorrow|today|next week)\b',
        ]

        date = None
        for pattern in date_patterns:
            match = re.search(pattern, text_lower)
            if match:
                date = match.group(1)
                break

        # Extract time patterns
        time_patterns = [
            r'\b(\d{1,2}:\d{2}\s*(?:am|pm)?)\b',
            r'\b(\d{1,2}\s*(?:am|pm))\b',
        ]

        time = None
        for pattern in time_patterns:
            match = re.search(pattern, text_lower)
            if match:
                time = match.group(1)
                break

        # Extract location
        location_patterns = [
            r'location[:\s]+([^\n]+)',
            r'where[:\s]+([^\n]+)',
            r'room[:\s]+([^\n]+)',
            r'at\s+(\w+\s+\w+\s+\w+)',
        ]

        location = None
        for pattern in location_patterns:
            match = re.search(pattern, text_lower)
            if match:
                location = match.group(1).strip()[:100]
                break

        # Extract attendees (from To field and body mentions)
        attendees = list(email.to)

        return MeetingResult(
            has_meeting=True,
            title=title,
            date=date,
            time=time,
            location=location,
            attendees=attendees[:10],
            confidence=0.7 if date or time else 0.5,
        )

    def generate_smart_replies(self, email: EmailInput) -> SmartReplyResult:
        """Generate smart reply suggestions."""
        text_lower = email.body.lower()
        replies = []

        # Question detection
        if "?" in email.body:
            replies.append({
                "text": "Thanks for reaching out! Let me look into this and get back to you.",
                "tone": "professional",
                "confidence": 0.8,
            })
            replies.append({
                "text": "Good question! I'll check and follow up shortly.",
                "tone": "friendly",
                "confidence": 0.7,
            })

        # Meeting request
        if any(word in text_lower for word in ["meet", "schedule", "calendar"]):
            replies.append({
                "text": "That time works for me. I'll send a calendar invite.",
                "tone": "professional",
                "confidence": 0.8,
            })
            replies.append({
                "text": "Let me check my calendar and get back to you with some available times.",
                "tone": "professional",
                "confidence": 0.7,
            })

        # Thank you / appreciation
        if any(word in text_lower for word in ["thank", "appreciate", "grateful"]):
            replies.append({
                "text": "You're welcome! Happy to help.",
                "tone": "friendly",
                "confidence": 0.9,
            })

        # Request / action needed
        if any(word in text_lower for word in ["please", "could you", "can you", "would you"]):
            replies.append({
                "text": "Sure, I'll take care of this.",
                "tone": "professional",
                "confidence": 0.8,
            })
            replies.append({
                "text": "Got it, I'll work on this and update you.",
                "tone": "friendly",
                "confidence": 0.7,
            })

        # Default replies if none matched
        if not replies:
            replies = [
                {"text": "Thanks for the update!", "tone": "friendly", "confidence": 0.6},
                {"text": "Noted, thank you.", "tone": "professional", "confidence": 0.6},
                {"text": "Got it, thanks!", "tone": "casual", "confidence": 0.5},
            ]

        return SmartReplyResult(replies=replies[:3])

    async def analyze_email(self, email: EmailInput) -> AnalysisResult:
        """Perform complete email analysis."""
        spam = self.detect_spam(email)
        phishing = self.detect_phishing(email)
        category = self.categorize_email(email)
        meeting = self.extract_meeting(email)

        return AnalysisResult(
            email_id=email.id,
            spam=spam,
            phishing=phishing,
            category=category,
            meeting=meeting if meeting.has_meeting else None,
            timestamp=datetime.utcnow().isoformat(),
        )


# ============================================================================
# FastAPI Application
# ============================================================================

ml_service = MLService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    await ml_service.initialize()
    yield
    await ml_service.shutdown()


app = FastAPI(
    title="Mycelix Mail ML Service",
    description="Machine learning service for email analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "embedding_model": ml_service.embedding_model is not None,
        "redis_connected": ml_service.redis_client is not None,
    }


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_email(email: EmailInput):
    """Perform complete email analysis."""
    return await ml_service.analyze_email(email)


@app.post("/spam", response_model=SpamResult)
async def detect_spam(email: EmailInput):
    """Detect if email is spam."""
    return ml_service.detect_spam(email)


@app.post("/phishing", response_model=PhishingResult)
async def detect_phishing(email: EmailInput):
    """Detect phishing attempts."""
    return ml_service.detect_phishing(email)


@app.post("/categorize", response_model=CategoryResult)
async def categorize_email(email: EmailInput):
    """Categorize email."""
    return ml_service.categorize_email(email)


@app.post("/embedding", response_model=EmbeddingResult)
async def generate_embedding(body: dict):
    """Generate semantic embedding for text."""
    text = body.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    return await ml_service.generate_embedding(text)


@app.post("/meeting", response_model=MeetingResult)
async def extract_meeting(email: EmailInput):
    """Extract meeting information from email."""
    return ml_service.extract_meeting(email)


@app.post("/smart-replies", response_model=SmartReplyResult)
async def generate_smart_replies(email: EmailInput):
    """Generate smart reply suggestions."""
    return ml_service.generate_smart_replies(email)


@app.post("/batch-analyze")
async def batch_analyze(emails: list[EmailInput], background_tasks: BackgroundTasks):
    """Analyze multiple emails in batch."""
    results = []
    for email in emails[:100]:  # Limit to 100 emails
        result = await ml_service.analyze_email(email)
        results.append(result)
    return {"results": results, "count": len(results)}


@app.post("/similarity")
async def compute_similarity(body: dict):
    """Compute similarity between two texts."""
    text1 = body.get("text1", "")
    text2 = body.get("text2", "")

    if not text1 or not text2:
        raise HTTPException(status_code=400, detail="Both text1 and text2 are required")

    emb1 = await ml_service.generate_embedding(text1)
    emb2 = await ml_service.generate_embedding(text2)

    # Cosine similarity
    vec1 = np.array(emb1.embedding)
    vec2 = np.array(emb2.embedding)
    similarity = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    return {"similarity": similarity}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
