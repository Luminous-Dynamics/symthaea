# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
LLM Service for Mycelix Mail

Advanced AI features powered by large language models:
- Email drafting and composition assistance
- Conversation summarization
- Smart reply generation
- Natural language query understanding
- Priority inbox scoring
- Relationship insights
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, AsyncGenerator
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import httpx

# ============================================================================
# Configuration
# ============================================================================

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434")  # Ollama default
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

router = APIRouter(prefix="/llm", tags=["LLM"])

# ============================================================================
# Models
# ============================================================================

class DraftRequest(BaseModel):
    """Request for email draft generation."""
    context: str = Field(description="Context for the email (replying to, forwarding, etc.)")
    intent: str = Field(description="What the user wants to communicate")
    tone: str = Field(default="professional", description="Desired tone: professional, casual, formal, friendly")
    recipient_info: Optional[str] = Field(default=None, description="Information about the recipient")
    constraints: Optional[str] = Field(default=None, description="Any constraints or requirements")


class DraftResponse(BaseModel):
    """Generated email draft."""
    subject: str
    body: str
    suggestions: list[str]
    confidence: float


class SummarizeRequest(BaseModel):
    """Request for conversation summarization."""
    emails: list[dict]
    max_length: int = Field(default=200, description="Maximum summary length in words")
    focus: Optional[str] = Field(default=None, description="Specific aspect to focus on")


class SummarizeResponse(BaseModel):
    """Conversation summary."""
    summary: str
    key_points: list[str]
    action_items: list[str]
    participants: list[str]
    date_range: dict


class NLQueryRequest(BaseModel):
    """Natural language query for email search."""
    query: str
    context: Optional[dict] = None


class NLQueryResponse(BaseModel):
    """Parsed query with structured filters."""
    interpretation: str
    filters: dict
    confidence: float
    suggestions: list[str]


class PriorityRequest(BaseModel):
    """Request for priority scoring."""
    email: dict
    user_patterns: Optional[dict] = None
    calendar: Optional[list[dict]] = None


class PriorityResponse(BaseModel):
    """Priority score and reasoning."""
    score: float
    urgency: str
    importance: str
    reasons: list[str]
    suggested_action: str
    deadline: Optional[str] = None


class RelationshipRequest(BaseModel):
    """Request for relationship insights."""
    contact_email: str
    interaction_history: list[dict]


class RelationshipResponse(BaseModel):
    """Relationship insights."""
    summary: str
    communication_style: str
    response_time_pattern: str
    common_topics: list[str]
    sentiment_trend: str
    suggestions: list[str]


# ============================================================================
# LLM Client
# ============================================================================

class LLMClient:
    """Client for interacting with LLM API."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
        self.api_url = LLM_API_URL
        self.model = LLM_MODEL

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text using the LLM."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            # Ollama API format
            response = await self.client.post(
                f"{self.api_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")

        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"LLM service unavailable: {str(e)}")

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream text generation."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            async with self.client.stream(
                "POST",
                f"{self.api_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": temperature},
                },
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if content := data.get("message", {}).get("content"):
                            yield content

        except httpx.HTTPError as e:
            raise HTTPException(status_code=503, detail=f"LLM service unavailable: {str(e)}")


llm_client = LLMClient()

# ============================================================================
# System Prompts
# ============================================================================

SYSTEM_PROMPTS = {
    "draft": """You are an expert email writing assistant for Mycelix Mail.
Your task is to help users compose clear, effective emails.

Guidelines:
- Match the requested tone precisely
- Be concise but complete
- Include appropriate greetings and sign-offs
- Consider the recipient's perspective
- Suggest improvements when appropriate

Output format: JSON with "subject", "body", "suggestions" (list), and "confidence" (0-1).""",

    "summarize": """You are an email conversation summarizer for Mycelix Mail.
Your task is to provide concise, actionable summaries of email threads.

Guidelines:
- Extract key points and decisions
- Identify action items with owners
- Note important dates and deadlines
- Maintain objectivity
- Highlight any unresolved questions

Output format: JSON with "summary", "key_points" (list), "action_items" (list), "participants" (list).""",

    "nl_query": """You are a natural language query parser for email search.
Convert user queries into structured search filters.

Available filters:
- from: sender email/name
- to: recipient email/name
- subject: subject keywords
- body: body keywords
- date_from, date_to: date range
- has_attachment: boolean
- is_read: boolean
- is_starred: boolean
- label: label name
- trust_score_min: minimum trust score (0-1)

Output format: JSON with "interpretation", "filters" (dict), "confidence" (0-1), "suggestions" (list).""",

    "priority": """You are an email priority scoring system for Mycelix Mail.
Analyze emails to determine urgency and importance.

Consider:
- Sender relationship and trust score
- Keywords indicating urgency
- Deadlines mentioned
- Meeting requests
- Action required indicators
- Thread activity
- User's calendar conflicts

Output format: JSON with "score" (0-1), "urgency" (low/medium/high/critical),
"importance" (low/medium/high), "reasons" (list), "suggested_action", "deadline" (optional).""",

    "relationship": """You are a relationship insight analyzer for Mycelix Mail.
Analyze communication patterns to provide useful insights.

Consider:
- Communication frequency and patterns
- Response time trends
- Common topics discussed
- Sentiment over time
- Professional vs personal tone

Output format: JSON with "summary", "communication_style", "response_time_pattern",
"common_topics" (list), "sentiment_trend", "suggestions" (list).""",
}

# ============================================================================
# Endpoints
# ============================================================================

@router.post("/draft", response_model=DraftResponse)
async def generate_draft(request: DraftRequest):
    """Generate an email draft based on context and intent."""
    prompt = f"""Generate an email draft with the following requirements:

Context: {request.context}
Intent: {request.intent}
Tone: {request.tone}
{"Recipient Info: " + request.recipient_info if request.recipient_info else ""}
{"Constraints: " + request.constraints if request.constraints else ""}

Respond with valid JSON only."""

    response = await llm_client.generate(
        prompt,
        system_prompt=SYSTEM_PROMPTS["draft"],
        temperature=0.7,
    )

    try:
        result = json.loads(response)
        return DraftResponse(**result)
    except json.JSONDecodeError:
        # Fallback parsing
        return DraftResponse(
            subject="",
            body=response,
            suggestions=[],
            confidence=0.5,
        )


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_conversation(request: SummarizeRequest):
    """Summarize an email conversation thread."""
    emails_text = "\n\n".join([
        f"From: {e.get('from', 'Unknown')}\n"
        f"Date: {e.get('date', 'Unknown')}\n"
        f"Subject: {e.get('subject', '')}\n"
        f"Body: {e.get('body', '')[:500]}"
        for e in request.emails
    ])

    prompt = f"""Summarize this email conversation (max {request.max_length} words):

{emails_text}

{"Focus on: " + request.focus if request.focus else ""}

Respond with valid JSON only."""

    response = await llm_client.generate(
        prompt,
        system_prompt=SYSTEM_PROMPTS["summarize"],
        temperature=0.3,
    )

    try:
        result = json.loads(response)
        result["participants"] = list(set(e.get("from", "") for e in request.emails if e.get("from")))
        result["date_range"] = {
            "start": request.emails[0].get("date") if request.emails else None,
            "end": request.emails[-1].get("date") if request.emails else None,
        }
        return SummarizeResponse(**result)
    except json.JSONDecodeError:
        return SummarizeResponse(
            summary=response[:request.max_length * 6],  # Approximate word to char
            key_points=[],
            action_items=[],
            participants=[],
            date_range={},
        )


@router.post("/parse-query", response_model=NLQueryResponse)
async def parse_natural_language_query(request: NLQueryRequest):
    """Parse natural language into structured search filters."""
    prompt = f"""Parse this email search query into structured filters:

Query: "{request.query}"

{"Additional context: " + json.dumps(request.context) if request.context else ""}

Respond with valid JSON only."""

    response = await llm_client.generate(
        prompt,
        system_prompt=SYSTEM_PROMPTS["nl_query"],
        temperature=0.2,
    )

    try:
        result = json.loads(response)
        return NLQueryResponse(**result)
    except json.JSONDecodeError:
        return NLQueryResponse(
            interpretation=request.query,
            filters={"body": request.query},
            confidence=0.3,
            suggestions=["Try being more specific"],
        )


@router.post("/priority", response_model=PriorityResponse)
async def calculate_priority(request: PriorityRequest):
    """Calculate email priority score."""
    email = request.email
    prompt = f"""Analyze this email for priority scoring:

From: {email.get('from', 'Unknown')}
Subject: {email.get('subject', '')}
Preview: {email.get('preview', '')[:300]}
Trust Score: {email.get('trust_score', 'Unknown')}
Has Attachments: {email.get('has_attachments', False)}
Received: {email.get('received_at', 'Unknown')}

{"User patterns: " + json.dumps(request.user_patterns) if request.user_patterns else ""}
{"Calendar context: " + json.dumps(request.calendar[:3]) if request.calendar else ""}

Respond with valid JSON only."""

    response = await llm_client.generate(
        prompt,
        system_prompt=SYSTEM_PROMPTS["priority"],
        temperature=0.3,
    )

    try:
        result = json.loads(response)
        return PriorityResponse(**result)
    except json.JSONDecodeError:
        return PriorityResponse(
            score=0.5,
            urgency="medium",
            importance="medium",
            reasons=["Unable to analyze"],
            suggested_action="Review manually",
        )


@router.post("/relationship-insights", response_model=RelationshipResponse)
async def analyze_relationship(request: RelationshipRequest):
    """Analyze communication relationship with a contact."""
    history_text = "\n".join([
        f"[{h.get('date', '')}] {h.get('direction', 'unknown')}: {h.get('subject', '')}"
        for h in request.interaction_history[:20]
    ])

    prompt = f"""Analyze the communication relationship with this contact:

Contact: {request.contact_email}
Recent Interactions:
{history_text}

Respond with valid JSON only."""

    response = await llm_client.generate(
        prompt,
        system_prompt=SYSTEM_PROMPTS["relationship"],
        temperature=0.5,
    )

    try:
        result = json.loads(response)
        return RelationshipResponse(**result)
    except json.JSONDecodeError:
        return RelationshipResponse(
            summary="Unable to analyze relationship",
            communication_style="Unknown",
            response_time_pattern="Unknown",
            common_topics=[],
            sentiment_trend="neutral",
            suggestions=[],
        )


@router.post("/smart-compose")
async def smart_compose(
    current_text: str,
    context: Optional[str] = None,
):
    """Get real-time composition suggestions."""
    prompt = f"""Continue this email naturally (provide 2-3 word continuation):

{context if context else ""}

Current text: "{current_text}"

Provide only the continuation, no explanation."""

    response = await llm_client.generate(
        prompt,
        temperature=0.8,
        max_tokens=20,
    )

    return {"suggestion": response.strip()}


@router.post("/rewrite")
async def rewrite_text(
    text: str,
    style: str = "professional",
    instruction: Optional[str] = None,
):
    """Rewrite text in a different style."""
    prompt = f"""Rewrite this text in a {style} style:

"{text}"

{"Additional instruction: " + instruction if instruction else ""}

Provide only the rewritten text, no explanation."""

    response = await llm_client.generate(
        prompt,
        temperature=0.6,
    )

    return {"rewritten": response.strip()}


@router.get("/health")
async def health_check():
    """Check LLM service health."""
    try:
        # Quick test generation
        response = await llm_client.generate(
            "Say 'OK' if you're working.",
            max_tokens=10,
        )
        return {
            "status": "healthy",
            "model": LLM_MODEL,
            "api_url": LLM_API_URL,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
