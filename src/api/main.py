"""
FastAPI server — POST /review returns structured bug findings.
Run with: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import ReviewRequest, ReviewResponse, Finding
from src.model.predict import review_code

app = FastAPI(
    title="AI Code Reviewer",
    description="Detects bugs in Python and JavaScript code using fine-tuned CodeBERT.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def compute_overall_risk(findings):
    severities = [f["severity"] for f in findings]
    if "high" in severities:
        return "high"
    elif "medium" in severities:
        return "medium"
    elif any(s == "none" for s in severities):
        return "clean"
    return "low"


def make_summary(findings, language):
    bugs = [f for f in findings if f["bug_type"] != "no_bug"]
    if not bugs:
        return f"✅ No bugs detected across {len(findings)} function(s) in {language} code."
    types = list({f["bug_type"] for f in bugs})
    return (f"⚠️ Found {len(bugs)} potential bug(s) in {len(findings)} function(s). "
            f"Issues: {', '.join(types)}. Review high-severity findings first.")


@app.get("/")
def root():
    return {"message": "AI Code Reviewer API is running. POST to /review"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/review", response_model=ReviewResponse)
def review(request: ReviewRequest):
    try:
        findings = review_code(request.code, request.language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    bugs_found = sum(1 for f in findings if f["bug_type"] != "no_bug")

    return ReviewResponse(
        language=request.language,
        total_functions=len(findings),
        bugs_found=bugs_found,
        overall_risk=compute_overall_risk(findings),
        findings=[Finding(**f) for f in findings],
        summary=make_summary(findings, request.language),
    )