"""
AI Recommendation Service using Groq API.
Sends a structured prompt with profiling statistics and receives
plain-English data quality recommendations.
"""
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_ai_recommendations(profile_data: dict, outlier_data: dict, score_data: dict) -> str:
    """
    Generate AI-powered recommendations based on the analysis results.

    Args:
        profile_data: Output from profiler.profile_dataset()
        outlier_data: Output from outliers.detect_outliers()
        score_data: Output from scorer.calculate_quality_score()

    Returns:
        Markdown-formatted string of recommendations.
    """
    if not GROQ_API_KEY:
        return "⚠️ Groq API key not configured. AI recommendations are unavailable."

    prompt = _build_prompt(profile_data, outlier_data, score_data)

    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Data Quality Analyst. You analyze dataset profiling "
                        "results and provide actionable, prioritized recommendations to improve "
                        "data quality. Format your response in clear markdown with numbered "
                        "recommendations. Be specific about which columns need attention and "
                        "what actions to take. Keep recommendations concise and practical."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1500,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI recommendation generation failed: {str(e)}"


def _build_prompt(profile_data: dict, outlier_data: dict, score_data: dict) -> str:
    """Construct a concise prompt summarizing the analysis findings."""
    # Build a compact summary to stay within token limits
    missing_info = profile_data.get("missing", {})
    dup_info = profile_data.get("duplicates", {})

    # Top columns with most missing values
    missing_cols = missing_info.get("per_column_pct", {})
    top_missing = sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:10]

    # Outlier summary
    outlier_cols = outlier_data.get("per_column", {})
    top_outlier = sorted(
        [(col, info["combined_count"]) for col, info in outlier_cols.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    prompt = f"""Analyze the following dataset quality report and provide prioritized recommendations:

**Dataset Overview:**
- Rows: {profile_data.get('row_count', 'N/A')}
- Columns: {profile_data.get('col_count', 'N/A')}
- Quality Score: {score_data.get('quality_score', 'N/A')}/100 ({score_data.get('risk_level', 'N/A')} Risk)

**Score Breakdown:**
{json.dumps(score_data.get('breakdown', {}), indent=2)}

**Missing Values:**
- Total missing: {missing_info.get('total_pct', 0)}%
- Top columns with missing data: {json.dumps(top_missing)}

**Duplicates:**
- Duplicate rows: {dup_info.get('count', 0)} ({dup_info.get('pct', 0)}%)

**Outliers:**
- Total rows with outliers: {outlier_data.get('total_outlier_pct', 0)}%
- Top columns with outliers: {json.dumps(top_outlier)}

**Type Issues:**
- Total type issue percentage: {profile_data.get('type_issue_pct', 0)}%

Provide:
1. Top 5 prioritized recommendations with specific actions
2. Which columns need immediate attention
3. Suggested data cleaning steps
4. Risk assessment summary
"""
    return prompt
