from __future__ import annotations

import json
import os

from src.config import ANTHROPIC_API_KEY

RULE_BASED_TEMPLATES = {
    "Critical": {
        "strategy_title": "Immediate retention intervention",
        "immediate_actions": [
            "Call within 24 hours",
            "Offer a personalized retention incentive",
            "Assign a relationship manager"
        ],
        "customer_message": "We value your business and want to help you get more from your account.",
    },
    "High": {
        "strategy_title": "Proactive retention campaign",
        "immediate_actions": [
            "Send a personalized email",
            "Offer a rewards enhancement",
            "Review recent product usage"
        ],
        "customer_message": "We noticed a few changes in your activity and have a plan to help you stay engaged.",
    },
    "Medium": {
        "strategy_title": "Engagement reactivation",
        "immediate_actions": [
            "Send a helpful usage recap",
            "Recommend one relevant product upgrade",
            "Invite a quick check-in"
        ],
        "customer_message": "We have a few ideas to help you get more value from your account.",
    },
    "Low": {
        "strategy_title": "Loyalty reinforcement",
        "immediate_actions": [
            "Send a thank-you message",
            "Offer a small benefit",
            "Keep engagement programs active"
        ],
        "customer_message": "Thank you for staying with us. We are here to keep your experience strong.",
    },
}


def generate_retention_strategy(customer: dict, prediction: dict) -> dict:
    risk_level = prediction.get("risk_level", "Medium")
    template = RULE_BASED_TEMPLATES.get(risk_level, RULE_BASED_TEMPLATES["Medium"])

    risk_reason = (
        f"The customer has {customer.get('months_inactive', 0)} inactive months, "
        f"{customer.get('contacts_count', 0)} service contacts, and "
        f"{customer.get('late_payments', 0)} late payments."
    )

    strategy = {
        "risk_explanation": risk_reason,
        "strategy_title": template["strategy_title"],
        "recommended_action": template["immediate_actions"][0],
        "offer": template["immediate_actions"][1],
        "communication_strategy": "Email + phone" if customer.get("digital_logins", 0) >= 5 else "Phone + direct outreach",
        "priority": risk_level,
        "customer_message": template["customer_message"],
        "source": "rule-based" if not ANTHROPIC_API_KEY else "optional-claude",
    }

    if ANTHROPIC_API_KEY:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            prompt = (
                f"Given this customer profile and churn probability {prediction.get('churn_probability', 0.0)}, "
                f"provide a concise retention plan in JSON with keys: risk_explanation, strategy_title, "
                f"recommended_action, offer, communication_strategy, priority, customer_message. "
                f"Customer: {customer}"
            )
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            ai_text = response.content[0].text
            strategy = json.loads(ai_text)
            strategy["source"] = "anthropic-claude"
        except Exception:
            pass

    return strategy
