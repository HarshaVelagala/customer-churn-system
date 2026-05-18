"""
retention_ai.py
GenAI-powered retention strategy generator.
Uses Anthropic Claude API (claude-sonnet-4-20250514).
Falls back to rule-based strategies when API key is absent.
"""
import os, json, textwrap
from typing import Optional

# ── Try Anthropic client ───────────────────────────────────────────
try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Rule-based fallback strategies ────────────────────────────────
RULE_BASED_TEMPLATES = {
    "Critical": {
        "strategy": "Immediate High-Touch Intervention",
        "actions": [
            "🚨 Assign dedicated relationship manager within 24 hours",
            "💳 Offer 3% unlimited cashback upgrade for 6 months",
            "📞 Personal outreach call from senior customer success team",
            "🎁 Complimentary annual fee waiver + $200 statement credit",
            "⚡ Priority fraud protection & 24/7 concierge service",
        ],
        "message": "Your loyalty means everything to us. We'd love to discuss an exclusive offer tailored just for you."
    },
    "High": {
        "strategy": "Proactive Retention Offer",
        "actions": [
            "📊 Personalized spending insights & financial health report",
            "🏆 Double rewards points for next 90 days",
            "💰 $100 bonus statement credit on next $500 spend",
            "📱 Premium app features unlocked (travel lounge access)",
            "🔔 Proactive fraud alert upgrade",
        ],
        "message": "We noticed you haven't been taking full advantage of your card benefits. Here's what you're missing."
    },
    "Medium": {
        "strategy": "Engagement Re-activation Campaign",
        "actions": [
            "🎯 Targeted category bonus (dining/travel/groceries 5% back)",
            "📧 Personalized monthly spend optimization email",
            "🎮 Gamified challenges with milestone rewards",
            "🛡️ Free credit score monitoring enrollment",
        ],
        "message": "Maximize your card's potential with these exclusive benefits, curated for your spending style."
    },
    "Low": {
        "strategy": "Loyalty Reinforcement",
        "actions": [
            "⭐ Loyalty anniversary reward: bonus points on cardmember date",
            "📰 Monthly exclusive member newsletter & deals",
            "🔄 Automatic rewards category optimization",
        ],
        "message": "Thank you for being a valued cardmember. Here's a small token of our appreciation."
    }
}


def _build_prompt(customer: dict, prediction: dict) -> str:
    prob_pct = round(prediction.get("churn_probability", 0) * 100, 1)
    return textwrap.dedent(f"""
        You are a senior customer retention strategist at a major US credit card company.
        Analyze this customer profile and generate a personalized retention strategy.

        CUSTOMER PROFILE:
        - Age: {customer.get('age')} | Gender: {customer.get('gender')}
        - Card Type: {customer.get('card_type')} | Months on Book: {customer.get('months_on_book')}
        - Income Bracket: {customer.get('income_bracket')} | Education: {customer.get('education_level')}
        - Credit Limit: ${customer.get('credit_limit'):,.0f} | Utilization: {customer.get('utilization_ratio', 0)*100:.1f}%
        - Total Annual Spend: ${customer.get('total_trans_amount'):,.0f}
        - Transactions (last year): {customer.get('total_trans_count')}
        - Months Inactive: {customer.get('months_inactive_12m')} | Support Contacts: {customer.get('contacts_count_12m')}
        - Late Payments: {customer.get('late_payments')} | Rewards Redemptions: {customer.get('rewards_redemptions')}
        - Digital Logins (30d): {customer.get('digital_logins_30d')}
        - Spend Change Q4→Q1: {customer.get('total_amt_change_q4_q1')} | Tx Count Change: {customer.get('total_ct_change_q4_q1')}

        AI PREDICTION:
        - Churn Probability: {prob_pct}%
        - Risk Level: {prediction.get('risk_level')}
        - Segment: {prediction.get('segment')}

        Generate a JSON response with these exact keys:
        {{
          "churn_reason": "2-sentence explanation of why this customer is at risk",
          "strategy_title": "Short retention strategy name",
          "immediate_actions": ["action1", "action2", "action3"],
          "offer_recommendation": "Specific personalized offer",
          "communication_channel": "Best channel to reach this customer",
          "expected_retention_lift": "Expected improvement percentage",
          "customer_message": "Personalized 2-sentence message to send the customer"
        }}
        
        Return ONLY valid JSON, no markdown.
    """).strip()


def generate_retention_strategy(customer: dict, prediction: dict) -> dict:
    """
    Generate AI-powered retention strategy.
    Uses Claude API if available; otherwise falls back to rule-based.
    """
    risk = prediction.get("risk_level", "Medium")

    # ── Attempt Claude API ─────────────────────────────────────────
    if _ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY:
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                messages=[{"role": "user", "content": _build_prompt(customer, prediction)}]
            )
            raw = msg.content[0].text.strip()
            strategy = json.loads(raw)
            strategy["source"] = "Claude AI"
            return strategy
        except Exception as e:
            print(f"⚠️  API error: {e}. Using rule-based fallback.")

    # ── Rule-based fallback ────────────────────────────────────────
    template = RULE_BASED_TEMPLATES.get(risk, RULE_BASED_TEMPLATES["Medium"])
    return {
        "churn_reason": (
            f"Customer shows {risk.lower()} churn risk with "
            f"{prediction.get('churn_probability', 0)*100:.1f}% probability. "
            f"Key signals: {customer.get('months_inactive_12m', 0)} inactive months, "
            f"{customer.get('contacts_count_12m', 0)} support contacts, "
            f"{customer.get('late_payments', 0)} late payments."
        ),
        "strategy_title": template["strategy"],
        "immediate_actions": template["actions"],
        "offer_recommendation": template["actions"][1] if len(template["actions"]) > 1 else template["actions"][0],
        "communication_channel": (
            "Mobile Push + Email" if customer.get("digital_logins_30d", 0) >= 5
            else "Phone + Direct Mail"
        ),
        "expected_retention_lift": (
            "65-75%" if risk == "Critical" else
            "45-55%" if risk == "High" else
            "30-40%" if risk == "Medium" else "15-25%"
        ),
        "customer_message": template["message"],
        "source": "Rule-Based Engine"
    }


if __name__ == "__main__":
    # Demo
    sample_customer = {
        "age": 38, "gender": "Female", "card_type": "Gold",
        "months_on_book": 48, "income_bracket": "$60K-$80K",
        "education_level": "Graduate", "credit_limit": 12000,
        "utilization_ratio": 0.08, "total_trans_amount": 2200,
        "total_trans_count": 28, "months_inactive_12m": 4,
        "contacts_count_12m": 3, "late_payments": 2,
        "rewards_redemptions": 1, "digital_logins_30d": 4,
        "total_amt_change_q4_q1": 0.5, "total_ct_change_q4_q1": 0.45
    }
    sample_pred = {
        "churn_probability": 0.72, "risk_level": "Critical",
        "segment": "At-Risk"
    }
    result = generate_retention_strategy(sample_customer, sample_pred)
    print(json.dumps(result, indent=2))
