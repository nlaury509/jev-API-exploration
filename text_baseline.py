"""Fixed, interpretable sentence/keyword rules; no trained model or API calls.

Freeze this file's hash before evaluating new documents. These rules produce
labels, not calibrated probabilities. They never emit the unclear class.
"""
import re


def classify(text: str) -> dict:
    sentences = re.split(r'(?<=[.!?;])\s+', ' '.join(text.split()).lower())
    evidence = {q: [] for q in ('debt_refinancing', 'maturity_extension',
                                'covenant_waiver', 'capital_expenditure_funding')}
    for sentence in sentences:
        financing = re.search(r'\b(proceeds|financing|borrowings|loan|loans|credit facility)\b', sentence)
        debt = re.search(r'\b(debt|notes|credit|facility|facilities|loan|loans)\b', sentence)
        refi = re.search(r'\b(refinanc(?:e|es|ed|ing)|replac(?:e|es|ed|ing))\b.{0,160}\b(debt|notes|credit|facility|facilities|loan|loans)\b', sentence)
        repay = re.search(r'\b(repay|repayment|prepay|prepayment|redeem|redemption|retire)\b', sentence)
        if refi or (financing and debt and repay):
            evidence['debt_refinancing'].append(sentence)
        extension = re.search(r'\b(extend(?:s|ed|ing)?|extension of)\b.{0,90}\b(maturity|expiration)\b', sentence)
        optional = re.search(r'\b(may|option|right|request|could)\b', sentence)
        if extension and not optional:
            evidence['maturity_extension'].append(sentence)
        waiver = re.search(r'\b(waiver|forbearance)\b', sentence)
        granted = re.search(r'\b(granted|obtained|provided|agreed)\b', sentence)
        breach = re.search(r'\b(covenant|breach|default)\b', sentence)
        if waiver and granted and breach:
            evidence['covenant_waiver'].append(sentence)
        capex = re.search(r'\b(capital expenditures?|construction|equipment|manufacturing plants?)\b', sentence)
        use = re.search(r'\b(fund|funding|finance|financing|used|use)\b', sentence)
        if financing and capex and use:
            evidence['capital_expenditure_funding'].append(sentence)
    return {q: {'choice': 'disclosed' if matches else 'not_disclosed', 'matched_sentences': matches}
            for q, matches in evidence.items()}
