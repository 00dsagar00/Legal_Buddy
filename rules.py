"""Rule-based knowledge layer: benchmarks, contract-type detection, red flags,
checklists, scoring and alternative wording.

Carried over from the original scratch code (app.py) with the Streamlit and
model dependencies removed, so it can be tested without any UI or ML model.
"""
from typing import Dict, List

import re


class RuleEngine:
    def __init__(self):
        self.load_benchmarks()

    def load_benchmarks(self):
        """Load benchmark data for contract analysis"""
        self.benchmarks = {
            "employment": {
                "notice_period": {
                    "normal": "30 days",
                    "range": "15-90 days",
                    "unit": "days",
                },
                "probation": {
                    "normal": "3-6 months",
                    "max_acceptable": "12 months",
                    "unit": "months",
                },
                "non_compete": {
                    "normal": "6-12 months",
                    "max_acceptable": "24 months",
                    "unit": "months",
                },
                "bond_penalty": {
                    "normal": "0-2 months salary",
                    "suspicious": ">6 months salary",
                    "unit": "months",
                },
                "severance": {
                    "typical": "0-3 months salary",
                    "suspicious": ">6 months salary",
                },
                "garden_leave": {
                    "typical": "0-3 months",
                    "suspicious": ">6 months",
                    "unit": "months",
                },
            },
            "rental": {
                "security_deposit": {
                    "normal": "2-3 months",
                    "suspicious": ">6 months",
                    "unit": "months",
                },
                "rent_increase": {
                    "normal": "5-10% annually",
                    "suspicious": ">15% annually",
                    "unit": "%",
                },
                "notice_period": {
                    "normal": "1-2 months",
                    "range": "15 days - 3 months",
                    "unit": "days",
                },
                "maintenance_responsibility": {
                    "flag_if_tenant": True,
                    "note": "Tenant should not be responsible for structural repairs",
                },
                "early_termination_penalty": {
                    "typical": "forfeiture of deposit",
                    "suspicious": "multiple months rent",
                },
            },
            "service": {
                "payment_terms": {
                    "normal": "30 days",
                    "range": "15-60 days",
                    "unit": "days",
                },
                "liability_cap": {
                    "normal": "contract value",
                    "suspicious": "unlimited",
                },
                "termination_notice": {
                    "normal": "30 days",
                    "range": "15-90 days",
                    "unit": "days",
                },
                "ip_assignment": {
                    "expected": "work-for-hire or client license",
                    "suspicious": "assignment of contractor pre-existing IP",
                },
                "sla_response_time": {
                    "critical": "<24 hours",
                    "typical": "24-72 hours",
                },
            },
            "nda": {
                "confidentiality_duration_years": {
                    "normal": "1-5",
                    "suspicious": ">5 or perpetual",
                },
                "disclosure_exceptions": {"required": True},
                "return_or_destruction": {"required": True},
            },
            "supplier": {
                "delivery_window_days": {
                    "typical": "30-90",
                    "suspicious": ">180",
                    "unit": "days",
                },
                "warranty_months": {
                    "typical": "3-24",
                    "suspicious": "none",
                    "unit": "months",
                },
                "payment_terms": {"normal": "30-60 days", "unit": "days"},
            },
            "contractor": {
                "milestone_payments": {"recommended": True},
                "retention_percentage": {
                    "typical": "5-10%",
                    "suspicious": ">20%",
                    "unit": "%",
                },
                "insurance_required": {"recommended": True},
            },
            "loan": {
                "interest_rate_max": {
                    "typical_annual_pct": "market_rate",
                    "suspicious_high_pct": ">30%",
                },
                "prepayment_penalty": {"flag_if_present": True},
                "security_description_required": {"required": True},
            },
            "partnership": {
                "profit_sharing_defined": {"required": True},
                "exit_mechanism": {"required": True},
                "voting_rights": {"required": True},
            },
            "software": {
                "license_scope": {"required": True},
                "support_sla": {"recommended": True},
                "source_code_escrow": {"recommended_if_critical": True},
            },
            "construction": {
                "performance_bond": {"typical_pct": "5-10%", "unit": "%"},
                "defect_liability_period_months": {"typical": "6-24", "unit": "months"},
                "liquidated_damages_rate": {
                    "typical": "0.1-0.5% per week",
                    "suspicious": ">1% per week",
                },
            },
            "procurement": {
                "order_of_precedence": {"recommended": True},
                "price_adjustment_clause": {"flag_if_absent": True},
                "termination_for_convenience": {"expected": True},
            },
            "sale_purchase": {
                "transfer_of_title": {"required": True},
                "inspection_period_days": {"typical": "7-30", "unit": "days"},
                "warranty_duration": {"typical_months": "3-12", "unit": "months"},
            },
            # Cross-cutting / general contract benchmarks
            "general": {
                "effective_date_required": {"required": True},
                "governing_law_required": {"required": True},
                "currency_specified": {"required": True},
                "nda_duration": {
                    "normal_years": "1-5",
                    "suspicious": "perpetual or >5 years",
                },
                "unlimited_liability": {"suspicious": True},
                "perpetual_terms": {
                    "suspicious_phrases": ["perpetual", "in perpetuity"]
                },
                "indemnity_scope": {
                    "flag_if_includes_innocent_negligence": True,
                    "note": "Indemnities covering innocent negligence or unlimited IP indemnities are high risk",
                },
                "ambiguous_payment_terms": {
                    "flag_phrases": ["reasonable time", "as agreed", "promptly"]
                },
                "assignment_restrictions": {
                    "flag_if_unrestricted": True,
                    "note": "Assignment without consent can transfer risk",
                },
            },
        }

    def identify_contract_type(self, text: str) -> str:
        """Identify the type of contract"""
        text_lower = text.lower()

        # Expanded keyword lists for more specific contract categories
        keyword_categories = {
            "employment": [
                "employee",
                "employment",
                "salary",
                "designation",
                "probation",
                "resignation",
                "epf",
                "esi",
                "gratuity",
            ],
            "rental": [
                "tenant",
                "landlord",
                "rent",
                "lease",
                "property",
                "premises",
                "security deposit",
                "broker",
            ],
            "service": [
                "service",
                "client",
                "deliverables",
                "milestone",
                "payment terms",
                "sla",
                "support",
            ],
            "nda": [
                "confidential",
                "non-disclosure",
                "nda",
                "confidentiality",
                "proprietary",
                "trade secret",
            ],
            "software": [
                "license",
                "software",
                "source code",
                "escrow",
                "api",
                "saas",
                "hosting",
            ],
            "construction": [
                "construction",
                "contractor",
                "builder",
                "site",
                "drawings",
                "completion",
                "defect liability",
            ],
            "supplier": [
                "supply",
                "supplier",
                "goods",
                "delivery",
                "inspection",
                "warranty",
            ],
            "procurement": [
                "procure",
                "purchase order",
                "tender",
                "bid",
                "po",
                "rfq",
                "rfi",
            ],
            "loan": [
                "loan",
                "interest",
                "principal",
                "repayment",
                "mortgage",
                "security",
                "foreclosure",
            ],
            "partnership": [
                "partner",
                "partnership",
                "profit share",
                "capital contribution",
                "dissolution",
            ],
            "contractor": [
                "independent contractor",
                "subcontractor",
                "retention",
                "milestone",
                "work order",
            ],
            "sale_purchase": [
                "sale",
                "purchase",
                "delivery",
                "goods",
                "transfer of title",
                "inspection",
            ],
        }

        # Score each category by keyword matches
        scores = {cat: 0 for cat in keyword_categories}
        for cat, keywords in keyword_categories.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[cat] += 1

        # Pick the highest-scoring category; tiebreaker by a simple priority list
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_cat, top_score = sorted_scores[0]

        # Return the top category
        return top_cat

    def detect_red_flags(
        self, clauses: List[Dict], contract_type: str, full_text: str = None
    ) -> List[Dict]:
        """Detect red flags in the contract.

        Uses clause-level risk plus some cross-cutting checks on the full contract text
        when available (e.g., missing governing law, missing effective date, currency).
        """
        red_flags: List[Dict] = []

        # Clause-level red flags
        for clause in clauses:
            if clause.get("risk_level") == "high":
                issue = self.identify_red_flag_issue(clause, contract_type)
                red_flag = {
                    "clause": clause.get("text", "")[:200] + "...",
                    "issue": issue,
                    "severity": "High",
                    "recommendation": self.get_red_flag_recommendation(
                        clause, contract_type
                    ),
                }
                red_flags.append(red_flag)

        # Cross-cutting checks using full_text
        if full_text:
            text_lower = full_text.lower()
            clause_types = {c.get("type") for c in clauses}

            # Governing law / jurisdiction
            if (
                "governing_law" not in clause_types
                and "jurisdiction" not in clause_types
                and not re.search(
                    r"governing law|jurisdiction|seat of arbitration", text_lower
                )
            ):
                red_flags.append(
                    {
                        "clause": "",
                        "issue": "Governing law / jurisdiction missing",
                        "severity": "High",
                        "recommendation": "Specify governing law and forum (e.g., courts/arbitration seat in India)",
                    }
                )

            # Effective date
            if "effective_date" not in clause_types and not re.search(
                r"effective date|effective from", text_lower
            ):
                red_flags.append(
                    {
                        "clause": "",
                        "issue": "Effective date not specified",
                        "severity": "Medium",
                        "recommendation": "Add an explicit effective date to avoid ambiguity",
                    }
                )

            # Currency
            if not re.search(r"₹|inr|rs\.?\s|rupee", text_lower):
                red_flags.append(
                    {
                        "clause": "",
                        "issue": "Currency not specified",
                        "severity": "Medium",
                        "recommendation": "Specify the currency for payments (e.g., INR, ₹)",
                    }
                )

            # Unlimited liability mention anywhere
            if re.search(
                r"unlimited liability|liability shall be unlimited", text_lower
            ):
                red_flags.append(
                    {
                        "clause": "",
                        "issue": "Unlimited liability exposure",
                        "severity": "High",
                        "recommendation": "Seek to cap liability to contract value or insurance limits",
                    }
                )

        return red_flags

    def identify_red_flag_issue(self, clause: Dict, contract_type: str) -> str:
        """Identify specific issues in red flag clauses using clause meta and benchmarks.

        clause: Dict should contain keys 'type', 'text', 'meta'.
        """
        clause_type = clause.get("type", "")
        text = clause.get("text", "").lower()
        meta = clause.get("meta", {}) or {}

        # Helper to parse numeric benchmark values (e.g., '24 months' -> 24)
        def _parse_first_int(s: str):
            if not s:
                return None
            m = re.search(r"(\d+)", s)
            return int(m.group(1)) if m else None

        # Use benchmarks when available
        bench = (
            self.benchmarks.get(contract_type, {})
            if hasattr(self, "benchmarks")
            else {}
        )

        # Specific clause handlers
        if clause_type == "non_compete":
            months = meta.get("months", [])
            if months:
                max_months = None
                # check bench fallback to employment.non_compete.max_acceptable
                if "non_compete" in bench:
                    max_months = _parse_first_int(
                        bench["non_compete"].get("max_acceptable", "")
                    )
                elif (
                    "employment" in self.benchmarks
                    and "non_compete" in self.benchmarks["employment"]
                ):
                    max_months = _parse_first_int(
                        self.benchmarks["employment"]["non_compete"].get(
                            "max_acceptable", ""
                        )
                    )
                if max_months and max(months) > max_months:
                    return f"Overly broad non-compete restrictions ({max(months)} months — exceeds {max_months} months)"
            return "Overly broad non-compete restrictions"

        if clause_type in ("penalty", "bond_penalty"):
            # Check percentages or amounts
            percents = meta.get("percent", [])
            amounts = meta.get("amounts", [])
            if percents and max(percents) > 15:
                return f"Excessive penalty rate ({max(percents)}%)"
            if amounts and any(a > 1e6 for a in amounts):
                return "Excessive monetary penalty amounts"
            return "Excessive penalty amounts"

        if clause_type == "liability":
            if "unlimited" in text:
                return "Unlimited liability exposure"
            return "Potentially broad liability clause"

        if clause_type in ("nda", "confidentiality"):
            if re.search(r"perpetual|in perpetuity", text):
                return "Perpetual confidentiality obligation (NDA)"
            years = meta.get("months", [])
            # months->years heuristic
            if years and max(years) >= 60:
                return "Very long confidentiality duration (>5 years)"
            return "Confidentiality / NDA clause"

        if clause_type == "indemnity":
            if re.search(r"innocent negligence|negligence", text):
                return "Indemnity covering negligence (including potentially innocent negligence)"
            return "Broad indemnity clause"

        if clause_type == "governing_law":
            return "Governing law / jurisdiction specified"

        if clause_type == "effective_date":
            return "Effective date specified"

        if clause_type == "security_deposit":
            # Check if deposit months exceed suspicious value
            months = meta.get("months", [])
            if months and max(months) > 6:
                return f"High security deposit detected ({max(months)} months)"
            return "Security deposit clause"

        # Default
        default_issues = {
            "non_compete": "Overly broad non-compete restrictions",
            "penalty": "Excessive penalty amounts",
            "liability": "Unlimited liability exposure",
            "termination": "Unfair termination conditions",
        }
        return default_issues.get(clause_type, "Potentially unfavorable terms")

    def get_red_flag_recommendation(self, clause: Dict, contract_type: str) -> str:
        """Get targeted recommendations for a red-flag clause using clause meta and benchmarks.

        clause should be a dict with keys 'type', 'text', and optional 'meta'.
        """
        clause_type = clause.get("type", "")
        text = clause.get("text", "")
        meta = clause.get("meta", {}) or {}

        # Helpers
        def _first_meta(key: str):
            vals = meta.get(key, [])
            return vals[0] if vals else None

        # Try to use benchmarks for contract_type, fall back to general employment values
        bench = (
            self.benchmarks.get(contract_type, {})
            if hasattr(self, "benchmarks")
            else {}
        )
        employment_bench = (
            self.benchmarks.get("employment", {}) if hasattr(self, "benchmarks") else {}
        )

        # Recommendations tailored by clause type
        if clause_type == "non_compete":
            months = _first_meta("months")
            max_allowed = None
            if "non_compete" in bench:
                max_allowed = bench["non_compete"].get("max_acceptable")
            elif "non_compete" in employment_bench:
                max_allowed = employment_bench["non_compete"].get("max_acceptable")
            rec = "Negotiate for reasonable time limits and geographic restrictions"
            if months and max_allowed:
                rec += f" — detected {months} months; consider reducing to <= {max_allowed}"
            rec += ". Consider compensation or narrower scope (role/geography)."
            return rec

        if clause_type in ("penalty", "bond_penalty"):
            perc = _first_meta("percent")
            if perc and perc > 15:
                return f"Request clarification and cap on penalty rate ({perc}%). Seek proportional penalties and a clear formula."
            amt = _first_meta("amounts")
            if amt:
                return "Request justification for penalty amounts and seek proportionality to actual damages. Consider an upper cap tied to contract value."
            return "Request clarification on penalty calculation and reasonableness. Consider capping penalties."

        if clause_type == "liability":
            return "Seek a liability cap (e.g., contract value or specified sum) and purchase/confirm appropriate insurance. Exclude indirect/consequential damages."

        if clause_type in ("nda", "confidentiality"):
            years = _first_meta("months")
            rec = "Limit confidentiality duration (e.g., 1-5 years) and add clear carve-outs for pre-existing information and legal disclosures."
            if years and years >= 60:
                rec = "Confidentiality appears very long — negotiate a limited duration (1-5 years) or specify narrow scope."
            if re.search(r"perpetual|in perpetuity", text, re.IGNORECASE):
                rec = "Perpetual NDA detected — negotiate a fixed duration and explicit carve-outs for pre-existing and independently developed information."
            return rec

        if clause_type == "indemnity":
            return "Narrow indemnity obligations: limit to breach, wilful misconduct or third-party IP infringement; exclude innocent negligence where appropriate; cap indemnity exposure."

        if clause_type == "security_deposit":
            months = _first_meta("months")
            rec = "Clarify security deposit amount, interest (if any), and refund conditions. Require reasonableness and timelines for refund."
            if months and months > 6:
                rec = f"Security deposit seems high ({months} months) — negotiate down or request interest and a cap (typical 2-3 months)."
            return rec

        if clause_type == "sla":
            return "Define SLA metrics clearly, include remedies like service credits (not unlimited damages), and cap total liability for SLA breaches. Ensure measurement and reporting are defined."

        if clause_type == "retention":
            return "Negotiate reasonable retention (typical 5-10%) and a defined release schedule after acceptance or defects liability period."

        if clause_type == "performance_bond":
            return "Request a reasonable performance security (e.g., 5-10% of contract) and specify expiry/claim conditions. Consider bank guarantee instead of upfront cash."

        if clause_type == "governing_law":
            return "Ensure governing law and dispute resolution mechanism are clearly defined; prefer arbitration seat in a mutually agreed city in India if suitable."

        if clause_type == "effective_date":
            return (
                "Add an explicit effective date and clarify when obligations commence."
            )

        if clause_type == "taxes":
            return "Specify who bears GST and withholding tax implications (TDS); include gross-up clause if required for certain payments."

        if clause_type == "assignment":
            return "Restrict assignment without consent, or require notice and conditions for assignment to protect parties from unexpected transfers."

        # Default recommendation
        return "Seek legal review for this clause and negotiate for clearer limits, caps or carve-outs as appropriate."

    def generate_checklist(self, contract_type: str, clauses: List[Dict]) -> List[str]:
        """Generate personalized checklist

        Supports multiple contract categories. Returns an ordered, deduplicated
        list of checklist items combining category-specific items, India- specific
        general checks, and items derived from detected clauses.
        """
        checklist: List[str] = []

        # Base checklist items by contract type (expanded categories)
        base_checklists: Dict[str, List[str]] = {
            "employment": [
                "Verify salary amount, components (basic, HRA, allowances) and in-hand pay",
                "Confirm EPF/ESI applicability and employer contributions",
                "Check gratuity, bonus/variable pay and calculation method",
                "Understand probation period, confirmation criteria and termination during probation",
                "Check notice period length and pay-in-lieu rules",
                "Review leave policies (paid leave, sick leave, maternity/paternity) and encashment",
                "Confirm working hours, overtime rules and compensation",
                "Review non-compete, non-solicit and restrictive covenants for scope and duration",
                "Check IP and invention assignment clauses (pre-existing contractor IP should be carved out)",
                "Verify performance targets/KPIs and appraisal frequency",
                "Confirm relocation/transfer and travel expense reimbursement policies",
                "Ensure statutory compliance (Shops & Establishment, labour laws) is referenced",
                "Confirm tax/TDS treatment and whether gross-up applies for certain benefits",
                "Look for background-check or offer-conditional clauses and their limits",
            ],
            "rental": [
                "Confirm rent amount, payment frequency and accepted payment methods",
                "Check security deposit amount, interest (if any) and refund conditions",
                "Verify whether stamp duty/registration is required (e.g., >11 months in many states)",
                "Clarify responsibility for maintenance, repairs and major structural works",
                "Check rent escalation clause and maximum cap or formula",
                "Confirm notice period for termination by landlord/tenant and early termination penalties",
                "Verify subletting and assignment restrictions",
                "Ensure inventory/condition report is attached or described",
                "Confirm who pays municipal/property taxes and utilities",
                "Check use-of-premises clause (residential vs commercial) and required approvals",
                "Look for broker fee responsibility and any other prepaid charges",
                "Check renewal terms and whether rent at renewal is capped or negotiable",
            ],
            "service": [
                "Clarify detailed scope, deliverables, acceptance criteria and milestone definitions",
                "Verify payment schedule, invoice requirements and GST handling",
                "Check late payment interest rate and dispute resolution on invoices",
                "Confirm IP ownership or license scope; carve out contractor pre-existing IP",
                "Review confidentiality and data protection obligations (PII handling, cross-border transfer)",
                "Check liability cap, exclusions (consequential/indirect) and indemnity carve-outs",
                "Verify SLA metrics, remedies, service credits and caps on remedies",
                "Confirm insurance requirements (professional indemnity, public liability)",
                "Check subcontracting/assignment rules and approval process",
                "Confirm termination for convenience vs for cause and transition assistance obligations",
                "Review tax clauses: GST applicability, who bears taxes, and TDS provisions",
                "Check governing law, dispute resolution and seat of arbitration (city in India)",
                "Ensure change control process and pricing for out-of-scope work is defined",
                "Check warranty period, support obligations and defect remediation timelines",
            ],
            "nda": [
                "Confirm parties and defined confidential information",
                "Check duration of confidentiality and whether perpetual NDA is used",
                "Verify permitted disclosures (court order, affiliates) and notice requirements",
                "Confirm return/destruction obligations for confidential materials",
                "Ensure carve-outs for pre-existing and independently developed information",
            ],
            "supplier": [
                "Check delivery terms, incoterms (if international) and delivery timelines",
                "Verify quality acceptance criteria and inspection rights",
                "Ensure clear pricing, taxes (GST) and invoice documentation",
                "Confirm warranty and remedy for defective goods/services",
                "Check lead times and penalties for late delivery",
            ],
            "contractor": [
                "Confirm contractor status (independent contractor vs employee) and liabilities",
                "Check milestones, deliverables and acceptance testing",
                "Verify IP assignment vs license and pre-existing IP carve-outs",
                "Review indemnity and insurance requirements",
                "Check payment milestones and retention terms",
            ],
            "loan": [
                "Confirm principal amount, interest rate (fixed/variable) and repayment schedule",
                "Check default interest, grace periods and events of default",
                "Verify security/collateral descriptions and perfection steps (hypothecation, mortgage)",
                "Confirm prepayment, foreclosure charges and break costs",
                "Review representations, covenants and cross-default clauses",
            ],
            "partnership": [
                "Verify capital contributions, ownership percentages and profit sharing",
                "Check management voting rights and decision-making processes",
                "Confirm exit/transfer rules, buy-sell mechanisms and valuation methods",
                "Review non-compete and non-solicit obligations between partners",
                "Ensure dispute resolution and dissolution processes are defined",
            ],
            "software": [
                "Confirm licensed rights (scope, duration, users) and restrictions",
                "Check support and maintenance obligations, SLAs and response times",
                "Verify escrow and source code access on vendor insolvency (if critical)",
                "Review data protection, hosting location and cross-border transfer terms",
                "Ensure clear warranty, limitation of liability and IP indemnity clauses",
            ],
            "construction": [
                "Verify scope, drawings, specifications and approval process",
                "Check performance bonds, retention and defect liability period",
                "Confirm milestone payments, variations and change orders handling",
                "Review health & safety, site access and local statutory compliance",
                "Ensure delay/liquidated damages clauses are reasonable and capped",
            ],
            "procurement": [
                "Confirm tender/PO references, acceptance and order of precedence",
                "Verify supplier qualification criteria and documentation requirements",
                "Check price adjustment, indexation and escalation clauses",
                "Ensure termination for convenience with defined notice and compensation",
                "Review audit and compliance rights for suppliers",
            ],
            "sale_purchase": [
                "Confirm description of goods/services, quantity and delivery terms",
                "Check transfer of title and risk allocation",
                "Verify warranties, returns and indemnities for breach of specifications",
                "Confirm pricing, taxes (GST) and payment terms",
                "Ensure conditions precedent (inspections, approvals) are clearly stated",
            ],
        }

        checklist.extend(base_checklists.get(contract_type, []))
        if not base_checklists.get(contract_type):
            # For unknown contract types, include a small generic starter
            checklist.append("Confirm scope, parties, effective date and payment terms")

        # Add specific items based on detected clauses
        for clause in clauses:
            if clause.get("risk_level") in ["high", "medium"]:
                if "non_compete" in clause.get("type", ""):
                    checklist.append(
                        "Ask for clarification on non-compete restrictions and duration"
                    )
                elif "penalty" in clause.get("type", ""):
                    checklist.append("Request justification for penalty amounts")
                elif "confidentiality" in clause.get("type", ""):
                    checklist.append(
                        "Understand what information is considered confidential"
                    )

        # Preserve order and remove duplicates
        return list(dict.fromkeys(checklist))

    def calculate_confidence_score(
        self, clauses: List[Dict], red_flags: List[Dict]
    ) -> Dict:
        """Calculate overall confidence score"""
        total_clauses = len(clauses)
        high_risk_clauses = sum(1 for c in clauses if c["risk_level"] == "high")
        medium_risk_clauses = sum(1 for c in clauses if c["risk_level"] == "medium")

        # Calculate score (0-100)
        if total_clauses == 0:
            score = 50
        else:
            penalty = (high_risk_clauses * 12) + (medium_risk_clauses * 4)
            score = max(0, 100 - penalty)

        # Determine rating
        if score >= 80:
            rating = "Excellent"
            color = "#4CAF50"
        elif score >= 60:
            rating = "Good"
            color = "#FFC107"
        elif score >= 40:
            rating = "Fair"
            color = "#FF9800"
        else:
            rating = "Poor"
            color = "#F44336"

        return {
            "score": score,
            "rating": rating,
            "color": color,
            "details": {
                "total_clauses": total_clauses,
                "high_risk": high_risk_clauses,
                "medium_risk": medium_risk_clauses,
                "red_flags": len(red_flags),
            },
        }

    def suggest_alternatives(
        self, red_flags: List[Dict], contract_type: str
    ) -> List[Dict]:
        """Suggest alternative clause wordings"""
        alternatives = []

        for flag in red_flags:
            alternative = {
                "original_issue": flag["issue"],
                "suggested_wording": self.get_alternative_wording(flag, contract_type),
                "explanation": "This alternative provides more balanced terms",
            }
            alternatives.append(alternative)

        return alternatives

    def get_alternative_wording(self, red_flag: Dict, contract_type: str) -> str:
        """Get alternative clause wording"""
        alternatives = {
            "Overly broad non-compete restrictions": "Employee agrees not to work for direct competitors in the same role within [specific geographic area] for [reasonable time period] months.",
            "Excessive penalty amounts": "Any penalties shall be reasonable and proportionate to actual damages incurred.",
            "Unlimited liability exposure": "Liability shall be limited to the total value of this contract or actual damages, whichever is lower.",
            "Unfair termination conditions": "Either party may terminate this agreement with [reasonable notice period] days written notice.",
        }

        return alternatives.get(
            red_flag["issue"], "Consider negotiating for more balanced and fair terms."
        )


    def generate_plain_language_explanation(
        self, clauses: List[Dict], contract_type: str
    ) -> List[Dict]:
        """Generate plain language explanations for clauses"""
        explanations = []

        for clause in clauses:
            explanation = {
                "original": clause["text"],
                "plain_english": self.convert_to_plain_language(
                    clause["text"], clause["type"]
                ),
                "practical_meaning": self.get_practical_meaning(
                    clause["text"], clause["type"], contract_type
                ),
                "risk_level": clause["risk_level"],
            }
            explanations.append(explanation)

        return explanations

    def convert_to_plain_language(self, clause: str, clause_type: str) -> str:
        """Convert legal jargon to plain language"""
        # Simple rule-based conversion (in production, use GPT API)
        conversions = {
            "hereby": "by this document",
            "whereas": "since",
            "heretofore": "before this",
            "hereafter": "after this",
            "notwithstanding": "despite",
            "pursuant to": "according to",
            "in consideration of": "in exchange for",
            "terminate": "end",
            "commence": "begin",
            "remuneration": "payment/salary",
        }

        plain_text = clause
        for legal_term, plain_term in conversions.items():
            plain_text = re.sub(legal_term, plain_term, plain_text, flags=re.IGNORECASE)

        return plain_text

    def get_practical_meaning(
        self, clause: str, clause_type: str, contract_type: str
    ) -> str:
        """Get practical meaning of clauses"""
        # Expanded template-based practical meanings for many clause types
        templates = {
            "termination": {
                "employment": "Explains how and when employment can be ended, notice required, and consequences for termination.",
                "rental": "Explains how the tenancy can be ended by tenant or landlord and what notices are required.",
                "service": "Explains termination rights for both parties and any transition assistance required.",
            },
            "termination_for_convenience": {
                "default": "Allows a party to end the contract without cause, usually with notice and sometimes with compensation."
            },
            "non_compete": {
                "employment": "Restricts post-employment work in certain roles, industries or geographies — affects future job options.",
                "service": "May restrict the contractor from doing similar work for others — check scope and duration.",
            },
            "confidentiality": {
                "default": "Defines what information must be kept confidential, duration of confidentiality and permitted disclosures."
            },
            "nda": {
                "default": "A separate NDA or confidentiality section: sets limits on use and disclosure of sensitive information."
            },
            "payment": {
                "employment": "Describes salary, payroll cycle, and deductions (taxes, benefits).",
                "rental": "Specifies rent due dates, late fees and payment methods.",
                "service": "Specifies invoicing, due dates, and accepted payment methods.",
            },
            "payment_terms": {
                "default": "Describes when invoices are due, any milestones, and consequences of late payment."
            },
            "liability": {
                "default": "Defines financial responsibility for loss; look for caps, exclusions and insurance requirements."
            },
            "indemnity": {
                "default": "Obligation to compensate for third-party claims — watch for scope (IP, negligence) and caps."
            },
            "notice": {
                "default": "How to give formal notices (method, address, and timelines) which trigger contractual processes."
            },
            "penalty": {
                "default": "Penalty clauses impose remedies or charges for breaches — ensure they're proportional and defined."
            },
            "governing_law": {
                "default": "Specifies which country's/state's law governs the contract and where disputes will be resolved."
            },
            "effective_date": {
                "default": "Specifies when contract obligations start — important for timelines, limitations and notice periods."
            },
            "security_deposit": {
                "rental": "An amount held to secure performance — check amount, interest, and refund conditions.",
                "default": "Funds held as security for obligations; check refund and use conditions.",
            },
            "probation": {
                "employment": "Initial period of employment when termination may be easier and benefits/entitlements may be limited."
            },
            "severance": {
                "employment": "Compensation on termination — check amount, triggers and calculation method."
            },
            "ip": {
                "default": "Describes ownership and licensing of intellectual property created or used under the contract."
            },
            "sla": {
                "service": "Service levels the supplier must meet (uptime, response time); remedies are often credits rather than direct damages."
            },
            "warranty": {
                "default": "Promises about product/service quality and the remedies available if warranties are breached."
            },
            "retention": {
                "contractor": "A portion of payment held back until completion/defects liability period to ensure performance."
            },
            "performance_bond": {
                "construction": "Security (bond/bank guarantee) to ensure contractor completes work; check claim conditions and expiry."
            },
            "assignment": {
                "default": "Whether parties can transfer rights/obligations to third parties — unrestricted assignment can transfer risk."
            },
            "taxes": {
                "default": "Specifies who bears GST/TDS/withholding and any gross-up obligations."
            },
            "stamp_duty": {
                "default": "Indicates whether stamp duty/registration is required which can affect enforceability or additional costs."
            },
            "arbitration": {
                "default": "Specifies arbitration as dispute resolution — check seat, rules and enforceability."
            },
        }

        # Return a contract-type-specific template if available, otherwise default
        entry = templates.get(clause_type, {})
        return entry.get(
            contract_type,
            entry.get(
                "default",
                "This clause defines specific terms and conditions for this agreement.",
            ),
        )
