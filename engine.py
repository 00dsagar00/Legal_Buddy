"""Core engine: contract segmentation, InLegalBERT clause classification and
retrieval, and GPT-generated explanations / summaries / chat answers.

No Streamlit imports here, so everything can be tested headlessly.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from rules import RuleEngine

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
MODEL_NAME = "law-ai/InLegalBERT"


def load_inlegalbert():
    """Return (tokenizer, model) or (None, None) if loading fails."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        return tok, model
    except Exception as e:  # network, disk, missing deps ...
        print(f"[engine] Could not load {MODEL_NAME}: {e}")
        return None, None


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
_CLAUSE_START = re.compile(
    r"^\s*(?:\d{1,2}(?:\.\d{1,2})*[\.\)]\s+|\(?[a-z]\)\s+|\(?[ivx]{1,4}\)\s+|"
    r"(?:clause|article|section|schedule)\s+\d+)",
    re.IGNORECASE,
)


def segment_contract(text: str, min_chars: int = 60, max_chars: int = 900) -> List[str]:
    """Split raw PDF text into clause-sized chunks.

    PDF extraction hard-wraps lines, so we rebuild paragraphs first (blank
    lines / numbered clause starts begin a new block), then split oversized
    blocks on sentence boundaries and glue tiny fragments (headings) onto the
    following block.
    """
    blocks: List[str] = []
    cur: List[str] = []
    for line in text.replace("\r", "").split("\n"):
        s = line.strip()
        if not s:
            if cur:
                blocks.append(" ".join(cur))
                cur = []
            continue
        if cur and _CLAUSE_START.match(s):
            blocks.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        blocks.append(" ".join(cur))

    pieces: List[str] = []
    for b in blocks:
        b = re.sub(r"\s+", " ", b).strip()
        if len(b) <= max_chars:
            pieces.append(b)
            continue
        buf = ""
        for sent in re.split(r"(?<=[.;:])\s+(?=[A-Z(\d])", b):
            if buf and len(buf) + len(sent) + 1 > max_chars:
                pieces.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            pieces.append(buf)

    merged: List[str] = []
    carry = ""
    for p in pieces:
        p = f"{carry} {p}".strip() if carry else p
        if len(p) < min_chars:
            carry = p
            continue
        merged.append(p)
        carry = ""
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]} {carry}"
        else:
            merged.append(carry)
    return merged


# --------------------------------------------------------------------------
# Clause taxonomy: keyword evidence + prototype sentences for embeddings
# --------------------------------------------------------------------------
KEYWORDS: Dict[str, str] = {
    "termination_for_convenience": r"termination for convenience|terminate .{0,40}without cause",
    "termination": r"\bterminat(?:e|es|ed|ion)\b",
    "non_compete": r"non[\s-]?compet|restraint of trade|shall not (?:join|work for) .{0,40}compet",
    "nda": r"non[\s-]?disclosure|\bnda\b",
    "confidentiality": r"\bconfidential",
    "payment_terms": r"payment terms|invoice.{0,40}(?:days|due)|net\s*\d+",
    "payment": r"\bpayment|\bsalary\b|\bremuneration\b|\brent\b|\bfees?\b",
    "liability": r"\bliabilit",
    "indemnity": r"\bindemni",
    "notice": r"\bnotice\b",
    "penalty": r"\bpenalt|liquidated damages|\bforfeit",
    "governing_law": r"governing law|jurisdiction|governed by the laws|courts? (?:at|of) ",
    "effective_date": r"effective date|effective from|commenc(?:es|ement) (?:on|from)",
    "security_deposit": r"security deposit|caution money|\bdeposit\b",
    "probation": r"\bprobation",
    "severance": r"\bseverance",
    "ip": r"intellectual property|\binventions?\b|ownership of deliverables|work[\s-]for[\s-]hire|copyright",
    "sla": r"\bsla\b|service level|\buptime\b|response time",
    "warranty": r"\bwarrant(?:y|ies)\b",
    "retention": r"\bretention (?:money|amount|percentage)|\bretention\b",
    "performance_bond": r"performance bond|bank guarantee",
    "assignment": r"\bassign(?:s|ment|ed)?\b|sub[\s-]?contract",
    "taxes": r"\bgst\b|\btds\b|\btax(?:es)?\b|withholding",
    "stamp_duty": r"stamp duty|registration charges|\bregistered\b",
    "arbitration": r"\barbitra",
}
_KW_RX = {k: re.compile(v, re.IGNORECASE) for k, v in KEYWORDS.items()}
LABELS = list(KEYWORDS)

PROTOTYPES: Dict[str, List[str]] = {
    "termination": [
        "Either party may terminate this agreement by giving written notice if the other party commits a material breach.",
        "This agreement shall stand terminated upon expiry of the term or on default not cured within thirty days.",
    ],
    "termination_for_convenience": [
        "Either party may terminate this agreement without cause by giving thirty days prior written notice.",
    ],
    "non_compete": [
        "The employee shall not engage in any competing business or join a competitor for twelve months after leaving employment.",
    ],
    "nda": [
        "The receiving party shall not disclose the disclosing party's proprietary information to any third party.",
    ],
    "confidentiality": [
        "The parties shall keep all confidential information secret and use it only for the purposes of this agreement.",
    ],
    "payment_terms": [
        "Invoices shall be raised monthly and the client shall pay each invoice within thirty days of receipt.",
    ],
    "payment": [
        "The salary or rent shall be paid on or before the seventh day of every month by bank transfer.",
        "The fees payable under this agreement are set out in the schedule and are exclusive of GST.",
    ],
    "liability": [
        "The total liability of the service provider under this agreement shall not exceed the fees paid in the preceding twelve months.",
    ],
    "indemnity": [
        "The contractor shall indemnify and hold harmless the client against all losses, claims and damages arising from its breach.",
    ],
    "notice": [
        "Any notice under this agreement shall be in writing and delivered by registered post or email to the address stated above.",
        "Either party must give two months notice before vacating or resigning.",
    ],
    "penalty": [
        "In case of default the defaulting party shall pay a penalty or liquidated damages of an amount equal to three months of fees.",
    ],
    "governing_law": [
        "This agreement shall be governed by the laws of India and the courts at New Delhi shall have exclusive jurisdiction.",
    ],
    "effective_date": [
        "This agreement shall come into force and be effective from the date of signing.",
    ],
    "security_deposit": [
        "The tenant has paid an interest free refundable security deposit equal to two months rent, refundable at the end of tenancy.",
    ],
    "probation": [
        "The employee shall be on probation for six months, extendable at the sole discretion of the company.",
    ],
    "severance": [
        "On termination without cause the employee shall be entitled to severance pay equal to one month of salary per year of service.",
    ],
    "ip": [
        "All intellectual property rights in the deliverables and inventions created during the engagement shall belong to the client.",
    ],
    "sla": [
        "The service provider shall maintain ninety nine percent uptime and respond to critical issues within four hours.",
    ],
    "warranty": [
        "The supplier warrants that the goods are free from defects for a period of twelve months from delivery.",
    ],
    "retention": [
        "Five percent of each running bill shall be retained as retention money until the end of the defects liability period.",
    ],
    "performance_bond": [
        "The contractor shall furnish a bank guarantee equal to ten percent of the contract value as performance security.",
    ],
    "assignment": [
        "Neither party may assign or transfer its rights or obligations under this agreement without prior written consent.",
    ],
    "taxes": [
        "All payments shall be subject to deduction of TDS and the client shall bear applicable GST.",
    ],
    "stamp_duty": [
        "The stamp duty and registration charges on this agreement shall be borne equally by both parties.",
    ],
    "arbitration": [
        "Any dispute shall be referred to a sole arbitrator under the Arbitration and Conciliation Act, 1996, with the seat at Mumbai.",
    ],
}

# Hybrid classifier knobs. NOTE: these were set from first principles; tune
# them on a few of your own labelled contracts once InLegalBERT is available.
EMB_ONLY_MIN_SIM = 0.40  # centred cosine needed to label a chunk with no keyword hit
EMB_ONLY_MIN_MARGIN = 0.05  # gap between best and second-best label for emb-only labels

HIGH_RISK_WORDS = [
    "unlimited", "perpetual", "in perpetuity", "irrevocable", "sole discretion",
    "waive all", "forfeit", "unconditional", "without notice", "at its discretion",
    "indemnify and hold harmless", "liquidated damages", "no refund", "non-refundable",
]
MEDIUM_RISK_WORDS = [
    "material breach", "exclusive", "restrict", "prohibit", "binding", "lock-in",
    "lock in", "substantial", "significant", "auto-renew", "automatically renew",
]


def extract_clause_meta(clause_text: str) -> Dict:
    """Pull numbers (months, days, %, rupee amounts, dates) out of a clause."""
    meta: Dict = {"months": [], "days": [], "percent": [], "amounts": [], "dates": []}
    meta["months"] = [int(x) for x in re.findall(r"(\d+)\s*(?:months?|mos)\b", clause_text, re.I)]
    meta["days"] = [int(x) for x in re.findall(r"(\d+)\s*days?\b", clause_text, re.I)]
    meta["percent"] = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", clause_text)]
    for amt in re.findall(r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)", clause_text, re.I):
        try:
            meta["amounts"].append(float(amt.replace(",", "")))
        except ValueError:
            pass
    meta["dates"] = re.findall(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}", clause_text)
    return meta


# --------------------------------------------------------------------------
# Document index (chunks + embeddings + lexical stats), built once per upload
# --------------------------------------------------------------------------
_STOP = set(
    "the a an and or of to in on for by with is are be as at this that shall may will "
    "any all it its from not which such under upon if has have been was were".split()
)


def _tok(s: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP]


@dataclass
class DocIndex:
    chunks: List[str]
    emb: Optional[np.ndarray] = None  # L2-normalised, mean-centred
    mu: Optional[np.ndarray] = None  # centring vector (shared with prototypes + queries)
    tf: List[Counter] = field(default_factory=list)
    idf: Dict[str, float] = field(default_factory=dict)
    avg_len: float = 1.0


# --------------------------------------------------------------------------
# GPT layer
# --------------------------------------------------------------------------
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODELS = {"gemini": "gemini-3.5-flash", "openai": "gpt-4o-mini"}


def _parse_json(raw: str) -> dict:
    """Parse model JSON, tolerating ```json fences or stray text around the object."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


class GPTExplainer:
    """LLM layer (Gemini or OpenAI, both through the OpenAI SDK). Every method
    degrades to None/'' when no key is configured, so the app still works
    (rule-based) without an LLM.

    Provider choice: LLM_PROVIDER=gemini|openai, otherwise Gemini if
    GEMINI_API_KEY is set, else OpenAI if OPENAI_API_KEY is set."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 provider: str = "openai"):
        self.provider = provider
        self.api_key = api_key
        self.model = model or DEFAULT_MODELS.get(provider, "gpt-4o-mini")
        self.client = None
        self.last_error: Optional[str] = None
        if self.api_key:
            try:
                from openai import OpenAI

                kwargs = {"api_key": self.api_key}
                if provider == "gemini":
                    kwargs["base_url"] = GEMINI_BASE_URL
                self.client = OpenAI(**kwargs)
            except Exception as e:
                self.last_error = str(e)

    @classmethod
    def from_env(cls, get=os.getenv) -> "GPTExplainer":
        """Build from backend config. `get(name)` returns a string or ''/None."""
        provider = (get("LLM_PROVIDER") or "").strip().lower()
        gemini_key = (get("GEMINI_API_KEY") or "").strip()
        openai_key = (get("OPENAI_API_KEY") or "").strip()
        if provider not in ("gemini", "openai"):
            provider = "gemini" if gemini_key else "openai"
        if provider == "gemini":
            return cls(gemini_key or None, (get("GEMINI_MODEL") or "").strip() or None, "gemini")
        return cls(openai_key or None, (get("OPENAI_MODEL") or "").strip() or None, "openai")

    @property
    def available(self) -> bool:
        return self.client is not None

    def _complete(self, messages, json_mode=False, max_tokens=1200, temperature=0.2):
        if self.provider == "gemini":
            max_tokens *= 3  # Gemini "thinking" tokens count toward the limit
        kwargs = dict(
            model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def _json(self, system: str, user: str, max_tokens=1500) -> Optional[dict]:
        if not self.available:
            return None
        try:
            raw = self._complete(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                json_mode=True,
                max_tokens=max_tokens,
            )
            return _parse_json(raw)
        except Exception as e:
            self.last_error = str(e)
            return None

    # -- clause explanations -------------------------------------------------
    def explain_clauses(self, items: List[dict], contract_type: str) -> Dict[int, dict]:
        """items: [{'id', 'type', 'text', 'rule_risk'}] -> {id: explanation}"""
        system = (
            "You help ordinary people in India understand contracts. For each clause, reply in "
            "plain English a non-lawyer can follow (Class 10 reading level, no jargon). Be "
            "faithful to the clause text; never invent terms that are not in it. This is "
            "information, not legal advice. Reply with JSON only."
        )
        out: Dict[int, dict] = {}
        for i in range(0, len(items), 6):  # small batches keep JSON reliable
            batch = items[i : i + 6]
            user = (
                f"Contract type: {contract_type}\n\nClauses (JSON):\n"
                f"{json.dumps(batch, ensure_ascii=False)}\n\n"
                'Return {"items":[{"id":int,"plain_english":str (1-2 sentences),'
                '"practical_meaning":str (what it means for the signer in practice),'
                '"risk_level":"low|medium|high","risk_reason":str (one sentence),'
                '"negotiation_tip":str (one sentence, or "")}]}. '
                "Judge risk from the signer's point of view under Indian law/practice; a clause is "
                "high only if it is unusually one-sided, uncapped, or likely unenforceable/harsh."
            )
            data = self._json(system, user, max_tokens=1800)
            for it in (data or {}).get("items", []):
                try:
                    out[int(it["id"])] = it
                except (KeyError, ValueError, TypeError):
                    continue
        return out

    # -- risk-analysed summary -----------------------------------------------
    def summarize(self, contract_type: str, clause_digest: List[dict], red_flags: List[dict]) -> Optional[dict]:
        system = (
            "You summarise contracts for non-lawyers in India. Be specific and faithful to the "
            "provided clauses; do not invent terms. This is information, not legal advice. JSON only."
        )
        user = (
            f"Contract type: {contract_type}\n"
            f"Clauses (id, type, risk, text):\n{json.dumps(clause_digest, ensure_ascii=False)}\n"
            f"Red flags found by rules:\n{json.dumps(red_flags[:10], ensure_ascii=False)}\n\n"
            'Return {"headline":str,"summary":str (4-6 sentences),'
            '"key_terms":[str] (max 6),"top_risks":[{"title":str,"why":str,"clause_id":int|null}] (max 5),'
            '"overall_risk":"low|medium|high"}.'
        )
        return self._json(system, user, max_tokens=1200)

    # -- grounded chat ---------------------------------------------------------
    def answer(self, question: str, passages: List[Tuple[int, str]], history: List[dict]) -> str:
        context = "\n\n".join(f"[Section {i + 1}] {t}" for i, t in passages)
        system = (
            "You are a careful assistant answering questions about ONE uploaded Indian contract. "
            "Use ONLY the excerpts provided. Cite the sections you used like [Section 3]. If the "
            "excerpts do not answer the question, say the contract excerpts you found don't cover "
            "it and suggest what to look for - do not guess. Use plain language, be concise "
            "(under 150 words unless asked for more), and note when something should be checked with "
            "a lawyer. You are not giving legal advice."
        )
        msgs = [{"role": "system", "content": system}]
        for m in history[-6:]:
            msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": f"Contract excerpts:\n{context}\n\nQuestion: {question}"})
        try:
            return self._complete(msgs, max_tokens=500, temperature=0.2).strip()
        except Exception as e:
            self.last_error = str(e)
            print(f"[engine] AI request failed: {e}")  # server log only
            return "⚠️ The AI model is unavailable right now. The most relevant contract sections are shown below."


# --------------------------------------------------------------------------
# The analyzer used by the Streamlit app
# --------------------------------------------------------------------------
class ContractAnalyzer(RuleEngine):
    def __init__(self, tokenizer=None, model=None, gpt: Optional[GPTExplainer] = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.legal_model = model
        self.gpt = gpt or GPTExplainer()
        self._proto_raw: Optional[np.ndarray] = None
        self._proto_labels: List[str] = []

    # ---- helpers ------------------------------------------------------------
    @property
    def has_model(self) -> bool:
        return self.tokenizer is not None and self.legal_model is not None

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text with pypdf. Returns '' if the PDF has no text layer."""
        from pypdf import PdfReader

        try:
            reader = PdfReader(pdf_file)
            pages = [(p.extract_text() or "") for p in reader.pages]
            return "\n".join(pages).strip()
        except Exception as e:
            self.last_pdf_error = str(e)
            return ""

    def _embed(self, texts: List[str], max_length: int = 256, batch: int = 16) -> np.ndarray:
        import torch

        device = next(self.legal_model.parameters()).device
        outs = []
        for i in range(0, len(texts), batch):
            enc = self.tokenizer(
                texts[i : i + batch], return_tensors="pt", truncation=True,
                max_length=max_length, padding=True,
            ).to(device)
            with torch.no_grad():
                hidden = self.legal_model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # masked mean pooling
            outs.append(pooled.cpu().numpy())
        return np.vstack(outs)

    @staticmethod
    def _center_norm(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        x = x - mu
        return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)

    def _prototypes(self):
        if self._proto_raw is None:
            texts, labels = [], []
            for lab, sents in PROTOTYPES.items():
                for s in sents:
                    texts.append(s)
                    labels.append(lab)
            self._proto_raw = self._embed(texts)
            self._proto_labels = labels
        return self._proto_raw, self._proto_labels

    # ---- indexing -----------------------------------------------------------
    def build_index(self, text: str) -> DocIndex:
        chunks = segment_contract(text)
        idx = DocIndex(chunks=chunks)

        # lexical stats (BM25) - always available
        idx.tf = [Counter(_tok(c)) for c in chunks]
        n = max(len(chunks), 1)
        df = Counter(w for c in idx.tf for w in c)
        idx.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        idx.avg_len = max(sum(sum(c.values()) for c in idx.tf) / n, 1.0)

        # InLegalBERT embeddings (mean-centred; raw BERT vectors are very anisotropic)
        if self.has_model and chunks:
            try:
                raw = self._embed(chunks)
                proto, _ = self._prototypes()
                idx.mu = np.vstack([raw, proto]).mean(axis=0, keepdims=True)
                idx.emb = self._center_norm(raw, idx.mu)
            except Exception as e:
                print(f"[engine] embedding failed, falling back to lexical only: {e}")
                idx.emb = idx.mu = None
        return idx

    # ---- classification -----------------------------------------------------
    def assess_clause_risk(self, clause_type: str, clause_text: str, meta: Optional[Dict] = None) -> str:
        t = clause_text.lower()
        meta = meta or extract_clause_meta(clause_text)
        if any(w in t for w in HIGH_RISK_WORDS):
            return "high"
        months = max(meta["months"], default=0)
        days = max(meta["days"], default=0)
        if clause_type == "non_compete" and months > 24:
            return "high"
        if clause_type == "security_deposit" and months > 6:
            return "high"
        if clause_type in ("notice", "termination") and (days > 90 or months > 3):
            return "medium"
        if any(w in t for w in MEDIUM_RISK_WORDS):
            return "medium"
        return "low"

    def analyze_legal_clauses(self, text: str, index: Optional[DocIndex] = None) -> List[Dict]:
        """Classify every chunk into a clause type (keyword evidence + InLegalBERT
        prototype similarity) and attach metadata + a rule-based risk level."""
        index = index or self.build_index(text)
        sims = None
        if index.emb is not None:
            proto_raw, plabels = self._prototypes()
            proto = self._center_norm(proto_raw, index.mu)
            # per-label similarity = best prototype of that label
            sims = np.full((len(index.chunks), len(LABELS)), -1.0)
            raw = index.emb @ proto.T
            for j, lab in enumerate(plabels):
                col = LABELS.index(lab)
                sims[:, col] = np.maximum(sims[:, col], raw[:, j])

        clauses: List[Dict] = []
        for i, chunk in enumerate(index.chunks):
            cands = [lab for lab in LABELS if _KW_RX[lab].search(chunk)]
            label, conf, source = None, 0.0, ""
            if sims is not None:
                row = sims[i]
                if cands:
                    label = max(cands, key=lambda l: row[LABELS.index(l)])
                    conf = min(0.99, 0.6 + 0.4 * max(0.0, float(row[LABELS.index(label)])))
                    source = "keyword+InLegalBERT"
                else:
                    order = np.argsort(-row)
                    best, second = row[order[0]], row[order[1]]
                    if best >= EMB_ONLY_MIN_SIM and (best - second) >= EMB_ONLY_MIN_MARGIN:
                        label = LABELS[order[0]]
                        conf = float(min(0.9, best))
                        source = "InLegalBERT"
            elif cands:
                label, conf, source = cands[0], 0.5, "keyword"
            if label is None:
                continue
            meta = extract_clause_meta(chunk)
            clauses.append(
                {
                    "id": i,
                    "type": label,
                    "text": chunk,
                    "meta": meta,
                    "risk_level": self.assess_clause_risk(label, chunk, meta),
                    "rule_risk": self.assess_clause_risk(label, chunk, meta),
                    "confidence": round(conf, 2),
                    "source": source,
                    "also_matches": [c for c in cands if c != label],
                }
            )
        return clauses

    # ---- GPT enrichment -----------------------------------------------------
    def enrich_with_gpt(self, clauses: List[Dict], contract_type: str, max_clauses: int = 18) -> int:
        """Add plain-language text + refined risk to the most important clauses.
        Returns the number of clauses enriched (0 when GPT is unavailable)."""
        if not self.gpt.available or not clauses:
            return 0
        order = {"high": 0, "medium": 1, "low": 2}
        picked, seen_low = [], set()
        for c in sorted(clauses, key=lambda c: (order[c["risk_level"]], -c["confidence"])):
            if c["risk_level"] == "low":
                if c["type"] in seen_low:
                    continue
                seen_low.add(c["type"])
            picked.append(c)
            if len(picked) >= max_clauses:
                break
        payload = [
            {"id": c["id"], "type": c["type"], "text": c["text"][:700], "rule_risk": c["rule_risk"]}
            for c in picked
        ]
        result = self.gpt.explain_clauses(payload, contract_type)
        by_id = {c["id"]: c for c in clauses}
        n = 0
        for cid, ex in result.items():
            c = by_id.get(cid)
            if not c:
                continue
            c["plain_english"] = ex.get("plain_english", "")
            c["practical_meaning"] = ex.get("practical_meaning", "")
            c["risk_reason"] = ex.get("risk_reason", "")
            c["negotiation_tip"] = ex.get("negotiation_tip", "")
            if ex.get("risk_level") in ("low", "medium", "high"):
                c["risk_level"] = ex["risk_level"]  # GPT refines the keyword heuristic
            n += 1
        return n

    def generate_plain_language_explanation(self, clauses: List[Dict], contract_type: str) -> List[Dict]:
        out = []
        for c in clauses:
            out.append(
                {
                    "original": c["text"],
                    "plain_english": c.get("plain_english")
                    or self.convert_to_plain_language(c["text"], c["type"]),
                    "practical_meaning": c.get("practical_meaning")
                    or self.get_practical_meaning(c["text"], c["type"], contract_type),
                    "risk_level": c["risk_level"],
                    "risk_reason": c.get("risk_reason", ""),
                    "negotiation_tip": c.get("negotiation_tip", ""),
                    "gpt": "plain_english" in c,
                }
            )
        return out

    def risk_summary(self, clauses: List[Dict], contract_type: str, full_text: str) -> Optional[dict]:
        if not self.gpt.available:
            return None
        flags = self.detect_red_flags(clauses, contract_type, full_text=full_text)
        order = {"high": 0, "medium": 1, "low": 2}
        digest = [
            {"id": c["id"], "type": c["type"], "risk": c["risk_level"], "text": c["text"][:350]}
            for c in sorted(clauses, key=lambda c: order[c["risk_level"]])[:25]
        ]
        return self.gpt.summarize(contract_type, digest, flags)

    # ---- retrieval + chat ---------------------------------------------------
    def _bm25(self, index: DocIndex, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        q = _tok(query)
        scores = np.zeros(len(index.chunks))
        for i, tf in enumerate(index.tf):
            dl = sum(tf.values())
            for w in q:
                f = tf.get(w, 0)
                if f:
                    scores[i] += index.idf.get(w, 0) * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / index.avg_len))
        return scores

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)

    def semantic_search_chunks(self, index: DocIndex, query: str, top_k: int = 4) -> List[Tuple[int, float, str]]:
        """Hybrid retrieval: InLegalBERT cosine (centred) + BM25, min-max fused."""
        if not index.chunks:
            return []
        lex = self._minmax(self._bm25(index, query))
        if index.emb is not None and self.has_model:
            q = self._center_norm(self._embed([query[:1000]]), index.mu)[0]
            score = 0.5 * self._minmax(index.emb @ q) + 0.5 * lex
        else:
            score = lex
        top = np.argsort(-score)[:top_k]
        return [(int(i), float(score[i]), index.chunks[i]) for i in top if score[i] > 0]

    def chat_with_document(self, prompt: str, chat_history: List[Dict], index: DocIndex, top_k: int = 4) -> Dict:
        """Returns {"answer": str, "sources": [(section_no, text), ...]}."""
        if index is None or not index.chunks:
            return {"answer": "⚠️ No contract loaded. Upload and analyse a contract first.", "sources": []}

        # Short follow-ups ("and the penalty?") need the previous question for retrieval
        query = prompt
        if len(prompt.split()) <= 6:
            prev = [m["content"] for m in chat_history[:-1] if m["role"] == "user"]
            if prev:
                query = f"{prev[-1]} {prompt}"

        hits = self.semantic_search_chunks(index, query, top_k=top_k)
        if not hits:
            return {
                "answer": "I couldn't find a part of the contract that relates to that. "
                "Try rephrasing with terms used in the contract.",
                "sources": [],
            }
        passages = [(i, t) for i, _, t in hits]

        if self.gpt.available:
            # history already contains the current question as the last item; exclude it
            ans = self.gpt.answer(prompt, passages, chat_history[:-1])
        else:
            ans = (
                "The AI model is not configured, so here are the most relevant "
                "contract sections instead."
            )
        return {"answer": ans, "sources": [(i + 1, t) for i, t in passages]}
