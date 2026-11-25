"""Legal Buddy - Streamlit app.

InLegalBERT: clause classification + retrieval.  Gemini/OpenAI LLM: plain-language
explanations, risk-analysed summary and grounded follow-up chat.
Run:  streamlit run app.py
"""
import html
import os
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine import ContractAnalyzer, GPTExplainer, load_inlegalbert

try:  # load GEMINI_API_KEY / OPENAI_API_KEY etc. from a local .env file if present
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def backend_secret(name: str, default: str = "") -> str:
    """Read config from the server: environment / .env first, then .streamlit/secrets.toml."""
    val = os.getenv(name)
    if val:
        return val
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Legal Buddy",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
    :root {
        --bg: #0F1117; --panel: #171A23; --card: #1D2130; --line: #2C3245;
        --accent: #818CF8; --sky: #38BDF8; --text: #E7E9F2; --muted: #8D93AB;
        --high: #F87171; --med: #FBBF24; --low: #34D399;
    }
    html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
    .stApp { background: radial-gradient(1200px 600px at 10% -10%, #1c2142 0%, var(--bg) 55%); }
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none !important; }
    .block-container { padding-top: 1rem !important; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
    h1, h2, h3 { letter-spacing: .2px; }
    h2 { border-bottom: 1px solid var(--line); padding-bottom: .4rem; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#141829 0%,#0F1117 100%); border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h2 { color: var(--accent); border: none; }

    .main-header {
        background: linear-gradient(120deg, #1e2447 0%, #252B52 50%, #2b2a63 100%);
        border: 1px solid #38406b; border-bottom: 3px solid var(--accent);
        padding: 2.2rem 1.2rem 2rem; border-radius: 14px; min-height: 11.1rem;
        display: flex; flex-direction: column; justify-content: center; text-align: center; margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(0,0,0,.35);
    }
    .main-header h1 { font-family: 'Inter', sans-serif; font-weight: 700; color: #fff; margin: 0; padding: 0;
        font-size: clamp(1.78rem, 4.46vw, 2.93rem); line-height: 1.1; letter-spacing: -1px; }
    .main-header p  { color: #c3c9e6; margin: .7rem 0 0; font-size: clamp(1.1rem, 2.2vw, 1.5rem); }
    .main-header .accent { color: var(--accent); }

    .risk-high, .risk-medium, .risk-low {
        background: var(--card); border: 1px solid var(--line); border-left: 5px solid;
        padding: .95rem 1.1rem; margin: .6rem 0; border-radius: 10px; color: var(--text);
        box-shadow: 0 4px 14px rgba(0,0,0,.25);
    }
    .risk-high   { border-left-color: var(--high); background: linear-gradient(90deg, rgba(248,113,113,.13), var(--card) 40%); }
    .risk-medium { border-left-color: var(--med);  background: linear-gradient(90deg, rgba(251,191,36,.12), var(--card) 40%); }
    .risk-low    { border-left-color: var(--low);  background: linear-gradient(90deg, rgba(52,211,153,.12), var(--card) 40%); }

    .intro { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
        padding: 1.1rem 1.3rem; margin: -.4rem 0 1.4rem; color: var(--text); line-height: 1.65; }
    .intro p { margin: 0 0 .8rem; text-align: left; font-size: 1.05rem; }
    .intro p.welcome { text-align: center; font-size: 1.785rem; font-weight: 600; margin-bottom: .7rem; }
    .intro .steps { display: flex; flex-wrap: wrap; gap: .6rem; }
    .intro .step { flex: 1 1 200px; background: var(--card); border: 1px solid var(--line);
        border-radius: 10px; padding: .6rem .8rem; font-size: .966rem; color: var(--muted); }
    .intro .step b { color: var(--accent); }
    [data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; }
    [data-testid="stExpander"] summary { font-weight: 600; }
    [data-testid="stTabs"] button[role="tab"] { font-weight: 600; color: var(--muted); }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent); }
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, #818CF8, #6366F1); color: #0F1117; font-weight: 700;
        border: none; border-radius: 10px; transition: transform .1s, box-shadow .2s;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(129,140,248,.4); color: #0F1117; }
    [data-testid="stChatMessage"] { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; }
    [data-testid="stFileUploaderDropzone"] { background: var(--panel); border: 1.5px dashed var(--accent); border-radius: 12px; }
    [data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: .8rem 1rem; }
    [data-testid="stMetricValue"] { color: var(--accent); }
    textarea, input { border-radius: 8px !important; }
    /* footer pinned to the bottom of the screen; page gets extra bottom padding so nothing hides behind it */
    .block-container, [data-testid="stMainBlockContainer"] { padding-bottom: 7rem !important; }
    .site-footer { position: fixed; left: 0; right: 0; bottom: 0; z-index: 999; margin: 0;
        padding: .8rem 0 .9rem; text-align: center; background: var(--bg);
        border-top: 1px solid var(--line); pointer-events: none; }
    .site-footer a { pointer-events: auto; }
    .site-footer a { display: inline-flex; align-items: center; gap: .7rem; text-decoration: none;
        background: var(--panel); border: 1px solid var(--line); border-radius: 999px;
        padding: .6rem 1.5rem; color: var(--muted); font-size: 1rem; transition: border-color .15s, color .15s; }
    .site-footer a:hover { border-color: var(--accent); color: var(--text); }
    .site-footer .dot { opacity: .7; }
    .site-footer svg { width: 30px; height: 30px; fill: currentColor; display: block; }
    [data-testid="stRadio"] [role="radiogroup"] { justify-content: center; gap: .9rem; margin-bottom: .6rem; }
    [data-testid="stRadio"] label { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
        padding: .55rem 1.4rem; cursor: pointer; transition: border-color .15s, background .15s; }
    [data-testid="stRadio"] label:hover { border-color: var(--accent); }
    [data-testid="stRadio"] label:has(input:checked) { border-color: var(--accent); background: rgba(129,140,248,.16); }
    [data-testid="stRadio"] label p { font-size: 1.3rem !important; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

# Shared dark chart styling
CHART_FONT = dict(family="Inter, sans-serif", color="#E7E9F2")
RATING_COLORS = {"Excellent": "#34D399", "Good": "#38BDF8", "Fair": "#FBBF24", "Poor": "#F87171"}


def style_chart(fig, **kw):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=CHART_FONT,
        margin=dict(l=20, r=20, t=60, b=20), **kw,
    )
    fig.update_xaxes(gridcolor="#2C3245", zerolinecolor="#2C3245")
    fig.update_yaxes(gridcolor="#2C3245", zerolinecolor="#2C3245")
    return fig


RISK_ICON = {"high": "🔴", "medium": "🟠", "low": "🟢"}


# ---------------------------------------------------------------- setup ----
@st.cache_resource(show_spinner="Loading InLegalBERT (first run downloads the model)…")
def get_models():
    return load_inlegalbert()


def get_analyzer() -> ContractAnalyzer:
    tok, model = get_models()
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = ContractAnalyzer(tok, model, GPTExplainer.from_env(backend_secret))
    return st.session_state.analyzer


def run_analysis(uploaded_file):
    """Full pipeline for one upload; results are cached in session_state."""
    an = st.session_state.analyzer
    with st.status("Analysing contract…", expanded=True) as status:
        st.write("📖 Extracting text")
        text = an.extract_text_from_pdf(uploaded_file)
        if not text:
            status.update(label="Could not read the PDF", state="error")
            st.error(
                "No text could be extracted. If this is a scanned PDF, run OCR on it first "
                "(e.g. `ocrmypdf in.pdf out.pdf`) and upload the result."
            )
            return
        st.write("🧩 Splitting into clauses and embedding with InLegalBERT")
        index = an.build_index(text)
        st.write("🏷️ Classifying clauses")
        contract_type = an.identify_contract_type(text)
        clauses = an.analyze_legal_clauses(text, index)
        n = 0
        summary = None
        if an.gpt.available:
            st.write("✨ Generating plain-language explanations (AI)")
            n = an.enrich_with_gpt(clauses, contract_type)
            st.write("🧾 Writing risk summary (AI)")
            summary = an.risk_summary(clauses, contract_type, text)
        st.session_state.update(
            contract_text=text, index=index, contract_type=contract_type, clauses=clauses,
            summary=summary, gpt_enriched=n, chat_history=[],
            contract_filename=uploaded_file.name,
        )
        status.update(label=f"Done - {len(clauses)} clauses found", state="complete", expanded=False)


# ------------------------------------------------------------------- tabs ---
def show_summary_tab():
    st.header("🧾 Risk-Analysed Summary")
    summary = st.session_state.get("summary")
    clauses = st.session_state.clauses
    if not summary:
        if not st.session_state.analyzer.gpt.available:
            st.info("The AI-written summary isn't available right now. "
                    "Rule-based results are shown in the other tabs.")
        else:
            print(f"[app] summary failed: {st.session_state.analyzer.gpt.last_error}")  # server log only
            st.warning("The AI summary could not be generated right now. Rule-based results are in the other tabs.")
        return
    risk = summary.get("overall_risk", "medium")
    st.markdown(
        f'<div class="risk-{risk}"><strong>{html.escape(summary.get("headline", ""))}</strong><br>'
        f'Overall risk: <strong>{risk.upper()}</strong></div>',
        unsafe_allow_html=True,
    )
    st.write(summary.get("summary", ""))
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Key terms")
        for t in summary.get("key_terms", []):
            st.markdown(f"- {t}")
    with c2:
        st.subheader("Top risks")
        for r in summary.get("top_risks", []):
            ref = f" _(Section {r['clause_id'] + 1})_" if isinstance(r.get("clause_id"), int) else ""
            st.markdown(f"- **{r.get('title','')}**{ref}: {r.get('why','')}")


def show_plain_language_analysis(clauses, contract_type):
    st.header("🔍 Plain Language Explanations")
    if not clauses:
        st.warning("No specific legal clauses detected in the contract.")
        return
    st.info(f"📋 **Contract type detected:** {contract_type.title()}  ·  "
            f"{len(clauses)} clauses classified"
            + ("" if st.session_state.analyzer.has_model else "  ·  ⚠️ InLegalBERT not loaded - keyword mode"))

    c1, c2 = st.columns(2)
    risk_filter = c1.multiselect("Risk level", ["high", "medium", "low"], default=["high", "medium"])
    types = sorted({c["type"] for c in clauses})
    type_filter = c2.multiselect("Clause type", types, default=[])
    explanations = st.session_state.analyzer.generate_plain_language_explanation(clauses, contract_type)

    shown = 0
    order = {"high": 0, "medium": 1, "low": 2}
    for c, ex in sorted(zip(clauses, explanations), key=lambda p: order[p[1]["risk_level"]]):
        if c["risk_level"] not in risk_filter or (type_filter and c["type"] not in type_filter):
            continue
        shown += 1
        label = (f"{RISK_ICON[c['risk_level']]} Section {c['id'] + 1} · "
                 f"{c['type'].replace('_', ' ').title()}  (conf. {c['confidence']:.0%})")
        with st.expander(label, expanded=c["risk_level"] == "high"):
            l, r = st.columns(2)
            l.markdown("**📜 Original text**")
            l.write(ex["original"])
            r.markdown("**✨ Plain language**" + ("" if ex["gpt"] else " _(rule-based)_"))
            r.write(ex["plain_english"])
            extra = ""
            if ex.get("risk_reason"):
                extra += f"<br><strong>Why:</strong> {html.escape(ex['risk_reason'])}"
            if ex.get("negotiation_tip"):
                extra += f"<br><strong>Tip:</strong> {html.escape(ex['negotiation_tip'])}"
            st.markdown(
                f'<div class="risk-{c["risk_level"]}"><strong>What this means:</strong> '
                f'{html.escape(ex["practical_meaning"])}<br><strong>Risk:</strong> '
                f'{c["risk_level"].upper()}{extra}</div>',
                unsafe_allow_html=True,
            )
            nums = {k: v for k, v in (c.get("meta") or {}).items() if v}
            if nums:
                st.caption("Detected: " + ", ".join(f"{k}: {v}" for k, v in nums.items()))
    if not shown:
        st.info("No clauses match the filters.")


def show_red_flag_analysis(clauses, contract_type):
    """Show red flag detection"""
    st.header("🚨 Red Flag Detection")

    # Pass full contract text for comprehensive red flag detection
    full_text = st.session_state.get("contract_text", "")
    red_flags = st.session_state.analyzer.detect_red_flags(
        clauses, contract_type, full_text=full_text
    )

    if not red_flags:
        st.success("🎉 No major red flags detected in this contract!")
        return

    st.warning(f"⚠️ Found {len(red_flags)} potential red flags:")

    for i, flag in enumerate(red_flags, 1):
        with st.expander(f"🚨 Red Flag {i}: {flag['issue']}", expanded=True):
            st.markdown(
                f"""
            <div class="risk-high">
                <strong>Issue:</strong> {flag["issue"]}<br>
                <strong>Severity:</strong> {flag["severity"]}<br>
                <strong>Recommendation:</strong> {flag["recommendation"]}
            </div>
            """,
                unsafe_allow_html=True,
            )

            if flag.get("clause"):
                st.text_area(
                    "Problematic Clause:",
                    flag["clause"],
                    height=100,
                    key=f"redflag_{i}",
                    label_visibility="visible",
                )


def show_checklist(clauses, contract_type):
    """Show personalized checklist"""
    st.header("✅ Before You Sign Checklist")

    checklist = st.session_state.analyzer.generate_checklist(contract_type, clauses)

    st.info("📝 **Important questions to ask before signing:**")

    for i, item in enumerate(checklist, 1):
        checked = st.checkbox(f"{item}", key=f"checklist_{i}")
        if checked:
            st.success("✅ Completed")


def show_confidence_score(clauses):
    """Show confidence score and metrics"""
    st.header("📊 Contract Confidence Score")

    full_text = st.session_state.get("contract_text", "")
    red_flags = st.session_state.analyzer.detect_red_flags(
        clauses, st.session_state.contract_type, full_text=full_text
    )
    confidence = st.session_state.analyzer.calculate_confidence_score(
        clauses, red_flags
    )

    # Score visualization
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=confidence["score"],
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Contract Safety Score"},
            delta={"reference": 80},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": RATING_COLORS.get(confidence["rating"], "#818CF8"), "thickness": 0.3},
                "steps": [
                    {"range": [0, 40], "color": "rgba(248,113,113,.22)"},
                    {"range": [40, 60], "color": "rgba(251,191,36,.20)"},
                    {"range": [60, 80], "color": "rgba(56,189,248,.18)"},
                    {"range": [80, 100], "color": "rgba(52,211,153,.22)"},
                ],
                "threshold": {
                    "line": {"color": "#818CF8", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )

    style_chart(fig, height=300)
    st.plotly_chart(fig, width="stretch")

    # Score details
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Rating", confidence["rating"])

    with col2:
        st.metric("Total Clauses", confidence["details"]["total_clauses"])

    with col3:
        st.metric("High Risk", confidence["details"]["high_risk"])

    with col4:
        st.metric("Red Flags", confidence["details"]["red_flags"])

    # Recommendations
    st.subheader("📋 Recommendations")

    if confidence["score"] >= 80:
        st.success("✅ This contract appears to be fair and well-structured.")
    elif confidence["score"] >= 60:
        st.warning(
            "⚠️ This contract has some areas of concern. Review the red flags carefully."
        )
    else:
        st.error(
            "❌ This contract has significant issues. Consider legal review before signing."
        )


def show_alternatives(clauses, contract_type):
    """Show alternative clause suggestions"""
    st.header("💡 Suggested Alternatives")

    full_text = st.session_state.get("contract_text", "")
    red_flags = st.session_state.analyzer.detect_red_flags(
        clauses, contract_type, full_text=full_text
    )

    if not red_flags:
        st.success("🎉 No alternatives needed - your contract looks good!")
        return

    alternatives = st.session_state.analyzer.suggest_alternatives(
        red_flags, contract_type
    )

    st.info("💡 **Suggested improvements for problematic clauses:**")

    for i, alt in enumerate(alternatives, 1):
        with st.expander(f"Alternative {i}: {alt['original_issue']}", expanded=True):
            st.subheader("⚠️ Current Issue")
            st.error(alt["original_issue"])

            st.subheader("💡 Suggested Improvement")
            st.success(alt["suggested_wording"])

            st.subheader("📝 Why This Is Better")
            st.info(alt["explanation"])


def show_full_analysis():
    st.header("📄 Upload Your Contract")
    uploaded = st.file_uploader("Choose a PDF file", type="pdf", help="Text-based PDF of an Indian contract")
    if uploaded is not None:
        key = (uploaded.name, uploaded.size)
        if st.session_state.get("upload_key") != key:
            st.session_state.upload_key = key
            run_analysis(uploaded)

    if "clauses" not in st.session_state:
        st.info("Upload a PDF contract to begin analysis.")
        return
    st.caption(f"Analysing: **{st.session_state.contract_filename}**")
    clauses, ctype = st.session_state.clauses, st.session_state.contract_type
    tabs = st.tabs(["🧾 Summary", "🔍 Plain Language", "🚨 Red Flags", "✅ Checklist",
                    "📊 Score", "💡 Alternatives", "💬 Ask Questions"])
    with tabs[0]:
        show_summary_tab()
    with tabs[1]:
        show_plain_language_analysis(clauses, ctype)
    with tabs[2]:
        show_red_flag_analysis(clauses, ctype)
    with tabs[3]:
        show_checklist(clauses, ctype)
    with tabs[4]:
        show_confidence_score(clauses)
    with tabs[5]:
        show_alternatives(clauses, ctype)
    with tabs[6]:
        show_chat_interface()


SUGGESTED = [
    "What are the key terms and conditions?",
    "What are the risks or one-sided clauses?",
    "What are the payment terms?",
    "What are the termination conditions and notice period?",
]


def show_chat_interface():
    st.header("💬 Ask Questions About Your Contract")
    st.caption(f"Contract: **{st.session_state.contract_filename}** "
               f"({st.session_state.contract_type.title()})")
    hist = st.session_state.setdefault("chat_history", [])

    def render(m):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("Sources from the contract"):
                    for n, t in m["sources"]:
                        st.markdown(f"**Section {n}:** {t[:400]}{'…' if len(t) > 400 else ''}")

    msgs = st.container()          # messages are drawn here, above the input
    with msgs:
        for m in hist:
            render(m)

    pending = None
    if not hist:
        st.write("Try asking:")
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED):
            if cols[i % 2].button(q, key=f"sugg_{i}"):
                pending = q
    typed = st.chat_input("Ask about a clause, amount, deadline or penalty…")
    question = pending or typed
    if question:
        user_msg = {"role": "user", "content": question}
        hist.append(user_msg)
        with msgs:
            render(user_msg)
            with st.spinner("Searching the contract and drafting an answer…"):
                res = st.session_state.analyzer.chat_with_document(question, hist, st.session_state.index)
            bot_msg = {"role": "assistant", "content": res["answer"], "sources": res["sources"]}
            hist.append(bot_msg)
            render(bot_msg)
    if hist and st.button("Clear chat 🗑️"):
        st.session_state.chat_history = []
        st.rerun()


def show_bulk_analysis():
    """Show bulk analysis interface"""
    st.header("📊 Bulk Contract Analysis")
    st.info("Upload multiple contracts for comparative analysis")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload multiple contracts for comparison",
    )

    if uploaded_files:
        if st.button("Analyze All Contracts"):
            results = []
            progress_bar = st.progress(0)

            for i, file in enumerate(uploaded_files):
                with st.spinner(f"Analyzing {file.name}..."):
                    text = st.session_state.analyzer.extract_text_from_pdf(file)
                    if text:
                        contract_type = (
                            st.session_state.analyzer.identify_contract_type(text)
                        )
                        clauses = st.session_state.analyzer.analyze_legal_clauses(text)
                        red_flags = st.session_state.analyzer.detect_red_flags(
                            clauses, contract_type
                        )
                        confidence = (
                            st.session_state.analyzer.calculate_confidence_score(
                                clauses, red_flags
                            )
                        )

                        results.append(
                            {
                                "filename": file.name,
                                "type": contract_type,
                                "score": confidence["score"],
                                "rating": confidence["rating"],
                                "red_flags": len(red_flags),
                                "clauses": len(clauses),
                            }
                        )

                progress_bar.progress((i + 1) / len(uploaded_files))

            # Display results
            if results:
                df = pd.DataFrame(results)
                st.subheader("📊 Analysis Results")
                st.dataframe(df)

                # Visualization
                fig = px.bar(
                    df,
                    x="filename",
                    y="score",
                    title="Contract Safety Scores Comparison",
                    color="rating",
                    color_discrete_map=RATING_COLORS,
                )
                style_chart(fig, xaxis_tickangle=-45)
                st.plotly_chart(fig, width="stretch")

                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Score", f"{df['score'].mean():.1f}")
                with col2:
                    st.metric("Best Contract", df.loc[df["score"].idxmax(), "filename"])
                with col3:
                    st.metric("Total Red Flags", df["red_flags"].sum())

def main():
    st.markdown(
        '<div class="main-header"><h1>🏛️ Legal <span class="accent">Buddy</span></h1>'
        "<p>Your Indian legal contracts, explained in plain language</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="intro">
<p class="welcome">Welcome to Legal Buddy.</p>
<p>Upload any Indian legal contract: a rental, employment, service or other contract and get a
plain-language explanation of every important clause, the risks to watch out for, and answers to
your follow-up questions.</p>
<div class="steps">
<div class="step"><b>1. Upload</b><br>Add your contract as a PDF.</div>
<div class="step"><b>2. Review</b><br>See a summary, red flags and clause-by-clause explanations.</div>
<div class="step"><b>3. Ask</b><br>Chat with your contract about anything unclear.</div>
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    get_analyzer()

    mode = st.radio(
        "Mode", ["📄 Full Contract Analysis", "📊 Bulk Analysis"],
        horizontal=True, label_visibility="collapsed",
    )
    if mode.startswith("📄"):
        show_full_analysis()
    else:
        show_bulk_analysis()

    show_footer()


GITHUB_URL = "https://github.com/00dsagar00"
GITHUB_ICON = (
    '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 '
    "5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13"
    "-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2"
    "-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 "
    "2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 "
    "3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 "
    '0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'
)


def show_footer():
    st.markdown(
        f'''<div class="site-footer"><a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer"
        title="Divya Sagar on GitHub"><span>Created by Divya Sagar</span><span class="dot">·</span>{GITHUB_ICON}</a></div>''',
        unsafe_allow_html=True,
    )


main()
