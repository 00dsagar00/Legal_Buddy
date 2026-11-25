# Legal Buddy

InLegalBERT (clause classification + retrieval) + an LLM (Gemini or OpenAI) for plain-language explanations,
risk-analysed summary and grounded follow-up chat, served through Streamlit.

## Run
    pip install -r requirements.txt
    cp .env.example .env                 # then put your GEMINI_API_KEY (or OPENAI_API_KEY) in .env (never commit it)
    streamlit run app.py
    pytest -q tests                      # offline tests (no model / API key needed)

## Files
- `app.py`    Streamlit UI: Full Contract Analysis (summary, plain language, red flags, checklist, score, alternatives, chat) and Bulk Analysis
- `engine.py` segmentation, InLegalBERT embeddings, hybrid clause classifier, hybrid retrieval, GPT layer
- `rules.py`  your original benchmarks / red-flag / checklist / scoring logic (unchanged apart from a softer score penalty)

## Pipeline
1. PDF -> text (pypdf) -> clause-sized chunks (numbered clauses / paragraphs / sentences)
2. InLegalBERT masked-mean embeddings, mean-centred (raw BERT vectors are very anisotropic)
3. Clause type = keyword evidence + cosine to prototype sentences per type; low-risk numeric/keyword risk rules
4. The LLM explains the top ~18 clauses (batched JSON) and refines their risk level, then writes the summary
5. Chat: BM25 + InLegalBERT hybrid retrieval -> top 4 sections -> the LLM answers only from them, with section citations

Gemini is used automatically when `GEMINI_API_KEY` is set (free key: Google AI Studio); otherwise OpenAI. Set `LLM_PROVIDER=gemini|openai` to force one.
Note: Gemini's free tier may use your prompts to improve Google products, so use dummy contracts on it.
The key is read on the server from `.env`, the environment, or `.streamlit/secrets.toml`; visitors never see or enter it.
Without an API key the app falls back to rule-based explanations and extractive chat.
Without InLegalBERT it falls back to keyword classification and BM25 retrieval.

## Known limits / next steps
- InLegalBERT is a pre-trained encoder, not a fine-tuned classifier: tune `EMB_ONLY_MIN_SIM` / `EMB_ONLY_MIN_MARGIN`
  in `engine.py` on a few labelled contracts, or fine-tune a head on CUAD/LEDGAR-style data.
- Scanned PDFs need OCR first. Informational only, not legal advice.
