models = {
    "GPT models": [
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini-2025-04-14",
        "gpt-5-2025-08-07",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano",
        "gpt-5.4",
        "o4-mini",
        "gpt-4o-mini-search-preview",
        "gpt-4o-search-preview",
    ],
    "Gemini models": [
        "gemini-3-flash-preview",   
        "gemini-2.5-flash-lite", 
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-pro-preview",
    ],
}

gpt_default = "gpt-4.1-mini-2025-04-14"

REASONING_MODELS_MIN_VERSION = 5  # gpt-5 and above support the responses endpoint
VALID_REASONING_EFFORTS = ("low", "medium", "high")
