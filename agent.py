"""
Agent LLM — extraction du modèle de programme linéaire via OpenRouter (DeepSeek-R1).

Deux modes supportés :
  - "text"  : l'utilisateur décrit le problème en langage naturel
  - "lp"    : l'utilisateur colle directement un bloc LP formaté

Compatible Streamlit Cloud — utilise l'API OpenRouter (pas d'Ollama local requis).
La clé API est lue depuis st.secrets["OPENROUTER_API_KEY"].
"""

import json
import re
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "llama-3.3-70b-instruct:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

LP_SCHEMA = """
{
  "objective": {
    "type": "minimize" | "maximize",
    "coefficients": {"<var_name>": <float>, ...}
  },
  "constraints": [
    {
      "label": "<optionnel>",
      "coefficients": {"<var_name>": <float>, ...},
      "sense": "<=" | ">=" | "=",
      "rhs": <float>
    }
  ],
  "variables": {
    "<var_name>": {"lb": <float|null>, "ub": <float|null>}
  }
}
"""

SYSTEM_PROMPT_TEXT = f"""Tu es un expert en optimisation linéaire (recherche opérationnelle).
L'utilisateur te décrit un problème d'optimisation en langage naturel.
Tu dois extraire le programme linéaire correspondant et retourner UNIQUEMENT un objet JSON valide,
sans aucun texte avant ou après, sans balises markdown, sans commentaires.

Le JSON doit respecter strictement ce schéma :
{LP_SCHEMA}

Règles importantes :
- Toutes les variables doivent apparaître dans la section "variables".
- lb=0 par défaut si non précisé (non-négativité standard).
- ub=null si pas de borne supérieure.
- Les coefficients absents dans une contrainte valent 0 (ne les inclure que s'ils sont non nuls).
- "sense" doit être exactement "<=" , ">=" ou "=".
"""

SYSTEM_PROMPT_LP = f"""Tu es un expert en optimisation linéaire.
L'utilisateur te fournit un programme linéaire en format texte (style CPLEX LP ou similaire).
Tu dois le parser et retourner UNIQUEMENT un objet JSON valide,
sans aucun texte avant ou après, sans balises markdown, sans commentaires.

Le JSON doit respecter strictement ce schéma :
{LP_SCHEMA}

Règles :
- Interprète "min" / "minimize" → "minimize", "max" / "maximize" → "maximize".
- Extrais toutes les variables, contraintes et bornes.
- lb=0 par défaut si non précisé. ub=null si non borné supérieurement.
"""

# ---------------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------------

def _call_openrouter(prompt: str, system: str) -> str:
    """
    Appel à l'API OpenRouter (compatible OpenAI).
    DeepSeek-R1 produit un bloc <think>...</think> avant la réponse — on le supprime.
    """
    headers = {
        "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://lp-solver-souissi.streamlit.app",
        "X-Title": "LP Solver",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    try:
        response = requests.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=120
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        # Supprimer le raisonnement interne de DeepSeek-R1
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        body = e.response.text[:300] if e.response is not None else ""
        raise ValueError(
            f"OpenRouter a retourné une erreur HTTP {status_code}.\n{body}"
        )
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Impossible de joindre OpenRouter. Vérifiez votre connexion.")
    except requests.exceptions.Timeout:
        raise TimeoutError("OpenRouter n'a pas répondu dans les 120 secondes.")
    except (KeyError, IndexError):
        raise ValueError("Réponse inattendue de l'API OpenRouter.")


def _extract_json(raw: str) -> dict:
    clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(
            f"Le LLM n'a pas retourné un JSON valide.\nRéponse brute :\n{raw[:500]}"
        )


def _validate_lp_model(model: dict) -> dict:
    for key in ("objective", "constraints", "variables"):
        if key not in model:
            raise ValueError(f"Clé manquante dans le modèle LP : '{key}'")

    obj = model["objective"]
    if obj.get("type") not in ("minimize", "maximize"):
        raise ValueError(f"objective.type invalide : '{obj.get('type')}'.")
    if not obj.get("coefficients"):
        raise ValueError("objective.coefficients est vide ou absent.")

    for i, c in enumerate(model["constraints"]):
        for key in ("coefficients", "sense", "rhs"):
            if key not in c:
                raise ValueError(f"Contrainte {i} : clé manquante '{key}'.")
        if c["sense"] not in ("<=", ">=", "="):
            raise ValueError(f"Contrainte {i} : sense invalide '{c['sense']}'.")

    all_vars = set()
    for coef_dict in [obj["coefficients"]] + [c["coefficients"] for c in model["constraints"]]:
        all_vars.update(coef_dict.keys())
    for v in all_vars:
        if v not in model["variables"]:
            model["variables"][v] = {"lb": 0, "ub": None}

    return model


# ---------------------------------------------------------------------------
# Fonctions publiques
# ---------------------------------------------------------------------------

def extract_lp_from_text(user_text: str) -> dict:
    raw = _call_openrouter(prompt=user_text, system=SYSTEM_PROMPT_TEXT)
    model = _extract_json(raw)
    return _validate_lp_model(model)


def extract_lp_from_lp_format(lp_text: str) -> dict:
    raw = _call_openrouter(prompt=lp_text, system=SYSTEM_PROMPT_LP)
    model = _extract_json(raw)
    return _validate_lp_model(model)
