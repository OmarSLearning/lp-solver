"""
Agent LLM — extraction du modèle de programme linéaire via Ollama (mistral).

Deux modes supportés :
  - "text"  : l'utilisateur décrit le problème en langage naturel
  - "lp"    : l'utilisateur colle directement un bloc LP formaté
              (min/max ... subject to ... bounds)

Le LLM retourne toujours un JSON canonique validé avant transmission au solveur.
"""

import json
import re
import requests
from typing import Literal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

# Schéma JSON attendu en sortie du LLM
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
# Fonctions principales
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, system: str) -> str:
    """Appel brut à l'API Ollama. Retourne le texte généré."""
    payload = {
        "model": MODEL,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,   # déterministe pour l'extraction structurée
            "num_predict": 1024,
        },
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Impossible de joindre Ollama. "
            "Vérifiez qu'il tourne avec : `ollama serve`"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama n'a pas répondu dans les 120 secondes.")


def _extract_json(raw: str) -> dict:
    """
    Extrait et parse le JSON depuis la réponse du LLM.
    Robuste aux balises markdown résiduelles (```json ... ```).
    """
    # Supprimer les balises markdown si présentes
    clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    # Tenter un parse direct
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Chercher le premier objet JSON dans le texte
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(
            f"Le LLM n'a pas retourné un JSON valide.\n"
            f"Réponse brute :\n{raw[:500]}"
        )


def _validate_lp_model(model: dict) -> dict:
    """Validation minimale du schéma LP et normalisation."""
    # Vérification des clés obligatoires
    for key in ("objective", "constraints", "variables"):
        if key not in model:
            raise ValueError(f"Clé manquante dans le modèle LP : '{key}'")

    obj = model["objective"]
    if obj.get("type") not in ("minimize", "maximize"):
        raise ValueError(
            f"objective.type invalide : '{obj.get('type')}'. "
            "Attendu : 'minimize' ou 'maximize'."
        )
    if not obj.get("coefficients"):
        raise ValueError("objective.coefficients est vide ou absent.")

    for i, c in enumerate(model["constraints"]):
        for key in ("coefficients", "sense", "rhs"):
            if key not in c:
                raise ValueError(
                    f"Contrainte {i} : clé manquante '{key}'."
                )
        if c["sense"] not in ("<=", ">=", "="):
            raise ValueError(
                f"Contrainte {i} : sense invalide '{c['sense']}'. "
                "Attendu : '<=', '>=' ou '='."
            )

    # S'assurer que toutes les variables citées existent dans "variables"
    all_vars = set()
    for coef_dict in [obj["coefficients"]] + [c["coefficients"] for c in model["constraints"]]:
        all_vars.update(coef_dict.keys())

    for v in all_vars:
        if v not in model["variables"]:
            model["variables"][v] = {"lb": 0, "ub": None}

    return model


def extract_lp_from_text(user_text: str) -> dict:
    """
    Mode 'text' : extrait un modèle LP depuis une description en langage naturel.
    Retourne un dict validé conforme au schéma LP.
    """
    raw = _call_ollama(prompt=user_text, system=SYSTEM_PROMPT_TEXT)
    model = _extract_json(raw)
    return _validate_lp_model(model)


def extract_lp_from_lp_format(lp_text: str) -> dict:
    """
    Mode 'lp' : parse un bloc LP formaté (style CPLEX/GLPK).
    Retourne un dict validé conforme au schéma LP.
    """
    raw = _call_ollama(prompt=lp_text, system=SYSTEM_PROMPT_LP)
    model = _extract_json(raw)
    return _validate_lp_model(model)
