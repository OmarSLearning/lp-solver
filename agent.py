"""
Agent LLM — extraction du modèle MIP/LP via OpenRouter.

Supporte :
  - Variables continues (LP)
  - Variables entières (ILP)
  - Variables binaires (BLP)
  - Mixte continu + entier + binaire (MIP)
  - Problèmes apparemment non linéaires linéarisables :
      * Produit de variables liées (sélection de scénarios)
      * Valeur absolue
      * Max/min d'expressions
      * Fonctions affines par morceaux
      * Produit variable continue × binaire (McCormick)

Compatible Streamlit Cloud — clé API dans st.secrets["OPENROUTER_API_KEY"].
"""

import json
import re
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "meta-llama/llama-3.3-70b-instruct"
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
    "<var_name>": {
      "lb": <float|null>,
      "ub": <float|null>,
      "type": "continuous" | "integer" | "binary"
    }
  }
}
"""

SYSTEM_PROMPT_TEXT = f"""Tu es un expert en optimisation linéaire, MIP et en linéarisation de problèmes apparemment non linéaires.

L'utilisateur te décrit un problème d'optimisation en langage naturel.
Tu dois extraire un programme linéaire/MIP VALIDE et retourner UNIQUEMENT un objet JSON,
sans aucun texte avant ou après, sans balises markdown, sans commentaires.

Le JSON doit respecter strictement ce schéma :
{LP_SCHEMA}

════════════════════════════════════════════════════════
ÉTAPE 0 — DÉTECTION ET LINÉARISATION (OBLIGATOIRE)
════════════════════════════════════════════════════════

Avant toute modélisation, analyse si le problème contient des non-linéarités :

① PRODUIT DE VARIABLES (ex: revenu = prix × quantité)
   → Technique : sélection de scénarios discrets par variables binaires.
   
   Exemple canonique : prix p et quantité q sont liés par une relation affine
   p = p0 - α·k  et  q = q0 + β·k  avec k ∈ {{0,1,...,K}}
   
   Le produit p·q devient non linéaire. Linéarisation :
   - Énumérer tous les scénarios k = 0, 1, ..., K_max
   - Pour chaque k, calculer le profit scalaire π_k = (p_k - coût_unitaire)·q_k - coût_fixe
   - Introduire une variable binaire y_k pour chaque scénario
   - Maximiser Σ π_k · y_k
   - Contrainte : Σ y_k = 1 (exactement un scénario choisi)
   - K_max : borne naturelle (ex: prix_k ≥ coût_unitaire, ou capacité max).
     Si non précisé, déduire depuis le contexte.
   
   Variables auxiliaires de reporting (si l'énoncé demande prix/quantité) :
   - p = Σ p_k · y_k  (variable continue)
   - q = Σ q_k · y_k  (variable continue)
   Ajouter les contraintes d'égalité correspondantes dans "constraints".

② VALEUR ABSOLUE  |x|
   → Introduire z ≥ x  et  z ≥ -x,  minimiser z.

③ MAX/MIN D'EXPRESSIONS  max(a, b)
   → Variable auxiliaire z ≥ a,  z ≥ b.

④ FONCTION AFFINE PAR MORCEAUX
   → Décomposition en segments avec variables binaires (SOS2 ou Big-M).

⑤ PRODUIT VARIABLE CONTINUE × BINAIRE  x · y
   → Linéarisation de McCormick : w = x·y
     w ≤ U·y,   w ≥ L·y,   w ≤ x - L(1-y),   w ≥ x - U(1-y)

════════════════════════════════════════════════════════
ÉTAPE 1 — CONTRAINTES DE PROPORTION (OBLIGATOIRE)
════════════════════════════════════════════════════════
Quand un énoncé exprime une proportion/pourcentage d'un TOTAL, la variable
du numérateur doit être comparée au TOTAL, pas au reste.

ERREUR FRÉQUENTE À ÉVITER (les deux cas) :
  ✗  x ≤ P × y          (P% du complément — FAUX)
  ✓  x ≤ P × (x + y)    (P% du total     — CORRECT)

────────────────────────────────────────────────────────
CAS 1 — "au plus P% du total" (at most / no more than)
────────────────────────────────────────────────────────
  x ≤ P × (x + y + z + ...)
  (1-P)·x - P·y - P·z - ... ≤ 0
→ Identifier explicitement quelle variable est celle mentionnée dans la proportion
  → Cette variable concernée reçoit (1-P)
  → Toutes les autres variables reçoivent -P
  → VÉRIFICATION : la variable concernée a-t-elle le coefficient (1-P) ? Si non, corriger.
  → coefficients : {{x: (1-P), y: -P, z: -P, ...}}, sense: "<=", rhs: 0

────────────────────────────────────────────────────────
CAS 2 — "au moins P% du total" (at least / minimum)
────────────────────────────────────────────────────────
  x ≥ P × (x + y + z + ...)
  (1-P)·x - P·y - P·z - ... ≥ 0
→ Identifier explicitement quelle variable est celle mentionnée dans la proportion
  → Cette variable concernée reçoit (1-P)
  → Toutes les autres variables reçoivent -P
  → VÉRIFICATION : la variable concernée a-t-elle le coefficient (1-P) ? Si non, corriger.
  → coefficients : {{x: -(1-P), y: P, z: P, ...}}, sense: "<=", rhs: 0

────────────────────────────────────────────────────────
Formulations déclenchant cette règle :
────────────────────────────────────────────────────────
  - "at most X% of [total] can be [type]"       → CAS 1, sense "<="
  - "no more than X% of all [items] are [type]" → CAS 1, sense "<="
  - "[type] cannot exceed X% of [total]"        → CAS 1, sense "<="
  - "at least X% must be [type]"                → CAS 2, sense ">="
  - "minimum X% of [total] should be [type]"    → CAS 2, sense ">="

════════════════════════════════════════════════════════
RÈGLES GÉNÉRALES DU SCHÉMA JSON
════════════════════════════════════════════════════════

- Toutes les variables (y compris les y_k binaires) apparaissent dans "variables".
- "type" obligatoire : "continuous" | "integer" | "binary".
- lb=0 par défaut. ub=null si non borné supérieurement.
- Pour les binaires : lb=0, ub=1 automatiquement.
- "sense" ∈ {{"<=", ">=", "="}}.
- Les coefficients π_k dans l'objectif sont des scalaires PRÉ-CALCULÉS (float).
- Noms de variables : identifiants sans espaces (y_0, y_1, ...).
- Contraintes d'égalité pour variables auxiliaires incluses dans "constraints".
- "label" dans chaque contrainte est recommandé pour la lisibilité.
- Pour toute contrainte de proportion, les coefficients doivent être ceux issus
  du développement algébrique de x ≤ P×(x+y+...) pour une quantité au plus et x >= P×(x+y+...) pour une quantité au moins,
  jamais P appliqué à une seule variable isolée. La variable concernée reçoit
  toujours (1-P), toutes les autres reçoivent -P. Vérifier systématiquement avant d'écrire le JSON.
"""

SYSTEM_PROMPT_LP = f"""Tu es un expert en optimisation linéaire et MIP.
L'utilisateur te fournit un programme linéaire/MIP en format texte (style CPLEX LP ou similaire).
Tu dois le parser et retourner UNIQUEMENT un objet JSON valide,
sans aucun texte avant ou après, sans balises markdown, sans commentaires.

Le JSON doit respecter strictement ce schéma :
{LP_SCHEMA}

Règles :
- Interprète "min"/"minimize" → "minimize", "max"/"maximize" → "maximize".
- Section "general" ou "int" dans le LP → type "integer".
- Section "binary" ou "bin" dans le LP → type "binary".
- Variables non listées dans ces sections → type "continuous".
- Pour les variables binaires : lb=0, ub=1.
- lb=0 par défaut si non précisé. ub=null si non borné supérieurement.
- Si l'objectif contient un produit de variables, applique la même logique de
  linéarisation par scénarios que dans le prompt texte avant de générer le JSON.
"""

# ---------------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------------

def _call_openrouter(prompt: str, system: str) -> str:
    headers = {
        "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://lp-solver-souissi.streamlit.app",
        "X-OpenRouter-Title": "LP Solver",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 4096,   # augmenté : les scénarios énumérés peuvent être longs
        "provider": {
            "sort": "throughput",
        },
    }
    try:
        response = requests.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=120
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        body = e.response.text[:300] if e.response is not None else ""
        raise ValueError(f"OpenRouter a retourné une erreur HTTP {status_code}.\n{body}")
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
    """Validation et normalisation du schéma MIP/LP."""
    for key in ("objective", "constraints", "variables"):
        if key not in model:
            raise ValueError(f"Clé manquante dans le modèle : '{key}'")

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

    # Auto-complétion des variables manquantes
    all_vars = set()
    for coef_dict in [obj["coefficients"]] + [c["coefficients"] for c in model["constraints"]]:
        all_vars.update(coef_dict.keys())

    for v in all_vars:
        if v not in model["variables"]:
            model["variables"][v] = {"lb": 0, "ub": None, "type": "continuous"}

    # Normalisation des types et bornes
    VALID_TYPES = ("continuous", "integer", "binary")
    for name, cfg in model["variables"].items():
        if "type" not in cfg:
            cfg["type"] = "continuous"
        if cfg["type"] not in VALID_TYPES:
            cfg["type"] = "continuous"
        if cfg["type"] == "binary":
            cfg["lb"] = 0
            cfg["ub"] = 1

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
