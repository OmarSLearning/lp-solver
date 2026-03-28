"""
LP Solver — Interface Streamlit
Deux modes de saisie : texte naturel ou bloc LP formaté.
"""

import json
import streamlit as st
from agent import extract_lp_from_text, extract_lp_from_lp_format
from solver import solve, format_results

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LP Solver — Optimisation linéaire",
    page_icon="📐",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS minimal
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .result-box {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        font-family: monospace;
        white-space: pre-wrap;
    }
    .error-box {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        font-family: monospace;
        white-space: pre-wrap;
    }
    .model-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------

st.title("📐 LP Solver — Optimisation Linéaire")
st.caption(
    "Décrivez votre problème en texte naturel **ou** collez un bloc LP formaté. "
    "L'agent LLM (Mistral via Ollama) extrait le modèle, PuLP le résout."
)

st.divider()

# ---------------------------------------------------------------------------
# Sélection du mode de saisie
# ---------------------------------------------------------------------------

mode = st.radio(
    "Mode de saisie",
    options=["Texte naturel", "Format LP (CPLEX-style)"],
    horizontal=True,
    help=(
        "**Texte naturel** : décrivez le problème en français ou anglais. "
        "**Format LP** : collez directement un bloc LP structuré."
    ),
)

# ---------------------------------------------------------------------------
# Exemples pré-remplis
# ---------------------------------------------------------------------------

EXAMPLE_TEXT = (
    "Une usine produit deux produits A et B. "
    "Chaque unité de A rapporte 5€ et nécessite 2h de machine et 1h de main d'œuvre. "
    "Chaque unité de B rapporte 4€ et nécessite 1h de machine et 2h de main d'œuvre. "
    "On dispose au maximum de 100h de machine et 80h de main d'œuvre par semaine. "
    "Les quantités produites doivent être non négatives. "
    "Maximiser le profit."
)

EXAMPLE_LP = """\
maximize
  obj: 5 x1 + 4 x2

subject to
  machine:       2 x1 +   x2 <= 100
  main_oeuvre:     x1 + 2 x2 <= 80

bounds
  x1 >= 0
  x2 >= 0
"""

col1, col2 = st.columns([3, 1])

with col1:
    if mode == "Texte naturel":
        user_input = st.text_area(
            "Description du problème",
            value="",
            height=200,
            placeholder=EXAMPLE_TEXT,
        )
    else:
        user_input = st.text_area(
            "Bloc LP",
            value="",
            height=200,
            placeholder=EXAMPLE_LP,
            help="Format compatible CPLEX LP / GLPK.",
        )

with col2:
    st.markdown("**Exemple**")
    if mode == "Texte naturel":
        if st.button("📋 Charger l'exemple", use_container_width=True):
            st.session_state["example_loaded"] = EXAMPLE_TEXT
            st.rerun()
    else:
        if st.button("📋 Charger l'exemple", use_container_width=True):
            st.session_state["example_loaded"] = EXAMPLE_LP
            st.rerun()

    st.markdown("---")
    st.markdown("**Modèle LLM**")
    st.code("llama-3.3-70b-versatile", language=None)
    st.markdown("**Solveur**")
    st.code("PuLP / CBC", language=None)

# Injecter l'exemple si demandé
if "example_loaded" in st.session_state:
    user_input = st.session_state.pop("example_loaded")

# ---------------------------------------------------------------------------
# Bouton de résolution
# ---------------------------------------------------------------------------

st.divider()
solve_btn = st.button("🚀 Résoudre", type="primary", use_container_width=False)

if solve_btn:
    if not user_input or not user_input.strip():
        st.warning("Veuillez saisir un problème avant de lancer la résolution.")
    else:
        # --- Étape 1 : Extraction du modèle LP ---
        with st.status("🤖 Extraction du modèle LP via Mistral…", expanded=True) as status:
            try:
                st.write("Appel à Ollama (mistral)…")
                if mode == "Texte naturel":
                    lp_model = extract_lp_from_text(user_input)
                else:
                    lp_model = extract_lp_from_lp_format(user_input)

                st.write("✅ Modèle LP extrait et validé.")
                status.update(label="Modèle LP extrait.", state="complete")

            except ConnectionError as e:
                status.update(label="Erreur de connexion", state="error")
                st.markdown(
                    f'<div class="error-box">❌ {e}</div>',
                    unsafe_allow_html=True,
                )
                st.stop()
            except (ValueError, KeyError) as e:
                status.update(label="Erreur d'extraction", state="error")
                st.markdown(
                    f'<div class="error-box">❌ Extraction échouée :\n{e}</div>',
                    unsafe_allow_html=True,
                )
                st.stop()

        # --- Étape 2 : Affichage du modèle extrait ---
        with st.expander("🔍 Modèle LP extrait (JSON)", expanded=False):
            st.json(lp_model)

        # --- Étape 3 : Résolution PuLP ---
        with st.status("⚙️ Résolution avec PuLP / CBC…", expanded=False) as status:
            try:
                result = solve(lp_model)
                status.update(label="Résolution terminée.", state="complete")
            except Exception as e:
                status.update(label="Erreur de résolution", state="error")
                st.markdown(
                    f'<div class="error-box">❌ Erreur PuLP :\n{e}</div>',
                    unsafe_allow_html=True,
                )
                st.stop()

        # --- Étape 4 : Affichage des résultats ---
        st.subheader("📊 Résultats")

        if result["is_optimal"]:
            col_a, col_b = st.columns(2)

            with col_a:
                st.metric(
                    label=f"Valeur optimale ({result['objective_type']})",
                    value=f"{result['objective_value']:.6g}",
                )
                st.markdown("**Variables de décision**")
                for var, val in result["variables"].items():
                    st.metric(label=var, value=f"{val:.6g}")

            with col_b:
                st.markdown("**Résumé textuel**")
                st.markdown(
                    f'<div class="result-box">{format_results(result)}</div>',
                    unsafe_allow_html=True,
                )

            with st.expander("📄 Formulation LP interne (PuLP)", expanded=False):
                st.markdown(
                    f'<div class="model-box">{result["model_lp"]}</div>',
                    unsafe_allow_html=True,
                )

        elif result["status"] == "Rejected":
            st.markdown(
                f'<div class="error-box">'
                f'⛔ Problem rejected\n\n'
                f'{result["error"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

        elif result["status"] == "Timeout":
            st.markdown(
                f'<div class="error-box">'
                f'⏱️ Solver timeout\n\n'
                f'{result["error"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                f'<div class="error-box">'
                f'Status: {result["status"]}\n'
                f'No optimal solution found.\n'
                f'Check your constraints (infeasibility or unboundedness possible).'
                f'</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "LP Solver PoC · Mistral (Ollama) + PuLP/CBC · "
    "[Code source](https://github.com) · Phase 1 — GitHub/Streamlit Cloud"
)
