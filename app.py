"""
LP/MIP Solver — Interface Streamlit
Supporte LP, ILP, BLP et MIP avec détection automatique des types de variables.
"""

import json
import streamlit as st
from agent import extract_lp_from_text, extract_lp_from_lp_format
from solver import solve, format_results, VAR_TYPE_LABELS

st.set_page_config(
    page_title="LP/MIP Solver — Optimisation",
    page_icon="📐",
    layout="wide",
)

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
    .badge-continuous { background:#e0f2fe; color:#0369a1; padding:2px 8px; border-radius:4px; font-size:0.75rem; }
    .badge-integer    { background:#fef9c3; color:#854d0e; padding:2px 8px; border-radius:4px; font-size:0.75rem; }
    .badge-binary     { background:#fce7f3; color:#9d174d; padding:2px 8px; border-radius:4px; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)

st.title("📐 LP / MIP Solver — Optimisation Linéaire")
st.caption(
    "Supporte les programmes linéaires continus (LP), en nombres entiers (ILP/MIP) "
    "et binaires (BLP). Le LLM détecte automatiquement le type de chaque variable."
)
st.divider()

# ---------------------------------------------------------------------------
# Mode de saisie
# ---------------------------------------------------------------------------

mode = st.radio(
    "Mode de saisie",
    options=["Texte naturel", "Format LP (CPLEX-style)"],
    horizontal=True,
)

EXAMPLE_TEXT = (
    "Une entreprise doit décider d'ouvrir ou non 3 entrepôts (A, B, C). "
    "L'ouverture de A coûte 10k€, B 8k€, C 12k€. "
    "Chaque entrepôt peut livrer des clients : A peut livrer 500 unités, "
    "B 400 unités, C 600 unités. La demande totale est d'au moins 800 unités. "
    "Minimiser le coût total d'ouverture tout en satisfaisant la demande."
)

EXAMPLE_LP = """\
minimize
  obj: 10 xA + 8 xB + 12 xC

subject to
  demande: 500 xA + 400 xB + 600 xC >= 800

binary
  xA xB xC
"""

col1, col2 = st.columns([3, 1])
with col1:
    if mode == "Texte naturel":
        user_input = st.text_area("Description du problème", value="", height=200, placeholder=EXAMPLE_TEXT)
    else:
        user_input = st.text_area("Bloc LP/MIP", value="", height=200, placeholder=EXAMPLE_LP)

with col2:
    st.markdown("**Exemple MIP**")
    if st.button("📋 Charger l'exemple", use_container_width=True):
        st.session_state["example_loaded"] = EXAMPLE_TEXT if mode == "Texte naturel" else EXAMPLE_LP
        st.rerun()
    st.markdown("---")
    st.markdown("**Types supportés**")
    st.markdown('<span class="badge-continuous">Continue</span>', unsafe_allow_html=True)
    st.markdown('<span class="badge-integer">Entière</span>', unsafe_allow_html=True)
    st.markdown('<span class="badge-binary">Binaire 0/1</span>', unsafe_allow_html=True)

if "example_loaded" in st.session_state:
    user_input = st.session_state.pop("example_loaded")

st.divider()
solve_btn = st.button("🚀 Résoudre", type="primary")

if solve_btn:
    if not user_input or not user_input.strip():
        st.warning("Veuillez saisir un problème avant de lancer la résolution.")
    else:
        # Extraction LLM
        with st.status("🤖 Extraction du modèle via LLM…", expanded=True) as status:
            try:
                st.write("Appel à OpenRouter…")
                if mode == "Texte naturel":
                    lp_model = extract_lp_from_text(user_input)
                else:
                    lp_model = extract_lp_from_lp_format(user_input)
                st.write("✅ Modèle extrait et validé.")
                status.update(label="Modèle extrait.", state="complete")
            except Exception as e:
                status.update(label="Erreur d'extraction", state="error")
                st.markdown(f'<div class="error-box">❌ {e}</div>', unsafe_allow_html=True)
                st.stop()

        # Résumé des types détectés
        with st.expander("🔍 Modèle extrait (JSON)", expanded=False):
            st.json(lp_model)

        # Détecter le type de problème
        types = [v.get("type", "continuous") for v in lp_model["variables"].values()]
        has_integer = "integer" in types
        has_binary = "binary" in types
        if has_binary and has_integer:
            problem_type = "MIP — Mixte Entier/Binaire"
        elif has_binary:
            problem_type = "BLP — Programmation Binaire"
        elif has_integer:
            problem_type = "ILP — Programmation Entière"
        else:
            problem_type = "LP — Programmation Linéaire Continue"

        st.info(f"**Type de problème détecté :** {problem_type}")

        # Résolution
        with st.status("⚙️ Résolution avec PuLP / CBC…", expanded=False) as status:
            try:
                result = solve(lp_model)
                status.update(label="Résolution terminée.", state="complete")
            except Exception as e:
                status.update(label="Erreur de résolution", state="error")
                st.markdown(f'<div class="error-box">❌ Erreur PuLP :\n{e}</div>', unsafe_allow_html=True)
                st.stop()

        # Affichage résultats
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
                    vtype = result.get("variable_types", {}).get(var, "continuous")
                    badge_class = f"badge-{vtype}"
                    label_type = VAR_TYPE_LABELS.get(vtype, vtype)
                    st.markdown(
                        f'<span class="{badge_class}">{label_type}</span> '
                        f'**{var}** = `{val:.6g}`',
                        unsafe_allow_html=True,
                    )

            with col_b:
                st.markdown("**Résumé textuel**")
                st.markdown(f'<div class="result-box">{format_results(result)}</div>', unsafe_allow_html=True)

            with st.expander("📄 Formulation LP interne (PuLP)", expanded=False):
                st.markdown(f'<div class="model-box">{result["model_lp"]}</div>', unsafe_allow_html=True)

        elif result["status"] in ("Rejected", "Timeout"):
            icon = "⛔" if result["status"] == "Rejected" else "⏱️"
            st.markdown(
                f'<div class="error-box">{icon} {result["status"]}\n\n{result["error"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="error-box">Status: {result["status"]}\n'
                f'No optimal solution found.</div>',
                unsafe_allow_html=True,
            )

st.divider()
st.caption(
    "LP/MIP Solver PoC · OpenRouter + PuLP/CBC · "
    "Phase 1 — GitHub/Streamlit Cloud · CC BY-NC 4.0"
)
