"""
Solveur PuLP — construit et résout un programme linéaire à partir du modèle
JSON canonique produit par agent.py.

Retourne un dict de résultats structuré, indépendant de toute couche UI.
"""

import pulp
from typing import Any


# ---------------------------------------------------------------------------
# Résolution
# ---------------------------------------------------------------------------

def solve(lp_model: dict) -> dict:
    """
    Construit et résout le programme linéaire décrit par lp_model.

    Paramètres
    ----------
    lp_model : dict
        Modèle LP canonique (schéma défini dans agent.py).

    Retourne
    --------
    dict avec les clés :
        status          : str  — "Optimal", "Infeasible", "Unbounded", etc.
        objective_value : float | None
        variables       : dict[str, float]
        is_optimal      : bool
        model_lp        : str  — représentation LP textuelle (debug)
    """
    obj_cfg = lp_model["objective"]
    sense = (
        pulp.LpMinimize
        if obj_cfg["type"] == "minimize"
        else pulp.LpMaximize
    )

    prob = pulp.LpProblem("LP_Problem", sense)

    # --- Variables de décision ---
    lp_vars = {}
    for name, bounds in lp_model["variables"].items():
        lb = bounds.get("lb", 0)
        ub = bounds.get("ub", None)
        lp_vars[name] = pulp.LpVariable(name, lowBound=lb, upBound=ub)

    # --- Fonction objectif ---
    prob += pulp.lpSum(
        coef * lp_vars[var]
        for var, coef in obj_cfg["coefficients"].items()
        if var in lp_vars
    ), "Objectif"

    # --- Contraintes ---
    for i, c in enumerate(lp_model["constraints"]):
        label = c.get("label") or f"c{i+1}"
        expr = pulp.lpSum(
            coef * lp_vars[var]
            for var, coef in c["coefficients"].items()
            if var in lp_vars
        )
        sense_c = c["sense"]
        rhs = c["rhs"]

        if sense_c == "<=":
            prob += (expr <= rhs), label
        elif sense_c == ">=":
            prob += (expr >= rhs), label
        else:  # "="
            prob += (expr == rhs), label

    # --- Résolution (CBC solver, silencieux) ---
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    is_optimal = prob.status == pulp.constants.LpStatusOptimal

    obj_value = pulp.value(prob.objective) if is_optimal else None
    var_values = {
        name: pulp.value(var)
        for name, var in lp_vars.items()
    }

    # Représentation LP textuelle pour debug/affichage
    model_lp_str = str(prob)

    return {
        "status": status,
        "objective_value": obj_value,
        "variables": var_values,
        "is_optimal": is_optimal,
        "model_lp": model_lp_str,
        "objective_type": obj_cfg["type"],
    }


# ---------------------------------------------------------------------------
# Formatage lisible des résultats
# ---------------------------------------------------------------------------

def format_results(result: dict) -> str:
    """Retourne une représentation texte propre des résultats."""
    lines = []
    lines.append(f"Statut       : {result['status']}")

    if result["is_optimal"]:
        obj_type = result.get("objective_type", "")
        lines.append(
            f"Valeur optimale ({obj_type}) : {result['objective_value']:.6g}"
        )
        lines.append("\nVariables de décision :")
        for var, val in result["variables"].items():
            lines.append(f"  {var} = {val:.6g}")
    else:
        lines.append("Aucune solution optimale trouvée.")

    return "\n".join(lines)
