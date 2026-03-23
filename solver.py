"""
Solveur PuLP — construit et résout un programme linéaire à partir du modèle
JSON canonique produit par agent.py.

Protections intégrées :
  - Limite sur le nombre de variables et contraintes (anti-surcharge)
  - Timeout CBC de 30 secondes (anti-blocage sur instances NP-difficiles)

Retourne un dict de résultats structuré, indépendant de toute couche UI.
"""

import pulp

# ---------------------------------------------------------------------------
# Limites de sécurité
# ---------------------------------------------------------------------------

MAX_VARIABLES = 50
MAX_CONSTRAINTS = 100
CBC_TIME_LIMIT = 30  # secondes

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
        status          : str  — "Optimal", "Infeasible", "Unbounded", "Rejected", "Timeout"
        objective_value : float | None
        variables       : dict[str, float]
        is_optimal      : bool
        model_lp        : str  — représentation LP textuelle (debug)
        error           : str | None — message d'erreur si rejeté
    """

    # --- Vérification de la taille du problème ---
    n_vars = len(lp_model["variables"])
    n_constraints = len(lp_model["constraints"])

    if n_vars > MAX_VARIABLES or n_constraints > MAX_CONSTRAINTS:
        return {
            "status": "Rejected",
            "is_optimal": False,
            "objective_value": None,
            "variables": {},
            "model_lp": "",
            "objective_type": lp_model["objective"]["type"],
            "error": (
                f"Problem too large for this demo app "
                f"({n_vars} variables, {n_constraints} constraints). "
                f"Maximum allowed: {MAX_VARIABLES} variables and "
                f"{MAX_CONSTRAINTS} constraints."
            ),
        }

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
        else:
            prob += (expr == rhs), label

    # --- Résolution avec timeout ---
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=CBC_TIME_LIMIT))

    status = pulp.LpStatus[prob.status]
    is_optimal = prob.status == pulp.constants.LpStatusOptimal

    # Détecter un timeout (CBC retourne status 0 = Not Solved)
    if not is_optimal and prob.status == 0:
        return {
            "status": "Timeout",
            "is_optimal": False,
            "objective_value": None,
            "variables": {},
            "model_lp": str(prob),
            "objective_type": obj_cfg["type"],
            "error": (
                f"The solver exceeded the time limit of {CBC_TIME_LIMIT} seconds. "
                "Your problem may be too complex for this demo app. "
                "Consider simplifying the model or reducing the number of integer variables."
            ),
        }

    obj_value = pulp.value(prob.objective) if is_optimal else None
    var_values = {
        name: pulp.value(var)
        for name, var in lp_vars.items()
    }

    return {
        "status": status,
        "objective_value": obj_value,
        "variables": var_values,
        "is_optimal": is_optimal,
        "model_lp": str(prob),
        "objective_type": obj_cfg["type"],
        "error": None,
    }


# ---------------------------------------------------------------------------
# Formatage lisible des résultats
# ---------------------------------------------------------------------------

def format_results(result: dict) -> str:
    """Retourne une représentation texte propre des résultats."""
    lines = []
    lines.append(f"Status       : {result['status']}")

    if result["is_optimal"]:
        obj_type = result.get("objective_type", "")
        lines.append(
            f"Optimal value ({obj_type}) : {result['objective_value']:.6g}"
        )
        lines.append("\nDecision variables:")
        for var, val in result["variables"].items():
            lines.append(f"  {var} = {val:.6g}")
    elif result["status"] in ("Rejected", "Timeout"):
        lines.append(f"\n{result.get('error', 'No solution found.')}")
    else:
        lines.append("No optimal solution found.")

    return "\n".join(lines)
