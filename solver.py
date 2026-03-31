"""
Solveur PuLP MIP — supporte LP, ILP, BLP et MIP.

Protections :
  - Limite taille : 50 variables, 100 contraintes
  - Timeout CBC : 30 secondes
"""

import pulp

MAX_VARIABLES = 50
MAX_CONSTRAINTS = 100
CBC_TIME_LIMIT = 30

# Mapping type → PuLP cat
VAR_TYPE_MAP = {
    "continuous": pulp.LpContinuous,
    "integer":    pulp.LpInteger,
    "binary":     pulp.LpBinary,
}

# Labels lisibles pour l'affichage
VAR_TYPE_LABELS = {
    "continuous": "Continue",
    "integer":    "Entière",
    "binary":     "Binaire (0/1)",
}


def solve(lp_model: dict) -> dict:
    n_vars = len(lp_model["variables"])
    n_constraints = len(lp_model["constraints"])

    if n_vars > MAX_VARIABLES or n_constraints > MAX_CONSTRAINTS:
        return {
            "status": "Rejected",
            "is_optimal": False,
            "objective_value": None,
            "variables": {},
            "variable_types": {},
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
    sense = pulp.LpMinimize if obj_cfg["type"] == "minimize" else pulp.LpMaximize
    prob = pulp.LpProblem("MIP_Problem", sense)

    # --- Variables de décision avec type ---
    lp_vars = {}
    variable_types = {}
    for name, cfg in lp_model["variables"].items():
        var_type = cfg.get("type", "continuous")
        lb = cfg.get("lb", 0)
        ub = cfg.get("ub", None)
        pulp_cat = VAR_TYPE_MAP.get(var_type, pulp.LpContinuous)

        if var_type == "binary":
            lp_vars[name] = pulp.LpVariable(name, cat=pulp.LpBinary)
        else:
            lp_vars[name] = pulp.LpVariable(name, lowBound=lb, upBound=ub, cat=pulp_cat)

        variable_types[name] = var_type

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
        if c["sense"] == "<=":
            prob += (expr <= c["rhs"]), label
        elif c["sense"] == ">=":
            prob += (expr >= c["rhs"]), label
        else:
            prob += (expr == c["rhs"]), label

    # --- Résolution avec timeout ---
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=CBC_TIME_LIMIT))

    status = pulp.LpStatus[prob.status]
    is_optimal = prob.status == pulp.constants.LpStatusOptimal

    if not is_optimal and prob.status == 0:
        return {
            "status": "Timeout",
            "is_optimal": False,
            "objective_value": None,
            "variables": {},
            "variable_types": variable_types,
            "model_lp": str(prob),
            "objective_type": obj_cfg["type"],
            "error": (
                f"The solver exceeded the time limit of {CBC_TIME_LIMIT} seconds. "
                "Your problem may be too complex for this demo app."
            ),
        }

    obj_value = pulp.value(prob.objective) if is_optimal else None
    var_values = {name: pulp.value(var) for name, var in lp_vars.items()}

    return {
        "status": status,
        "objective_value": obj_value,
        "variables": var_values,
        "variable_types": variable_types,
        "is_optimal": is_optimal,
        "model_lp": str(prob),
        "objective_type": obj_cfg["type"],
        "error": None,
    }


def format_results(result: dict) -> str:
    lines = [f"Status : {result['status']}"]

    if result["is_optimal"]:
        obj_type = result.get("objective_type", "")
        lines.append(f"Optimal value ({obj_type}) : {result['objective_value']:.6g}")
        lines.append("\nDecision variables:")
        for var, val in result["variables"].items():
            vtype = result.get("variable_types", {}).get(var, "continuous")
            label = VAR_TYPE_LABELS.get(vtype, vtype)
            lines.append(f"  {var} = {val:.6g}  [{label}]")
    elif result["status"] in ("Rejected", "Timeout"):
        lines.append(f"\n{result.get('error', 'No solution found.')}")
    else:
        lines.append("No optimal solution found.")

    return "\n".join(lines)

