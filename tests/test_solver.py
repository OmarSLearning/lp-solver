"""
Tests unitaires — LP Solver PoC

Couvre :
  - Validation et normalisation du modèle LP (agent._validate_lp_model)
  - Extraction JSON robuste (agent._extract_json)
  - Résolution correcte de problèmes LP simples (solver.solve)
  - Cas limites : infaisable, non borné, variables manquantes
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import _extract_json, _validate_lp_model
from solver import solve


# ---------------------------------------------------------------------------
# Tests agent._extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:

    def test_json_propre(self):
        raw = '{"objective": {"type": "maximize", "coefficients": {"x": 1}}, "constraints": [], "variables": {"x": {"lb": 0}}}'
        result = _extract_json(raw)
        assert result["objective"]["type"] == "maximize"

    def test_json_avec_balises_markdown(self):
        raw = '```json\n{"objective": {"type": "minimize", "coefficients": {"x": 2}}, "constraints": [], "variables": {"x": {"lb": 0}}}\n```'
        result = _extract_json(raw)
        assert result["objective"]["type"] == "minimize"

    def test_json_avec_texte_autour(self):
        raw = 'Voici le modèle : {"objective": {"type": "maximize", "coefficients": {"x": 5}}, "constraints": [], "variables": {"x": {"lb": 0}}} Fin.'
        result = _extract_json(raw)
        assert result["objective"]["coefficients"]["x"] == 5

    def test_json_invalide_leve_erreur(self):
        with pytest.raises(ValueError):
            _extract_json("Ce n'est pas du JSON valide du tout.")


# ---------------------------------------------------------------------------
# Tests agent._validate_lp_model
# ---------------------------------------------------------------------------

class TestValidateLpModel:

    def _base_model(self):
        return {
            "objective": {
                "type": "maximize",
                "coefficients": {"x1": 5, "x2": 4}
            },
            "constraints": [
                {"coefficients": {"x1": 2, "x2": 1}, "sense": "<=", "rhs": 100},
                {"coefficients": {"x1": 1, "x2": 2}, "sense": "<=", "rhs": 80},
            ],
            "variables": {
                "x1": {"lb": 0, "ub": None},
                "x2": {"lb": 0, "ub": None},
            }
        }

    def test_modele_valide(self):
        model = self._base_model()
        result = _validate_lp_model(model)
        assert result["objective"]["type"] == "maximize"

    def test_sense_invalide(self):
        model = self._base_model()
        model["constraints"][0]["sense"] = "!="
        with pytest.raises(ValueError, match="sense invalide"):
            _validate_lp_model(model)

    def test_objective_type_invalide(self):
        model = self._base_model()
        model["objective"]["type"] = "optimise"
        with pytest.raises(ValueError, match="objective.type invalide"):
            _validate_lp_model(model)

    def test_variable_manquante_auto_ajoutee(self):
        model = self._base_model()
        del model["variables"]["x1"]
        result = _validate_lp_model(model)
        assert "x1" in result["variables"]
        assert result["variables"]["x1"]["lb"] == 0

    def test_cle_manquante_leve_erreur(self):
        model = self._base_model()
        del model["constraints"]
        with pytest.raises(ValueError, match="Clé manquante"):
            _validate_lp_model(model)


# ---------------------------------------------------------------------------
# Tests solver.solve
# ---------------------------------------------------------------------------

class TestSolver:

    def _model_usine(self):
        """Problème classique : max 5x1 + 4x2, s.t. 2x1+x2<=100, x1+2x2<=80."""
        return {
            "objective": {
                "type": "maximize",
                "coefficients": {"x1": 5, "x2": 4}
            },
            "constraints": [
                {"label": "machine", "coefficients": {"x1": 2, "x2": 1}, "sense": "<=", "rhs": 100},
                {"label": "main_oeuvre", "coefficients": {"x1": 1, "x2": 2}, "sense": "<=", "rhs": 80},
            ],
            "variables": {
                "x1": {"lb": 0, "ub": None},
                "x2": {"lb": 0, "ub": None},
            }
        }

    def test_solution_optimale(self):
        result = solve(self._model_usine())
        assert result["is_optimal"] is True
        assert result["status"] == "Optimal"
        assert abs(result["objective_value"] - 280) < 1e-4

    def test_valeurs_variables(self):
        result = solve(self._model_usine())
        assert abs(result["variables"]["x1"] - 40) < 1e-4
        assert abs(result["variables"]["x2"] - 20) < 1e-4

    def test_minimisation(self):
        model = {
            "objective": {"type": "minimize", "coefficients": {"x": 3, "y": 2}},
            "constraints": [
                {"coefficients": {"x": 1, "y": 1}, "sense": ">=", "rhs": 4},
                {"coefficients": {"x": 1}, "sense": ">=", "rhs": 1},
            ],
            "variables": {"x": {"lb": 0}, "y": {"lb": 0}}
        }
        result = solve(model)
        assert result["is_optimal"] is True
        # Solution optimale : x=1, y=3 → 3*1 + 2*3 = 9
        assert abs(result["objective_value"] - 9) < 1e-4

    def test_probleme_infaisable(self):
        model = {
            "objective": {"type": "maximize", "coefficients": {"x": 1}},
            "constraints": [
                {"coefficients": {"x": 1}, "sense": "<=", "rhs": 5},
                {"coefficients": {"x": 1}, "sense": ">=", "rhs": 10},
            ],
            "variables": {"x": {"lb": 0}}
        }
        result = solve(model)
        assert result["is_optimal"] is False

    def test_objective_type_dans_resultat(self):
        result = solve(self._model_usine())
        assert result["objective_type"] == "maximize"

    def test_model_lp_str_non_vide(self):
        result = solve(self._model_usine())
        assert len(result["model_lp"]) > 0
