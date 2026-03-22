# LP Solver — Optimisation Linéaire Agentique

Résolution de programmes linéaires pilotée par LLM.  
L'utilisateur saisit un problème en **texte naturel** ou en **format LP structuré**.  
L'agent LLM (Mistral via Ollama) extrait le modèle, PuLP/CBC le résout.

```
Entrée (texte ou LP) → Mistral (Ollama) → JSON LP → PuLP/CBC → Résultats
```

---

## Stack

| Composant | Technologie |
|-----------|-------------|
| Interface | Streamlit |
| Agent LLM | Mistral via Ollama (local) |
| Solveur   | PuLP + CBC |
| Déploiement PoC | Streamlit Community Cloud |

---

## Installation locale

### Prérequis

- Python 3.10+
- [Ollama](https://ollama.ai) installé et en cours d'exécution
- Modèle Mistral téléchargé

```bash
# Installer Ollama (Linux/macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# Télécharger le modèle
ollama pull mistral

# Vérifier qu'Ollama tourne
ollama serve
```

### Lancer l'application

```bash
# Cloner le dépôt
git clone https://github.com/<votre-username>/lp-solver.git
cd lp-solver

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer Streamlit
streamlit run app.py
```

L'application est accessible sur `http://localhost:8501`.

---

## Tests

```bash
pytest tests/ -v
```

Les tests couvrent :
- Extraction et validation JSON depuis la réponse LLM
- Normalisation du schéma LP
- Résolution correcte (maximisation, minimisation)
- Cas limites : infaisable, variables manquantes

---

## Structure du projet

```
lp-solver/
├── app.py               # Interface Streamlit
├── agent.py             # Agent LLM — extraction du modèle LP via Mistral/Ollama
├── solver.py            # Solveur PuLP/CBC
├── requirements.txt
├── .streamlit/
│   └── config.toml      # Thème Streamlit
├── tests/
│   └── test_solver.py   # Tests unitaires
└── README.md
```

---

## Déploiement sur Streamlit Community Cloud

> **Note** : Streamlit Community Cloud ne peut pas accéder à un Ollama local.  
> Pour le déploiement cloud, remplacez l'appel Ollama par une API LLM accessible
> (Groq, Together AI, OpenRouter — tous proposent Mistral gratuitement).

### Étapes

1. Pusher le projet sur GitHub (dépôt public ou privé)

```bash
git init
git add .
git commit -m "feat: LP Solver PoC initial"
git remote add origin https://github.com/<username>/lp-solver.git
git push -u origin main
```

2. Se connecter sur [share.streamlit.io](https://share.streamlit.io)
3. Cliquer **"New app"** → sélectionner le dépôt → `app.py` comme fichier principal
4. Ajouter les secrets si nécessaire (clé API Groq/Together) dans **Settings > Secrets**

### Adaptation pour Groq (Mistral gratuit en cloud)

Dans `agent.py`, remplacer `_call_ollama` par :

```python
import os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def _call_ollama(prompt: str, system: str) -> str:
    response = client.chat.completions.create(
        model="mistral-saba-24b",  # ou "llama-3.1-8b-instant"
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content
```

Ajouter `groq` dans `requirements.txt`.

---

## Exemple de problème (texte naturel)

> Une usine produit deux produits A et B.  
> Chaque unité de A rapporte 5€ et nécessite 2h de machine et 1h de main d'œuvre.  
> Chaque unité de B rapporte 4€ et nécessite 1h de machine et 2h de main d'œuvre.  
> On dispose de 100h de machine et 80h de main d'œuvre par semaine.  
> Maximiser le profit.

**Résultat attendu** : x1 = 40, x2 = 20, profit optimal = 280€

## Licence

Ce projet est sous licence [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).  
Usage commercial interdit sans autorisation explicite de l'auteur.
