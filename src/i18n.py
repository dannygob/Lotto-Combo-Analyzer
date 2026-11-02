# Internationalization (i18n) module for Lotto Combo Analyzer
# Default language is English; additional languages: Spanish, French, German

from __future__ import annotations

_CURRENT_LANGUAGE = "en"

# English (default) – identity mapping; every key maps to itself.
EN_TRANSLATIONS = {
    "frecuency": "Frequency",
    "clusters": "Clusters",
    "association": "Association Rules",
    "model-metrics": "Model Metrics",
    "correlation": "Correlation Matrix",
    "load": "Load",
    # add more as needed
}

# Spanish translations
ES_TRANSLATIONS = {
    "frecuency": "Frecuencia",
    "clusters": "Clústeres",
    "association": "Reglas de Asociación",
    "model-metrics": "Métricas de Modelos",
    "correlation": "Matriz de Correlación",
    "load": "Cargar",
    # add more as needed
}

# French translations
FR_TRANSLATIONS = {
    "frecuency": "Fréquence",
    "clusters": "Clusters",
    "association": "Règles d'association",
    "model-metrics": "Métriques de modèles",
    "correlation": "Matrice de corrélation",
    "load": "Charger",
    # add more as needed
}

# German translations
DE_TRANSLATIONS = {
    "frecuency": "Häufigkeit",
    "clusters": "Cluster",
    "association": "Assoziationsregeln",
    "model-metrics": "Modellmetriken",
    "correlation": "Korrelationsmatrix",
    "load": "Laden",
    # add more as needed
}


def get_catalog(language: str = "en") -> dict:
    """Return the translation catalog for the given language code."""
    catalogs = {
        "en": EN_TRANSLATIONS,
        "es": ES_TRANSLATIONS,
        "fr": FR_TRANSLATIONS,
        "de": DE_TRANSLATIONS,
    }
    return catalogs.get(language, EN_TRANSLATIONS)


def _(text: str, language: str = "en") -> str:
    """Translate *text* using the catalog for *language* (fallback to English)."""
    return get_catalog(language).get(text, text)


def set_language(language: str) -> None:
    """Set the global language for the application (default: 'en')."""
    pass
