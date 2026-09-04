import yaml
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("ConfigLoader")


def load_market_config(
    config_path: Path
) -> Dict[str, Any]:
    """Charge une configuration spécifique depuis un fichier YAML donné."""
    if not config_path.exists():
        logger.error(f"Fichier de config introuvable : {config_path}")
        return {}

    try:
        # Ajout de l'encodage utf-8 pour éviter les crashs sur les accents/emojis
        with open(config_path, 'r', encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        logger.error(f"Erreur YAML dans {config_path}: {e}")
        return {}


def get_ticker_names(market: str, base_dir: Path) -> dict:
    """
    Charge le mapping ticker -> nom complet en scannant config/markets/*.yml
    et en matchant sur le champ interne "market_name" (insensible à la casse).
    """
    markets_dir = base_dir / "config" / "markets"
    if not markets_dir.exists():
        logger.warning(f"Dossier de config introuvable : {markets_dir}")
        return {}

    # Scan sécurisé pour capter à la fois .yml et .yaml
    config_files = list(markets_dir.glob("*.yml")) + list(markets_dir.glob("*.yaml"))

    for config_path in config_files:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                
            if config.get("market_name", "").strip().upper() == market.strip().upper():
                return config.get("ticker_names", {})
                
        except Exception as e:
            logger.warning(f"Lecture échouée pour {config_path} : {e}")
            continue

    logger.warning(f"Aucune config trouvée pour le marché '{market}'")
    return {}


def apply_ticker_names(
    df: pd.DataFrame,
    ticker_names: dict,
    ticker_col: str = "Ticker",
    name_col: str = "Name"
) -> pd.DataFrame:
    """
    Ajoute une colonne `name_col` juste après `ticker_col` avec le nom complet
    de chaque ticker (fallback sur le ticker brut si absent du mapping).
    """
    if df.empty or ticker_col not in df.columns:
        return df

    df = df.copy()
    df[name_col] = df[ticker_col].map(ticker_names).fillna(df[ticker_col])

    cols = df.columns.tolist()
    cols.remove(name_col)
    insert_at = cols.index(ticker_col) + 1
    cols.insert(insert_at, name_col)

    return df[cols]
