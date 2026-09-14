from pathlib import Path

import pandas as pd


def show_composition(market_name="CAC40"):
    # Chemin vers l'historique de rebalancement
    path = Path(f"data/processed/{market_name}/rebalance_history.parquet")

    if not path.exists():
        print("❌ Fichier de rebalancement introuvable. Relance daily_run.py.")
        return

    df = pd.read_parquet(path)

    # On prend la dernière ligne disponible (le dernier rebalancement)
    latest_weights = df.iloc[-1]

    # On filtre les actions qui ont un poids > 0
    portfolio = latest_weights[latest_weights > 0.001].sort_values(ascending=False)

    print(f"\n--- Composition du Portefeuille {market_name} (Dernière mise à jour) ---")
    print(portfolio.to_string(formatters={"Weight": "{:,.2%}".format}))
    print(f"\nTotal investi : {portfolio.sum():.2%}")


if __name__ == "__main__":
    # Tu peux changer pour "US_TECH" si besoin
    show_composition("CAC40")
