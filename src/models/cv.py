"""
PurgedTimeSeriesSplit — López de Prado (2018), Advances in Financial ML.
Évite le leakage entre folds adjacents via purge + embargo.
"""
import os
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import BaseCrossValidator


@dataclass
class PurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Walk-forward CV avec purge et embargo pour séries financières.

    Purge  : retire du train les observations dont le label chevauche la fenêtre test.
    Embargo: retire les N observations immédiatement après le test (autocorrélation résiduelle).
            C'est aussi une méthode qui introduit un écart temporel entre les folds de validation
            croisée afin déviter la contamination des données entre les ensemblres d'entrainement et
            de test Ce qui réduit le biais de surapprentissage


    Parameters
    ----------
    n_splits    : nombre de folds
    embargo_pct : fraction des données utilisée comme embargo après chaque fold test
    """
    n_splits: int = 5
    embargo_pct: float = 0.01

    @classmethod
    def from_github_actions(
        cls
    ) -> "PurgedTimeSeriesSplit":
        """
        Instancie la validation croisée en lisant les variables d'environnement du workflow.
        Utilise les valeurs 5 et 0.01 par défaut si rien n'est défini dans le YAML.
        """
        splits = int(os.getenv("CV_SPLITS", 5))
        embargo = float(os.getenv("CV_EMBARGO", 0.01))

        return cls(n_splits=splits, embargo_pct=embargo)

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        embargo = int(n * self.embargo_pct)

        for i in range(1, self.n_splits + 1):
            test_start = i * fold_size
            test_end = test_start + fold_size
            purge_start = max(0, test_start - embargo)
            train_idx = np.concatenate([
                np.arange(0, purge_start),
                np.arange(min(test_end + embargo, n), n),
            ])
            test_idx = np.arange(test_start, min(test_end, n))

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
