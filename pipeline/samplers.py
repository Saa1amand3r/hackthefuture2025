from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

def _k_for_smote(y) -> int:
    minority = min(Counter(y).values())
    return max(1, min(5, minority - 1))

class AutoKSMOTE(SMOTE):
    def fit_resample(self, X, y):
        self.k_neighbors = _k_for_smote(y)
        return super().fit_resample(X, y)


SAMPLER_REGISTRY: dict[str, object] = {
    "none":  None,
    "up_floor": RandomOverSampler(
        random_state=42
    ),
    "down":  RandomUnderSampler(random_state=42),
    "smote": AutoKSMOTE(random_state=42),
}
