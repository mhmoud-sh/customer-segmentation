
---

### ✅ `utils/preprocessing.py`

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df, features):
    df = df.dropna()
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
