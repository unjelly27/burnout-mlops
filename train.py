import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("data.csv")

# ✅ SELECT ONLY REQUIRED FEATURES
selected_features = [
    "psyt", "jspe", "qcae_cog", "qcae_aff", "amsp",
    "erec_mean", "cesd", "stai_t", "mbi_cy", "mbi_ea"
]

# Create target
def categorize_burnout(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    else:
        return 2

df["burnout_level"] = df["mbi_ex"].apply(categorize_burnout)

# ✅ USE ONLY SAME FEATURES
X = df[selected_features]
y = df["burnout_level"]

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save BOTH model + feature order
with open("model.pkl", "wb") as f:
    pickle.dump((model, selected_features), f)

print("Model fixed and saved!")