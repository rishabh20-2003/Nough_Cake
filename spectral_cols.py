import joblib
spectral_cols = list(df.columns[df.columns.str.fullmatch(r'\d+')])  # Or however you define them
joblib.dump(spectral_cols, "spectral_cols.pkl")
