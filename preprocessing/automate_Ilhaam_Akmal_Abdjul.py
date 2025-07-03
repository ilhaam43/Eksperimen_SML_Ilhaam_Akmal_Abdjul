import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocessing(input_path, output_path):
    # Load data raw
    df = pd.read_csv(input_path)

    # Drop kolom tidak relevan
    cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    df = df.drop(columns=cols_to_drop)

    # Encode target Attrition
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    # One-hot encode fitur kategorikal
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Scaling fitur numerik kecuali target
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols.remove('Attrition')
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Simpan hasil preprocessing ke file baru
    df.to_csv(output_path, index=False)
    print(f"Preprocessing selesai, data tersimpan di {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing data otomatis")
    parser.add_argument('--input', type=str, required=True, help='Path ke file dataset raw csv')
    parser.add_argument('--output', type=str, required=True, help='Path output file csv hasil preprocessing')

    args = parser.parse_args()
    preprocessing(args.input, args.output)

## Test Trigger