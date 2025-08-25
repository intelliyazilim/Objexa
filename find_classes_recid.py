import pandas as pd
import difflib

def normalize_text(text):
    return str(text).strip().replace('ı', 'i').replace('İ', 'i').lower()

def find_rec_id(df, search_value):
    try:
        search_value_numeric = pd.to_numeric(search_value, errors='raise')
        df['ClassID'] = pd.to_numeric(df['ClassID'], errors='coerce')
        match_row = df[df['ClassID'] == search_value_numeric]
        if not match_row.empty:
            return (
                match_row.iloc[0]['Class_recID'],
                match_row.iloc[0]['Class_name_tr'],
                match_row.iloc[0]['Class_name_ing'],
                match_row.iloc[0]['Min_Height'],
                match_row.iloc[0]['Max_Height']
            )
    except ValueError:
        search_value_normalized = normalize_text(search_value)

        df['Class_name_tr_norm'] = df['Class_name_tr'].apply(normalize_text)
        df['Class_name_ing_norm'] = df['Class_name_ing'].apply(normalize_text)

        exact_match = df[
            (df['Class_name_tr_norm'] == search_value_normalized) |
            (df['Class_name_ing_norm'] == search_value_normalized)
        ]
        if not exact_match.empty:
            row = exact_match.iloc[0]
            return (
                row['Class_recID'],
                row['Class_name_tr'],
                row['Class_name_ing'],
                row['Min_Height'],
                row['Max_Height']
            )
        
        for _, row in df.iterrows():
            if search_value_normalized in row['Class_name_tr_norm'] or search_value_normalized in row['Class_name_ing_norm']:
                return (
                    row['Class_recID'],
                    row['Class_name_tr'],
                    row['Class_name_ing'],
                    row['Min_Height'],
                    row['Max_Height']
                )

        for col in ['Class_name_tr_norm', 'Class_name_ing_norm']:
            match_list = difflib.get_close_matches(search_value_normalized, df[col], n=1, cutoff=0.6)
            if match_list:
                matched_value = match_list[0]
                match_row = df[df[col] == matched_value]
                if not match_row.empty:
                    row = match_row.iloc[0]
                    return (
                        row['Class_recID'],
                        row['Class_name_tr'],
                        row['Class_name_ing'],
                        row['Min_Height'],
                        row['Max_Height']
                    )

    others_row = df[df['Class_name_tr'].str.lower() == 'others']
    if not others_row.empty:
        row = others_row.iloc[0]
        return (
            row['Class_recID'],
            row['Class_name_tr'],
            row['Class_name_ing'],
            row['Min_Height'],
            row['Max_Height']
        )

    return None