# app.py (με ενσωματωμένη επιλογή σεναρίου βάσει ισορροπίας φύλου και δυνατότητα εναλλαγής)

import streamlit as st
import pandas as pd
import math
import random
from io import BytesIO
from itertools import combinations

# --- Έλεγχος πρόσβασης ---
password = st.text_input("🔒 Εισάγετε τον κωδικό πρόσβασης:", type="password")
if password != "katanomi2025":
    st.warning("Μη εξουσιοδοτημένη πρόσβαση. Παρακαλώ εισάγετε τον σωστό κωδικό.")
    st.stop()

# --- Υπολογισμός αριθμού τμημάτων ---
def calculate_num_classes(num_students):
    return math.ceil(num_students / 25)

# --- Βοηθητική: Έλεγχος πλήρως αμοιβαίας φιλίας ---
def is_mutual_friend(df, child1, child2):
    friends1 = str(df.loc[df['ΟΝΟΜΑΤΕΠΩΝΥΜΟ'] == child1, 'ΦΙΛΟΙ'].values[0]).replace(' ', '').split(',')
    friends2 = str(df.loc[df['ΟΝΟΜΑΤΕΠΩΝΥΜΟ'] == child2, 'ΦΙΛΟΙ'].values[0]).replace(' ', '').split(',')
    return child2 in friends1 and child1 in friends2

# --- ΝΕΟ: Βήματα 1-3 για όλα τα πιθανά σενάρια κατανομής παιδιών εκπαιδευτικών ---
def generate_teacher_scenarios(df, num_classes):
    teacher_children = df[df['ΠΑΙΔΙ ΕΚΠΑΙΔΕΥΤΙΚΟΥ'] == 'Ν']
    scenarios = []

    from itertools import product
    all_assignments = list(product(range(num_classes), repeat=len(teacher_children)))
    seen = set()
    for assignment in all_assignments:
        if tuple(sorted(assignment)) in seen:
            continue
        seen.add(tuple(sorted(assignment)))
        scenario_df = df.copy()
        for idx, name in enumerate(teacher_children['ΟΝΟΜΑΤΕΠΩΝΥΜΟ']):
            scenario_df.loc[scenario_df['ΟΝΟΜΑΤΕΠΩΝΥΜΟ'] == name, 'ΠΡΟΤΕΙΝΟΜΕΝΟ ΤΜΗΜΑ'] = f'T{assignment[idx] + 1}'
        scenarios.append(scenario_df)

    return scenarios

# --- Εφαρμογή Βημάτων 2 και 3 για κάθε σενάριο ---
def apply_steps_2_and_3(df, num_classes):
    result_df = df.copy()

    # Βήμα 2: Ζωηροί
    lively = result_df[result_df['ΖΩΗΡΟΣ'] == 'Ν']
    for name in lively['ΟΝΟΜΑΤΕΠΩΝΥΜΟ']:
        if pd.notna(result_df.loc[result_df['ΟΝΟΜΑΤΕΠΩΝΥΜΟ'] == name, 'ΠΡΟΤΕΙΝΟΜΕΝΟ ΤΜΗΜΑ'].values[0]):
            continue
        for i in range(1, num_classes+1):
            if len(result_df[(result_df['ΠΡΟΤΕΙΝΟΜΕΝΟ ΤΜΗΜΑ'] == f'T{i}') & (result_df['ΖΩΗΡΟΣ'] == 'Ν')]) < 1:
                result_df.loc[result_df['ΟΝΟΜΑΤΕΠΩΝΥΜΟ'] == name, 'ΠΡΟΤΕΙΝΟΜΕΝΟ ΤΜΗΜΑ'] = f'T{i}'
                break

    # Βήμα 3: Ιδιαιτερότητες
    special = result_df[result_df['ΙΔΙΑΙΤΕΡΟΤΗΤΑ'] == 'Ν']
    for name in special['ΟΝΟΜΑΤΕΠΩΝΥΜΟ']:
        if pd.notna(result_df.loc[result_df['ΟΝΟΜΑΤΕΠΩΝΥΜΟ'] == name, 'ΠΡΟΤΕΙΝΟΜΕΝΟ ΤΜΗΜΑ'].values[0]):
            continue
        counts = result_df[result_df['ΙΔΙΑΙΤΕΡΟΤΗΤΑ'] == 'Ν'].groupby('ΠΡΟΤΕΙΝΟΜΕΝΟ ΤΜΗΜΑ').size()
        min_class = counts.idxmin() if not counts.empty else 'T1'
        result_df.loc[result_df['ΟΝΟΜΑΤΕΠΩΝΥΜΟ'] == name, 'ΠΡΟΤΕΙΝΟΜΕΝΟ ΤΜΗΜΑ'] = min_class

    return result_df

# --- Επιλογή σεναρίου βάσει ισορροπίας φύλου ---
def choose_best_scenario(scenarios):
    best_diff = float('inf')
    best_index = -1
    best_scenario = None
    for i, df in enumerate(scenarios):
        gender_counts = df.groupby('ΠΡΟΤΕΙΝΟΜΕΝΟ ΤΜΗΜΑ')['ΦΥΛΟ'].value_counts().unstack().fillna(0)
        diffs = [abs(gender_counts.loc[t, 'Α'] - gender_counts.loc[t, 'Κ']) for t in gender_counts.index if 'Α' in gender_counts.columns and 'Κ' in gender_counts.columns]
        max_diff = max(diffs) if diffs else 0
        if max_diff < best_diff:
            best_diff = max_diff
            best_index = i
            best_scenario = df
    return best_scenario, best_index

# --- Streamlit interface (βελτιωμένο με επιλογή σεναρίου) ---
st.title("Κατανομή Μαθητών - Επιλογή Σεναρίου")

uploaded_file = st.file_uploader("Ανεβάστε το Excel αρχείο μαθητών", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    num_classes = calculate_num_classes(len(df))
    scenarios = generate_teacher_scenarios(df, num_classes)
    processed_scenarios = [apply_steps_2_and_3(s, num_classes) for s in scenarios]

    best_df, best_index = choose_best_scenario(processed_scenarios)

    selected_option = st.radio("Επιλέξτε Σενάριο για Προβολή:", options=list(range(len(processed_scenarios))), format_func=lambda x: f"Σενάριο {x+1}" if x != best_index else f"Σενάριο {x+1} ⭐ (προτεινόμενο)")

    selected_df = processed_scenarios[selected_option]

    st.dataframe(selected_df[selected_df['ΠΡΟΤΕΙΝΟΜΕΝΟ ΤΜΗΜΑ'].notna()][['ΟΝΟΜΑΤΕΠΩΝΥΜΟ', 'ΦΥΛΟ', 'ΠΑΙΔΙ ΕΚΠΑΙΔΕΥΤΙΚΟΥ', 'ΖΩΗΡΟΣ', 'ΙΔΙΑΙΤΕΡΟΤΗΤΑ', 'ΠΡΟΤΕΙΝΟΜΕΝΟ ΤΜΗΜΑ']])

    if selected_option == best_index:
        st.success("✅ Επιλέξατε το σενάριο με καλύτερη ισορροπία φύλου.")
    else:
        st.info("ℹ Μπορείτε να εξετάσετε διαφορετικά σενάρια πριν προχωρήσετε.")

    # --- Κουμπί Οριστικοποίησης ---
    if st.button("🔒 Οριστικοποίηση Σενάριου για Επόμενα Βήματα"):
        selected_df['ΤΜΗΜΑ'] = selected_df['ΠΡΟΤΕΙΝΟΜΕΝΟ ΤΜΗΜΑ']
        st.success("Οι τιμές από τη στήλη 'ΠΡΟΤΕΙΝΟΜΕΝΟ ΤΜΗΜΑ' αντιγράφηκαν στη στήλη 'ΤΜΗΜΑ'.")

    # --- ΝΕΟ: Κουμπί Εξαγωγής Excel ---
    output = BytesIO()
    selected_df.to_excel(output, index=False)
    st.download_button(
        label="📥 Λήψη Excel Κατανομής",
        data=output.getvalue(),
        file_name="katanomi_senario.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
