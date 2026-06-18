import pandas as pd
import numpy as np
import joblib, json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

print("📦 Chargement dataset...")
df = pd.read_csv('dataset_orientation_ucd.csv')

# Drop colonnes inutiles pour ML
df = df.drop(columns=['id_etudiant', 'note_info', 'y_filiere_nom'])

# Colonnes catégorielles (texte → chiffres)
CAT_COLS = ['sexe','bac_type','bac_mention',
            'q1_matiere_preferee','q2_activite_preferee',
            'q3_type_problemes_prefere','q4_preference_travail',
            'q5_preference_environnement','q25_objectif_carriere']

# Colonnes numériques (notes + intérêts 1-5)
NUM_COLS = ['moyenne_generale_bac','note_math','note_physique','note_chimie','note_biologie',
            'q6_interet_biologie','q7_interet_chimie','q8_interet_math','q9_interet_physique',
            'q10_interet_informatique','q11_interet_terrain','q12_interet_sante',
            'q13_interet_environnement','q14_interet_industrie',
            'q15_interet_developpement_logiciel','q16_interet_data_ia',
            'q17_niveau_memorisation','q18_niveau_analyse','q19_niveau_abstraction',
            'q20_niveau_programmation','q21_aisance_laboratoire_bio',
            'q22_aisance_laboratoire_phys_chim','q23_aisance_projets_numeriques',
            'q24_aisance_sorties_terrain']

print("🔧 Encoding colonnes catégorielles...")
encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col+'_enc'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

le_target = LabelEncoder()
df['target'] = le_target.fit_transform(df['y_filiere_code'])

FEATURES = NUM_COLS + [c+'_enc' for c in CAT_COLS]
X = df[FEATURES]
y = df['target']

print(f"✅ X: {X.shape} | Y classes ({len(le_target.classes_)}): {list(le_target.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("🤖 Training XGBoost...")
model = XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cv  = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"\n✅ Accuracy test:      {acc*100:.2f}%")
print(f"✅ Cross-val (5-fold): {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
print(f"\n{classification_report(y_test, y_pred, target_names=le_target.classes_)}")

# Sauvegarder
joblib.dump(model,     'model_xgb.pkl')
joblib.dump(le_target, 'le_target.pkl')
joblib.dump(encoders,  'encoders.pkl')

meta = {
    "features": FEATURES, "cat_cols": CAT_COLS, "num_cols": NUM_COLS,
    "filieres": list(le_target.classes_),
    "accuracy": round(acc*100,2), "cv_mean": round(cv.mean()*100,2)
}
with open('model_meta.json','w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n✅ Sauvegardé: model_xgb.pkl | le_target.pkl | encoders.pkl | model_meta.json")
