from flask import Flask, render_template, request, jsonify
import joblib, json, numpy as np

app = Flask(__name__)

# Load model + encoders
model     = joblib.load('model_xgb.pkl')
le_target = joblib.load('le_target.pkl')
encoders  = joblib.load('encoders.pkl')
with open('model_meta.json', encoding='utf-8') as f:
    meta = json.load(f)

FEATURES = meta['features']
CAT_COLS = meta['cat_cols']
NUM_COLS = meta['num_cols']

FILIERES_INFO = {
    "3G":   {"nom":"Géosciences, Géodynamique et Géomatique","couleur":"#92400e","icone":"🌍","debouches":["Géologue","Ingénieur mines","Cartographe","Environnement"],"modules":["Géophysique","Cartographie","Géologie","Géostatistiques"]},
    "BPV":  {"nom":"Biotechnologie et Production Végétale","couleur":"#15803d","icone":"🌿","debouches":["Ingénieur agronome","Biotechnologiste","Agriculture","Industrie pharma"],"modules":["Biotechnologie végétale","Physiologie végétale","Génétique","Agroalimentaire"]},
    "COA":  {"nom":"Chimie Organique Appliquée","couleur":"#dc2626","icone":"🧪","debouches":["Chimiste","Agroalimentaire","Pharmacie","Cosmétique"],"modules":["Chimie organique","Polymères","Réactions organiques","Agroalimentaire"]},
    "CPA":  {"nom":"Chimie Physique et Applications","couleur":"#ea580c","icone":"🔬","debouches":["Ingénieur procédés","Chimiste analytique","Environnement"],"modules":["Electrochimie","Cinétique","Catalyse","Environnement"]},
    "GEER": {"nom":"Génie Electrique et Energies Renouvelables","couleur":"#ca8a04","icone":"⚡","debouches":["Ingénieur électrique","Energies renouvelables","Industrie électrique"],"modules":["Electronique","Energies renouvelables","Automatique","Réseaux électriques"]},
    "IMEI": {"nom":"Ingénierie Mécanique et Energétiques Industrielles","couleur":"#0369a1","icone":"⚙️","debouches":["Ingénieur mécanique","Industrie","Bureau d'études"],"modules":["Mécanique","Thermodynamique","Matériaux","CAO"]},
    "MA":   {"nom":"Mathématiques et Applications","couleur":"#059669","icone":"📐","debouches":["Chercheur","Enseignant","Actuaire","Grandes écoles"],"modules":["Analyse","Algèbre","Topologie","Calcul différentiel"]},
    "MI":   {"nom":"Développement et Bases de données","couleur":"#2563eb","icone":"💻","debouches":["Développeur web","Admin BDD","Ingénieur logiciel"],"modules":["Programmation web","Bases de données","Réseaux","POO Java"]},
    "MIA":  {"nom":"Matériaux Inorganiques et Applications","couleur":"#d97706","icone":"⚗️","debouches":["Ingénieur matériaux","Chercheur chimiste","Industrie"],"modules":["Chimie des matériaux","Cristallographie","Electrochimie","Spectroscopie"]},
    "PM":   {"nom":"Physique Moderne","couleur":"#7c3aed","icone":"🔭","debouches":["Chercheur physicien","Enseignant","Ingénieur R&D"],"modules":["Physique quantique","Optique","Mécanique","Physique des matériaux"]},
    "SD":   {"nom":"Sciences des données","couleur":"#6d28d9","icone":"📊","debouches":["Data Scientist","Analyste données","Ingénieur ML"],"modules":["Machine learning","Big data","Python","Statistiques"]},
    "SESBM":{"nom":"Santé, Environnement et Sciences Bio Médicales","couleur":"#0891b2","icone":"🧬","debouches":["Biologiste","Chercheur santé","Pharmacie","Médecine"],"modules":["Biochimie","Microbiologie","Génétique","Immunologie"]},
    "TR":   {"nom":"Télécommunications et Réseaux","couleur":"#0f766e","icone":"📡","debouches":["Ingénieur télécom","Réseaux","Opérateur télécoms"],"modules":["Réseaux","Télécoms","Protocoles","Systèmes embarqués"]},
}

RENFORCER = {
    "MI":   [("note_math",13,"Mathématiques"),("note_biologie",0,"—")],
    "SD":   [("note_math",13,"Mathématiques & Statistiques"),("note_biologie",0,"—")],
    "MA":   [("note_math",16,"Mathématiques (niveau élevé requis)")],
    "MIA":  [("note_chimie",13,"Chimie"),("note_physique",13,"Physique")],
    "COA":  [("note_chimie",13,"Chimie organique")],
    "CPA":  [("note_chimie",13,"Chimie physique"),("note_physique",13,"Physique")],
    "PM":   [("note_physique",14,"Physique")],
    "GEER": [("note_physique",13,"Physique"),("note_math",12,"Mathématiques")],
    "IMEI": [("note_physique",13,"Physique"),("note_math",12,"Mathématiques")],
    "TR":   [("note_physique",12,"Physique"),("note_math",12,"Mathématiques")],
    "SESBM":[("note_biologie",13,"Biologie / SVT")],
    "BPV":  [("note_biologie",13,"Biologie végétale")],
    "3G":   [("note_biologie",11,"SVT & Géologie")],
}

def safe_encode(col, val):
    le = encoders.get(col)
    if le is None: return 0
    try:    return int(le.transform([str(val)])[0])
    except: return 0

def build_X(d):
    row = {
        "moyenne_generale_bac": float(d.get("moyenne_generale_bac",12)),
        "note_math":      float(d.get("note_math",12)),
        "note_physique":  float(d.get("note_physique",12)),
        "note_chimie":    float(d.get("note_chimie",12)),
        "note_biologie":  float(d.get("note_biologie",12)),
        "q6_interet_biologie":              float(d.get("q6",3)),
        "q7_interet_chimie":               float(d.get("q7",3)),
        "q8_interet_math":                 float(d.get("q8",3)),
        "q9_interet_physique":             float(d.get("q9",3)),
        "q10_interet_informatique":        float(d.get("q10",3)),
        "q11_interet_terrain":             float(d.get("q11",3)),
        "q12_interet_sante":               float(d.get("q12",3)),
        "q13_interet_environnement":       float(d.get("q13",3)),
        "q14_interet_industrie":           float(d.get("q14",3)),
        "q15_interet_developpement_logiciel": float(d.get("q15",3)),
        "q16_interet_data_ia":             float(d.get("q16",3)),
        "q17_niveau_memorisation":         float(d.get("q17",3)),
        "q18_niveau_analyse":              float(d.get("q18",3)),
        "q19_niveau_abstraction":          float(d.get("q19",3)),
        "q20_niveau_programmation":        float(d.get("q20",3)),
        "q21_aisance_laboratoire_bio":     float(d.get("q21",3)),
        "q22_aisance_laboratoire_phys_chim": float(d.get("q22",3)),
        "q23_aisance_projets_numeriques":  float(d.get("q23",3)),
        "q24_aisance_sorties_terrain":     float(d.get("q24",3)),
        "sexe_enc":           safe_encode("sexe", d.get("sexe","M")),
        "bac_type_enc":       safe_encode("bac_type", d.get("bac_type","SM")),
        "bac_mention_enc":    safe_encode("bac_mention", d.get("bac_mention","Bien")),
        "q1_matiere_preferee_enc":        safe_encode("q1_matiere_preferee", d.get("q1","")),
        "q2_activite_preferee_enc":       safe_encode("q2_activite_preferee", d.get("q2","")),
        "q3_type_problemes_prefere_enc":  safe_encode("q3_type_problemes_prefere", d.get("q3","")),
        "q4_preference_travail_enc":      safe_encode("q4_preference_travail", d.get("q4","Individuel")),
        "q5_preference_environnement_enc":safe_encode("q5_preference_environnement", d.get("q5","Mixte")),
        "q25_objectif_carriere_enc":      safe_encode("q25_objectif_carriere", d.get("q25","")),
    }
    return np.array([[row[f] for f in FEATURES]])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommander", methods=["POST"])
def recommander():
    data  = request.json
    X     = build_X(data)
    proba = model.predict_proba(X)[0]
    top3  = np.argsort(proba)[::-1][:3]

    results = []
    for rank, idx in enumerate(top3):
        code  = le_target.classes_[idx]
        score = float(proba[idx])
        info  = FILIERES_INFO.get(code, {"nom":code,"couleur":"#64748b","icone":"🎓","debouches":[],"modules":[]})

        renforcer = []
        for nkey, seuil, label in RENFORCER.get(code, []):
            if seuil > 0 and float(data.get(nkey.replace("note_","note_"), 12)) < seuil:
                renforcer.append(label)
        if not renforcer:
            renforcer = ["✅ Bon profil — continuez!"]

        results.append({
            "rang": rank+1, "code": code,
            "nom": info["nom"], "couleur": info["couleur"], "icone": info["icone"],
            "taux": round(score*100, 1),
            "modules": info["modules"], "debouches": info["debouches"],
            "renforcer": renforcer
        })
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
