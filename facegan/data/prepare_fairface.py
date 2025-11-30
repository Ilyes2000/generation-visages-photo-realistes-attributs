import os, pandas as pd
from sklearn.model_selection import train_test_split

ETH_MAP = {
    "white":0, "black":1, "latino_hispanic":2,
    "east_asian":3, "southeast_asian":4, "indian":5, "middle_eastern":6
}
ALIASES = {
    "latino": "latino_hispanic", "hispanic":"latino_hispanic",
    "east asian":"east_asian", "east_asia":"east_asian",
    "south east asian":"southeast_asian", "southeast asian":"southeast_asian",
    "middle eastern":"middle_eastern", "middle east":"middle_eastern",
}
AGE_BINS = [(18,29),(30,39),(40,49),(50,59),(60,200)]

def _age_to_bin(v):
    s = str(v).strip().lower()
    if "-" in s:
        a,b = s.replace(" ", "").split("-")
        try:
            a, b = int(a), int(b)
            if b < 18: return None
            mid = max(18, (a+b)//2)
            return _age_to_bin(mid)
        except:
            return None
    try:
        n = int(s)
    except:
        return None
    if n < 18: return None
    for i,(lo,hi) in enumerate(AGE_BINS):
        if lo <= n <= hi: return i
    return 4

def _gender_to_idx(v):
    s = str(v).strip().lower()
    if s in ["male","m","man","0","masculin"]: return 0
    if s in ["female","f","woman","1","feminin"]: return 1
    return None

def _race_to_idx(v):
    s = str(v).strip().lower()
    s = ALIASES.get(s, s).replace("-", " ").replace("_", " ")
    s = ALIASES.get(s, s)  # re-map si besoin
    s = s.replace(" ", "_")
    return ETH_MAP.get(s, None)

def build_from_existing_csv(img_root, csv_in, out_csv,
                            file_col="file", age_col="age", gender_col="gender", race_col="race",
                            val_size=0.1, test_size=0.1, seed=42):
    img_root = os.path.abspath(img_root)
    df = pd.read_csv(csv_in)

    if not all(c in df.columns for c in [file_col, age_col, gender_col, race_col]):
        raise ValueError(f"Colonnes attendues absentes. Trouvées: {df.columns.tolist()}")

    df["age_bin"] = df[age_col].apply(_age_to_bin)
    df["gender"]  = df[gender_col].apply(_gender_to_idx)
    df["ethnicity"] = df[race_col].apply(_race_to_idx)

    df = df.dropna(subset=["age_bin","gender","ethnicity"]).copy()
    df[["age_bin","gender","ethnicity"]] = df[["age_bin","gender","ethnicity"]].astype(int)

    # fichier relatif -> chemin absolu (img_root / file)
    df["path"] = df[file_col].astype(str).apply(lambda p: os.path.join(img_root, p))

    # splits stratifiés
    df["strat"] = df["age_bin"].astype(str)+"_"+df["gender"].astype(str)+"_"+df["ethnicity"].astype(str)
    train_df, tmp_df = train_test_split(df, test_size=(val_size+test_size), random_state=seed, stratify=df["strat"])
    rel_val = val_size/(val_size+test_size)
    val_df, test_df = train_test_split(tmp_df, test_size=(1-rel_val), random_state=seed, stratify=tmp_df["strat"])

    train_df["split"]="train"; val_df["split"]="val"; test_df["split"]="test"
    out = pd.concat([train_df,val_df,test_df], axis=0)[["path","age_bin","gender","ethnicity","split"]].reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv
