import argparse
from .prepare_fairface import build_from_existing_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", type=str, required=True, help="Dossier racine des images, ex: /.../data")
    ap.add_argument("--csv_in", type=str, required=True, help="ethnicity_labels_filtered.csv")
    ap.add_argument("--out_csv", type=str, required=True, help="Chemin de sortie attrs.csv")
    ap.add_argument("--file_col", type=str, default="file")
    ap.add_argument("--age_col", type=str, default="age")
    ap.add_argument("--gender_col", type=str, default="gender")
    ap.add_argument("--race_col", type=str, default="race")
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = build_from_existing_csv(
        img_root=args.img_root,
        csv_in=args.csv_in,
        out_csv=args.out_csv,
        file_col=args.file_col,
        age_col=args.age_col,
        gender_col=args.gender_col,
        race_col=args.race_col,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print("Wrote:", out)

if __name__ == "__main__":
    main()
