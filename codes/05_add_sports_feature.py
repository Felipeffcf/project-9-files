#%%

import pandas as pd

LEARN = r"C:\Users\felif\Desktop\Machine Learning\FINAL PROJECT\project-9-files\raw_data\learn_dataset.csv"
TEST  = r"C:\Users\felif\Desktop\Machine Learning\FINAL PROJECT\project-9-files\raw_data\test_dataset.csv"

LEARN_SPORT = r"C:\Users\felif\Desktop\Machine Learning\FINAL PROJECT\project-9-files\raw_data\learn_dataset_sport.csv"
TEST_SPORT  = r"C:\Users\felif\Desktop\Machine Learning\FINAL PROJECT\project-9-files\raw_data\test_dataset_sport.csv"

OUT_LEARN = r"C:\Users\felif\Desktop\Machine Learning\FINAL PROJECT\project-9-files\raw_data\learn_plus_sport.csv"
OUT_TEST  = r"C:\Users\felif\Desktop\Machine Learning\FINAL PROJECT\project-9-files\raw_data\test_plus_sport.csv"

def add_sport_flag(core_path, sport_path, out_path):
    core = pd.read_csv(core_path, dtype={"INSEE": str})
    sport = pd.read_csv(sport_path)

    # People not listed are NOT members (per project statement)
    sport_uids = set(sport["uid"].unique())
    core["is_sport_member"] = core["uid"].isin(sport_uids).astype(int)

    core.to_csv(out_path, index=False)
    print("Saved:", out_path, "shape:", core.shape)

if __name__ == "__main__":
    add_sport_flag(LEARN, LEARN_SPORT, OUT_LEARN)
    add_sport_flag(TEST,  TEST_SPORT,  OUT_TEST)
#%%