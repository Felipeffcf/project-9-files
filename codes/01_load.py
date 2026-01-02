#%% 
import pandas as pd

learn = pd.read_csv(
    "../raw_data/learn_dataset.csv",
    dtype={"INSEE": str}
)

test = pd.read_csv(
    "../raw_data/test_dataset.csv",
    dtype={"INSEE": str}
)


print("Learn shape:", learn.shape)
print("Test shape :", test.shape)

print("uid unique (learn/test):", learn["uid"].is_unique, test["uid"].is_unique)
print("target in learn/test   :", "target" in learn.columns, "target" in test.columns)
print("INSEE dtype:", learn["INSEE"].dtype)

#%%
