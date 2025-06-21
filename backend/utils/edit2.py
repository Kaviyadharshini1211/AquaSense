import pandas as pd

# how many top rows to remove
N = 5200

# load your file
df = pd.read_excel('groundwater.xlsx')

# Method 1: via iloc
df = df.iloc[N:].reset_index(drop=True)

# — or —

# Method 2: via drop
# df.drop(df.index[:N], inplace=True)
# df.reset_index(drop=True, inplace=True)

# now df has the top N rows removed
df.to_excel('groundwater_trimmed1.xlsx', index=False)
print(f"Done — {N} rows removed; {len(df)} remain.")
