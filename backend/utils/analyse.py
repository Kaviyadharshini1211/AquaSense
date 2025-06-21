import pandas as pd

# Load your Excel file correctly
df = pd.read_excel('groundwater2024.xlsx')  # not read_csv!

# Create 3 classes (tertiles) based on DTWL
df['DTWL_Class'], bins = pd.qcut(df['DTWL'], q=3, labels=['Low', 'Medium', 'High'], retbins=True)

# Display the DTWL bin ranges for each class
print(f"Low DTWL Range: {bins[0]} to {bins[1]}")
print(f"Medium DTWL Range: {bins[1]} to {bins[2]}")
print(f"High DTWL Range: {bins[2]} to {bins[3]}")

# Save the modified dataframe to a new Excel file
df.to_excel('dtwl_classified.xlsx', index=False)

print("Excel file created: dtwl_classified.xlsx with DTWL_Class column.")
