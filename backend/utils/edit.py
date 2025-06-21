import pandas as pd

# Load your Excel file
df = pd.read_excel('dtwl_classified.xlsx')  # Load the file with the classified DTWL

# Count the occurrences of each class in the 'DTWL_Class' column
class_counts = df['DTWL_Class'].value_counts()

# Print the counts
print(f"Low class count: {class_counts.get('Low', 0)}")
print(f"Medium class count: {class_counts.get('Medium', 0)}")
print(f"High class count: {class_counts.get('High', 0)}")
