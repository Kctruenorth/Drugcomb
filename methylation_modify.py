import pandas as pd
import numpy as np

def process_methylation_data(file_path, output_path='filtered_methylation_matrix.csv'):
    # 1. Load the dataset
    # Assuming the file is formatted with Genes as rows and Cell Lines as columns
    # (This is standard for Sanger/GDSC data).
    df = pd.read_csv(file_path, index_col=0) 
    print(f"Initial shape: {df.shape}")

    # 2. Transpose
    # We need Cell Lines as rows and Genes as columns
    df = df.T
    
    # 3. Clean Index (Cell Line Names)
    df.index = df.index.str.upper().str.strip()
    df.index.name = 'cell_line_name'

    # 4. Handle Missing Values (Imputation)
    # Unlike mutations, we can't assume NaN = 0.
    # We fill NaNs with the MEAN of that column (gene).
    df = df.fillna(df.mean())

    # 5. Rename Columns
    # Add _METH suffix to distinguish from _MUT and _CNV
    df.columns = [f"{col}_METH" for col in df.columns]

    # 6. Reset Index to make cell_line_name a column
    df.reset_index(inplace=True)

    print(f"Final Matrix Shape: {df.shape}")

    # 7. Save
    df.to_csv(output_path, index=False)
    print(f"Successfully saved to: {output_path}")
    
    return df

# Run the function
# Make sure to point to the CpG ISLANDS file (the smaller one)
methylation_matrix = process_methylation_data('Sanger_molecular_data/methylation_islands.csv')
print(methylation_matrix.head())