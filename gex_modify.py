import pandas as pd
import numpy as np

def process_gex_data(file_path, output_path='filtered_gex_matrix.csv'):
    print("1. Loading Gene Expression Data...")
    # Load data
    df = pd.read_csv(file_path, index_col=0)
    
    print(f"   Initial shape: {df.shape}")

    # 2. Check orientation
    if df.shape[0] > df.shape[1]:
        print("   Transposing matrix (flipping Genes x Cells -> Cells x Genes)...")
        df = df.T

    # 3. Clean Index (Cell Line Names)
    df.index = df.index.str.upper().str.strip()
    df.index.name = 'cell_line_name'

    # 4. Handle Specific Missing Lines (Cleanup)
    # The documentation noted MDA-MB-175-VII and NCI-H1437 are missing.
    # If they exist as empty rows, drop them. If they don't exist, we are fine.
    df = df.dropna(how='all')

    # 5. Rename Columns
    # Example: "TP53" -> "TP53_GEX"
    # This prevents confusion if you have a TP53 mutation column and a TP53 expression column.
    df.columns = [f"{str(col)}_GEX" for col in df.columns]

    # 6. Reset Index for merging
    df.reset_index(inplace=True)

    print(f"   Final Matrix Shape: {df.shape} (Cells x Features)")

    # 7. Save
    df.to_csv(output_path, index=False)
    print(f"   Successfully saved processed data to: {output_path}")


process_gex_data('data_preprocessing/Sanger_molecular_data/gex.csv')
