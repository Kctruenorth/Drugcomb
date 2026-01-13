import pandas as pd

def process_mutation_data(file_path, output_path='filtered_mutation_matrix.csv'):
    # 1. Load the dataset
    df = pd.read_csv(file_path, sep=',') 
    print(f"Initial shape: {df.shape}")

    # 2. Filter: Discard Noise (SNPs) & Keep Somatic
    snp_mask = df['SNP'] != 'y'
    valid_status = [
        'Confirmed somatic variant', 
        'Reported in another cancer sample as somatic',
        'Variant of unknown origin' 
    ]
    status_mask = df['Mutation.somatic.status'].isin(valid_status)
    df_clean = df[snp_mask & status_mask].copy()

    # 2.5 CLEAN GENE NAMES
    # Keep only the part before the underscore (e.g. 'A1CF_ENST...' -> 'A1CF')
    df_clean['Gene.name'] = df_clean['Gene.name'].astype(str).apply(lambda x: x.split('_')[0])

    # 3. Binarisation (Create the Matrix)
    binary_matrix = pd.pivot_table(
        df_clean, 
        index='cell_line_name', 
        columns='Gene.name', 
        values='Mutation.ID', 
        aggfunc='count' # Count how many mutations exist per gene/cell
    )

    # Convert counts to binary (1 if mutation exists, 0 otherwise)
    binary_matrix = binary_matrix.notnull().astype(int)
    
    # Fill NaNs with 0 (Wild Type)
    binary_matrix = binary_matrix.fillna(0)

    # ---------------------------------------------------------
    # NEW STEP: Rename Columns and Reset Index
    # ---------------------------------------------------------
    
    # 1. Rename columns to include suffix (e.g., "TP53" -> "TP53_MUT")
    binary_matrix.columns = [f"{col}_MUT" for col in binary_matrix.columns]
    
    # 2. Reset index so 'cell_line_name' becomes a standard column (matches your CNV format)
    binary_matrix.reset_index(inplace=True)

    print(f"Final Matrix Shape: {binary_matrix.shape}")

    # 4. Save to CSV
    # set index=False because cell_line_name is now a column
    binary_matrix.to_csv(output_path, index=False) 
    print(f"Successfully saved filtered data to: {output_path}")
    
    return binary_matrix

# Run the function
mutation_matrix = process_mutation_data('data_preprocessing/Sanger_molecular_data/mutations.csv')
print(mutation_matrix.head())