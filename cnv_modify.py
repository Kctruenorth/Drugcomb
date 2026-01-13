import pandas as pd

def process_cnv_data(gene_file_path, segment_file_path):
    # This prevents the "Mixed Types" warning for chromosomes
    gene_dtypes = {
        'chr_GRCh38': str,
        'chr_GRCh37': str,
        'cell_line_name': str,
        'gene': str
    }
    
    # We also do it for the segment file to be safe
    segment_dtypes = {
        'chr_38': str,
        'chr_37': str,
        'cell_line_name': str
    }

    # Load with the dtype parameter
    df_gene = pd.read_csv(gene_file_path, dtype=gene_dtypes)
    df_segment = pd.read_csv(segment_file_path, dtype=segment_dtypes)


    # ---------------------------------------------------------
    # PART A: Gene Level Processing
    # ---------------------------------------------------------
    
    # 1. Filter out Y Chromosome (Artifact mitigation)
    # Convert column to string just in case it's read as distinct types
    df_gene['chr_GRCh38'] = df_gene['chr_GRCh38'].astype(str)
    df_gene_clean = df_gene[df_gene['chr_GRCh38'] != 'Y'].copy()


    # 2. CRITICAL FIX: Filter out -1 (Undetermined) values
    # We ensure the column is numeric first, coercing errors to NaN
    df_gene_clean['min_cn_GRCh38'] = pd.to_numeric(df_gene_clean['min_cn_GRCh38'], errors='coerce')
    
    # Remove rows where min_cn is -1 OR NaN
    df_gene_clean = df_gene_clean[df_gene_clean['min_cn_GRCh38'] != -1]
    df_gene_clean = df_gene_clean.dropna(subset=['min_cn_GRCh38'])

    # 2. Define Classification Logic
    #   - Amplification: >= 8 copies (using min_cn to be conservative that the whole gene is present)
    #   - Deletion: Homozygous only (min_cn == 0)
    def classify_gene(row):
        min_cn = row['min_cn_GRCh38']
        
        if min_cn == 0:
            return 'Deletion'
        elif min_cn >= 8:
            return 'Amplification'
        else:
            return 'Wild Type'

    # Apply classification
    df_gene_clean['cnv_call'] = df_gene_clean.apply(classify_gene, axis=1)

    # ---------------------------------------------------------
    # PART B: Segment Level Processing
    # ---------------------------------------------------------
    
    # 1. Filter out Y Chromosome
    df_segment['chr_38'] = df_segment['chr_38'].astype(str)
    df_segment_clean = df_segment[df_segment['chr_38'] != 'Y'].copy()

    # 2. Define Segment Logic
    #   - Uses totalCN column
    def classify_segment(row):
        total_cn = row['totalCN']
        
        if total_cn == 0:
            return 'Deletion'
        elif total_cn >= 8:
            return 'Amplification'
        else:
            return 'Neutral'

    df_segment_clean['segment_call'] = df_segment_clean.apply(classify_segment, axis=1)

    return df_gene_clean, df_segment_clean

# --- Usage Example ---
# Replace with your actual file paths
gene_file = 'data_preprocessing/Sanger_molecular_data/cnv/cnv_gene.csv'
segment_file = 'data_preprocessing/Sanger_molecular_data/cnv/cnv_segment.csv'

# Execute processing
processed_genes, processed_segments = process_cnv_data(gene_file, segment_file)

df = processed_genes

# 2. Pivot the table
# Index = Cell Line
# Columns = Gene Names
# Values = Minimum Copy Number (Raw numeric value is best for XGBoost)
cnv_matrix = df.pivot(index='cell_line_name', columns='gene', values='min_cn_GRCh38')

# 3. Handle Missing Data
# If a gene/cell combo is missing (or was filtered out), we assume it is Wild Type (2 copies)
cnv_matrix = cnv_matrix.fillna(2.0)

# 4. Rename Columns to avoid collision with Mutation data
# e.g., "TP53" becomes "TP53_CNV"
cnv_matrix.columns = [f"{str(col)}_CNV" for col in cnv_matrix.columns]

# 5. Reset index so 'cell_line_name' becomes a column again
cnv_matrix.reset_index(inplace=True)
#Save to new CSVs
cnv_matrix.to_csv('filtered_cnv_matrix.csv', index=False)