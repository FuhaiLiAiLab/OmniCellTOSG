import os
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

def count_and_cluster_cell_types(folder_path, output_counts, output_clusters):
    cell_type_counter = Counter()

    for file_name in os.listdir(folder_path):
        if file_name.endswith("_cell_type.csv"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            if 'Cell Types' in df.columns:
                for cell_types in df['Cell Types'].dropna():
                    for cell_type in map(str.strip, cell_types.split(';')):
                        cell_type_counter[cell_type] += 1

    sorted_counts = sorted(cell_type_counter.items(), key=lambda x: x[1], reverse=True)
    unique_cell_df = pd.DataFrame(sorted_counts, columns=['Cell Type', 'Count'])
    unique_cell_df.to_csv(output_counts, index=False)

    def clean_cell_type(cell_type):
        cell_type = cell_type.lower().strip()
        cell_type = re.sub(r'[_\-+,/]', ' ', cell_type)
        cell_type = re.sub(r'\s+', ' ', cell_type)
        cell_type = re.sub(r'(\bcd\d+)\+', r'\1', cell_type)
        return cell_type

    unique_cell_df['cell_type_clean'] = unique_cell_df['Cell Type'].apply(clean_cell_type)

    priority_clusters = {
        # Unannotated/Unknown/Unclassified assigned to 0
        r'\bunannote\b': 0, r'\bunknown\b': 0, r'\bunclassified\b': 0, r'\bunspecified\b': 0, r'\bunannotated\b': 0,
        r'\bunannoted\b': 0,

        # B cells
        r'\bB cell\b': 1, r'\bB\b': 1, r'^B-': 1, r'^B_': 1,
        r'\bCD19\b': 1, r'\bCD20\b': 1,

        # T cells
        r'\bT cell\b': 2, r'\bT\b': 2, r'^T-': 2, r'^T_': 2,
        r'\bCD3\b': 2, r'\bCD4\b': 2, r'\bCD8\b': 2,

        # Monocytes & Macrophages
        r'\bmonocyte\b': 3, r'\bmacrophage\b': 3, r'\bmoDC\b': 3, r'\bmonocytes\b': 3,
        r'\bmononuclear\b': 3, r'\bMo\b': 3, r'\bmono\b': 3, 

        # NK cells
        r'\bNK cell\b': 4, r'\bnatural killer\b': 4, r'\bNK\b': 4, r'\bNKT\b': 4, 

        # Dendritic cells
        r'\bdendritic cell\b': 5, r'\bDC\b': 5, r'plasmacytoid DC': 5,

        # Stem cells
        r'\bstem cell\b': 6, r'\bHSC\b': 6, r'hematopoietic': 6,

        # Granulocytes
        r'\bgranulocyte\b': 7, r'\bneutrophil\b': 7, r'eosinophil': 7, r'basophil': 7,

        # Oligodendrocytes & Precursors (Cluster 8)
        r'\bOligo\b': 8, r'\boligodendrocyte precursor cell\b': 8, 
        r'\boligodendrocyte precursor\b': 8, r'\bOPC\b': 8, 
        r'\bOligodendrocyte\b': 8,

        # Endothelial cells (Cluster 9)
        r'\bEndo\b': 9, r'\bendothelial cell\b': 9, 
        r'\bcapillary endothelial cell\b': 9, r'\bendothelium\b': 9,

        # Inhibitory Neurons (Cluster 10)
        r'\bInN\b': 10, r'\binhibitory neuron\b': 10, 
        r'\bInhibitory\b': 10, r'\bPVALB\b': 10, r'\bSST\b': 10, 
        r'\bVIP\b': 10, r'\bGABAergic\b': 10, r'\bLAMP5\b': 10,

        # Astrocytes (Cluster 11)
        r'\bAstro\b': 11, r'\bAstrocyte\b': 11, 
        r'\bastroglia\b': 11, r'\bAQP4\b': 11, r'\bSLC1A2\b': 11,

        # Microglia (Cluster 12)
        r'\bmicro\b': 12, r'\bmicroglia\b': 12, 
        r'\bmicroglial\b': 12, r'\bP2RY12\b': 12, r'\bAPBB1IP\b': 12,

        # Excitatory Neurons (Cluster 13)
        r'\bCUX2\b': 13, r'\bL2-3\b': 13,

        # Intratelencephalic Projection Neurons (Cluster 14)
        r'\bIT\b': 14, r'\bintratelencephalic\b': 14,

        # Corticospinal Projection Neurons (Cluster 15)
        r'\bFEZF2\b': 15,

        # Corticothalamic Projection Neurons (Cluster 16)
        r'\bCT\b': 16, r'\bcorticothalamic\b': 16,

        # OPRK1 (Cluster 17)
        r'\bOPRK1\b': 17,

        # muscle cells (Cluster 18)
        r'\bmuscle\b': 18, r'\bmuscle cell\b': 18,

        # RORB (Cluster 19)
        r'\bRORB\b': 19,

        # VLMC
        r'\bVLMC\b': 20,

        # lymphoid cell
        r'\blymphoid cell\b': 21, r'\bILC1\b': 21,
    }

    def assign_priority_cluster(cell_type):
        for pattern, cluster_id in priority_clusters.items():
            if re.search(pattern, cell_type, re.IGNORECASE):
                return cluster_id
        return -1

    unique_cell_df['priority_cluster'] = unique_cell_df['cell_type_clean'].apply(assign_priority_cluster)

    priority_df = unique_cell_df[unique_cell_df['priority_cluster'] >= 0]
    non_priority_df = unique_cell_df[unique_cell_df['priority_cluster'] == -1]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(non_priority_df['cell_type_clean'])

    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        linkage='ward',
        distance_threshold=1.5
    )

    labels = clustering_model.fit_predict(tfidf_matrix.toarray())
    non_priority_df['cluster'] = labels + max(priority_clusters.values()) + 1

    final_df = pd.concat([
        priority_df[['Cell Type', 'priority_cluster']].rename(columns={'priority_cluster': 'cluster'}),
        non_priority_df[['Cell Type', 'cluster']]
    ], ignore_index=True)

    grouped_clusters = final_df.groupby('cluster')['Cell Type'].apply(list).reset_index()
    grouped_clusters.rename(columns={'Cell Type': 'group_values'}, inplace=True)

    default_unknown_entries = ['unannote', 'unannotated', 'unknown', 'unclassified', 'unspecified']
    if 0 not in grouped_clusters['cluster'].values:
        grouped_clusters = pd.concat([
            pd.DataFrame({'cluster': [0], 'group_values': [default_unknown_entries]}),
            grouped_clusters
        ], ignore_index=True)
    else:
        grouped_clusters.loc[grouped_clusters['cluster'] == 0, 'group_values'] = \
            grouped_clusters.loc[grouped_clusters['cluster'] == 0, 'group_values'].apply(lambda x: list(set(x + default_unknown_entries)))

    grouped_clusters.to_csv(output_clusters, index=False)

if __name__ == "__main__":
    folder_path = "file_count_result/cell_type"
    output_counts = "file_count_result/cell_type/cell_type_counts.csv"
    output_clusters = "file_count_result/cell_type/grouped_cell_types.csv"
    count_and_cluster_cell_types(folder_path, output_counts, output_clusters)
