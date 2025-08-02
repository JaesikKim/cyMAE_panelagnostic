import os
import numpy as np
import pandas as pd
import h5py
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


BLACK_LIST = [
    # not readable FCS
    # 'MDIPA_MESSI_994573_D0_Normalized.csv', 'MDIPA_MS_003_V3_Normalized_concat.csv', 'MDIPA_MS_065_V9_Normalized_concat.csv', 'MDIPA_MS_081_V8_Normalized_concat.csv',
    # QC failed
    'MDIPA_AALC_08_V1_Processed', 'MDIPA_AALC_20_V1_Processed', 'MDIPA_IHCV2020_009_T1_Normalized', 'MDIPA_ISPY_768450_Day7_Normalized', 'MDIPA_ISPY_803327_Day7_Normalized',
    'MDIPA_MS_HD_34_Normalized','MDIPA_MS_HD_32_Normalized','MDIPA_MS_HD_33_Normalized','MDIPA_MS_HD_35_Normalized','MDIPA_MS_HD_36_Normalized','MDIPA_MS_HD_37_Normalized',
    'MDIPA_MS_HD_38_Normalized','MDIPA_MS_HD_39_Normalized','MDIPA_MS_HD_40_Normalized','MDIPA_MS_HD_41_Normalized','MDIPA_MS_HD_42_Normalized','MDIPA_MS_HD_43_Normalized',
    'MDIPA_MS_HD_44_Normalized','MDIPA_MS_HD_45_Normalized','MDIPA_MS_HD_46_Normalized','MDIPA_PREPRO_HD_5_Processed',
    'MDIPA_MS_001_V13_Normalized','MDIPA_MS_001_V14_Normalized','MDIPA_MS_004_V8_Normalized','MDIPA_MS_004_V9_Normalized','MDIPA_MS_008_V6_Normalized','MDIPA_MS_010_V14_Normalized',
    'MDIPA_MS_012_V10_Normalized','MDIPA_MS_012_V13_Normalized','MDIPA_MS_012_V14_Normalized','MDIPA_MS_012_V9_Normalized','MDIPA_MS_016_V7_Normalized','MDIPA_MS_016_V8_Normalized',
    'MDIPA_MS_021_V10_Normalized','MDIPA_MS_022_V10_Normalized','MDIPA_MS_023_V10_Normalized','MDIPA_MS_023_V14_Normalized','MDIPA_MS_024_V10_Normalized','MDIPA_MS_025_V9_Normalized',
    'MDIPA_MS_026_V10_Normalized','MDIPA_MS_027_V9_Normalized','MDIPA_MS_029_V13_Normalized','MDIPA_MS_029_V14_Normalized','MDIPA_MS_031_V13_Normalized','MDIPA_MS_031_V14_Normalized',
    'MDIPA_MS_033_V10_Normalized','MDIPA_MS_033_V14_Normalized','MDIPA_MS_034_V10_Normalized','MDIPA_MS_034_V9_Normalized','MDIPA_MS_035_V9_Normalized','MDIPA_MS_036_V10_Normalized',
    'MDIPA_MS_036_V13_Normalized','MDIPA_MS_038_V10_Normalized','MDIPA_MS_040_V10_Normalized','MDIPA_MS_042_V10_Normalized','MDIPA_MS_042_V13_Normalized','MDIPA_MS_043_V10_Normalized',
    'MDIPA_MS_043_V9_Normalized','MDIPA_MS_044_V8_Normalized','MDIPA_MS_045_V10_Normalized','MDIPA_MS_046_V9_Normalized','MDIPA_MS_047_V10_Normalized','MDIPA_MS_048_V10_Normalized',
    'MDIPA_MS_048_V9_Normalized','MDIPA_MS_049_V10_Normalized','MDIPA_MS_050_V10_Normalized','MDIPA_MS_052_V10_Normalized','MDIPA_MS_052_V9_Normalized','MDIPA_MS_054_V10_Normalized',
    'MDIPA_MS_055_V10_Normalized','MDIPA_MS_056_V9_Normalized','MDIPA_MS_057_V9_Normalized','MDIPA_MS_058_V8_Normalized','MDIPA_MS_058_V9_Normalized','MDIPA_MS_059_V10_Normalized',
    'MDIPA_MS_059_V13_Normalized','MDIPA_MS_060_V6_Normalized','MDIPA_MS_060_V8_Normalized','MDIPA_MS_065_V9_Normalized','MDIPA_MS_066_V9_Normalized','MDIPA_MS_068_V8_Normalized',
    'MDIPA_MS_069_V10_Normalized','MDIPA_MS_069_V9_Normalized','MDIPA_MS_070_V9_Normalized','MDIPA_MS_071_V6_Normalized','MDIPA_MS_071_V7_Normalized','MDIPA_MS_071_V8_Normalized',
    'MDIPA_MS_072_V9_Normalized','MDIPA_MS_073_V8_Normalized','MDIPA_MS_073_V9_Normalized','MDIPA_MS_074_V9_Normalized','MDIPA_MS_076_V8_Normalized','MDIPA_MS_076_V9_Normalized',
    'MDIPA_MS_077_V9_Normalized','MDIPA_MS_078_V6_Normalized','MDIPA_MS_078_V8_Normalized','MDIPA_MS_078_V9_Normalized','MDIPA_MS_079_V8_Normalized','MDIPA_MS_079_V9_Normalized',
    'MDIPA_MS_080_V6_Normalized','MDIPA_MS_080_V7_Normalized','MDIPA_MS_080_V8_Normalized','MDIPA_MS_081_V6_Normalized','MDIPA_MS_081_V8_Normalized','MDIPA_MS_082_V6_Normalized',
    'MDIPA_MS_082_V8_Normalized','MDIPA_MS_083_V6_Normalized','MDIPA_MS_083_V8_Normalized','MDIPA_MS_084_V6_Normalized','MDIPA_MS_084_V8_Normalized','MDIPA_MS_085_V6_Normalized','MDIPA_MS_085_V8_Normalized'
]

# NO_QCfile_LIST = ['MDIPA_MESSI_221270_FreshDay0_Normalized.csv', 'MDIPA_MESSI_994724_D0_Normalized.csv']


# cell_types = ['basophil' 'bcell' 'eosinophil' 'mdc' 'monocyte_classical'
#  'monocyte_nonclassical' 'neutrophil' 'nkcell' 'pdc' 'plasmablast'
#  'tcell_cd4' 'tcell_cd8' 'tcell_dn' 'tcell_dp' 'tcell_gd']


def count_lines_in_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        line_count = sum(1 for line in file)-1 # excluding header
    return line_count

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_csv_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, 'csv')

def is_fcs_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, 'fcs')


def read_file(data_file_path):
    # import time
    # start_time = time.time()

    with h5py.File(data_file_path, 'r') as h5file:
        # Load features and cell types
        data = torch.tensor(h5file['data'][:])
        labels = [ct.decode('utf-8') for ct in h5file['labels'][:]]
        marker_list = [ct.decode('utf-8') for ct in h5file['marker_list'][:]]

        # Create mask to filter out uncertain cells
        certain_mask = torch.tensor([ct != "Uncertain" for ct in labels])
        
        # Filter features and cell types
        data = data[certain_mask]
        labels = [ct for ct, mask in zip(labels, certain_mask) if mask]
    # print(time.time()-start_time)    
    return data, labels, marker_list


def simple_label_mapper(labels):
    cell_group_map = {
        # B Cells
        "Plasmablast": "Plasmablast",

        "IgDposMemB": "Mem B",
        "IgDnegMemB": "Mem B",

        "NaiveB": "NaiveB",

        # CD4+ T Cells
        "Th2/activated": "CD4+ T",
        "Treg/activated": "CD4+ T",
        "Treg": "CD4+ T",
        "CD4Naive": "CD4+ T",
        "Th2": "CD4+ T",
        "Th17": "CD4+ T",
        "nnCD4CXCR5pos/activated": "CD4+ T",
        "Th1": "CD4+ T",
        "Th1/activated": "CD4+ T",
        "CD4Naive/activated": "CD4+ T",
        "Th17/activated": "CD4+ T",
        "nnCD4CXCR5pos": "CD4+ T",

        # CD8+ T Cells
        "CD8Naive": "CD8+ T",
        "CD8TEM2": "CD8+ T",
        "CD8Naive/activated": "CD8+ T",
        "CD8TEMRA/activated": "CD8+ T",
        "CD8TEM3/activated": "CD8+ T",
        "CD8TEM2/activated": "CD8+ T",
        "CD8TEM1/activated": "CD8+ T",
        "CD8TEMRA": "CD8+ T",
        "CD8TCM/activated": "CD8+ T",
        "CD8TEM1": "CD8+ T",
        "CD8TEM3": "CD8+ T",
        "CD8TCM": "CD8+ T",

        # Other T Cells
        "DPT": "Other T",
        "MAITNKT": "Other T",
        "gdT": "Other T",
        "DNT": "Other T",
        "DNT/activated": "Other T",
        "DPT/activated": "Other T",

        # NK & ILC
        "EarlyNK": "NK & ILC",
        "LateNK": "NK & ILC",
        "ILC": "NK & ILC",

        # Monocytes & Dendritic Cells
        "pDC": "Dendritic Cell",
        "mDC": "Dendritic Cell",
        "ClassicalMono": "Monocyte",
        "TotalMonocyte": "Monocyte",

        # Granulocytes
        "CD66bnegCD45lo": "Other Granulocyte", # Or Other/Debris
        "CD45hiCD66bpos": "Other Granulocyte",

        "Basophil": "Basophil",
        "Eosinophil": "Eosinophil",
        "Neutrophil": "Neutrophil",
    }

    return [cell_group_map[l] if l in cell_group_map.keys() else l for l in labels]

class CyTOFDataset(Dataset):
    def __init__(
            self,
            union_marker_list: list,
            data_path: str,
            is_perm: bool = True,
            seed: int = 0, 
            transform: Optional[Callable] = None
    ):
        super(CyTOFDataset, self).__init__()
        self.union_marker_list = union_marker_list
        self.union_marker_to_index = {
            marker: idx for idx, marker in enumerate(self.union_marker_list)
        }
        self.transform = transform
        self.rng = np.random.default_rng(seed)
        self.is_perm = is_perm

        self.filenames, self.data_paths = self.find_h5(data_path)
        print("Total num of samples:", len(self.data_paths))


    def find_h5(self, data_path):

        data_paths = []
        filenames = []
        # Iterate through all .h5 files in data_path
        for filename in os.listdir(data_path):
            if filename.endswith('.h5'):
                sample_name = filename.replace('.h5', '')
                if sample_name not in BLACK_LIST:
                    data_paths.append(os.path.join(data_path, filename))
                    filenames.append(filename)
        return filenames, data_paths

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data_file_path = self.data_paths[index]
        return data_file_path

    def __len__(self) -> int:
        return len(self.data_paths)
    
    def custom_collate_fn(self, batch):
        batched_cells = []
        batched_labels = []
        batched_marker_indices = []
        for data_file_path in batch:
            cells, labels, marker_list = read_file(data_file_path)
            # protein index
            marker_indices = []
            for marker in marker_list:
                if marker in self.union_marker_list:
                    marker_indices.append(self.union_marker_to_index[marker])
                else:
                    raise ValueError(f"Marker '{marker}' not found in union_marker_list")

            tensor_cells  = torch.tensor(cells)         # [N_cells, ...]

            if self.is_perm:
                perm = self.rng.permutation(tensor_cells.size(0))

                tensor_cells  = tensor_cells[perm]
                labels = [labels[i] for i in perm]

            batched_cells.append(tensor_cells)
            batched_labels.append(labels)
            batched_marker_indices.append(marker_indices)

        return torch.stack(batched_cells, dim=0), batched_labels, batched_marker_indices

