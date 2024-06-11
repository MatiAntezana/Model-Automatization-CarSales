import logging
from src.utils.trasform_ruth_module import modularization_ruth


def get_dataloader(features, param_sets, features_used, type_set, initial_index):
    size_dataset = len(features)

    size_set = int(size_dataset * param_sets[type_set])
    final_index = initial_index + size_set

    dif_index_size = (size_dataset - 1) - final_index

    if dif_index_size < 0:
        final_index += dif_index_size
        logging.info(f"Se realizó la resta del indice en get_dataloader: {final_index}")
    final_set = features.iloc[initial_index:final_index]

    file_modify_data = modularization_ruth(f"models/{features_used.file_modify_data}")

    file_modify_data.data_modify(features_used, final_set)

    return final_set, final_index