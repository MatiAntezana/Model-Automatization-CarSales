from src.utils.module_loader import load_module_from_path


def get_data_split(features, data_config, feature_config, split_name, initial_index):
    """Build one shuffled split and apply the configured split-level transformations."""
    split_ratios = data_config.split_ratios
    dataset_size = len(features)

    split_size = int(dataset_size * split_ratios[split_name])
    if split_name == "test":
        final_index = dataset_size
    else:
        final_index = min(initial_index + split_size, dataset_size)

    final_split = features.iloc[initial_index:final_index].copy()
    final_split = final_split.sample(frac=1, random_state=data_config.random_seed).reset_index(drop=True)

    transformer_module = load_module_from_path(f"models/{feature_config.data_transformer_file}")
    transformed_split = transformer_module.data_modify(feature_config, final_split)

    return transformed_split, final_index
