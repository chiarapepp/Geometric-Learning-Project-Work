from normalizers.minmax import minmax_normalizer
from normalizers.meanvar import meanvar_normalizer

def get_normalizer(normalizer_name):
    """
    Factory function to get the appropriate normalizer based on the name provided.
    Args:
        normalizer_name (str): Name of the normalizer to be used.
    """
    if normalizer_name == 'minmax':
        return minmax_normalizer
    elif normalizer_name == 'meanvar':
        return meanvar_normalizer
    else:
        raise NotImplementedError(f'Normalizer {normalizer_name} not implemented')