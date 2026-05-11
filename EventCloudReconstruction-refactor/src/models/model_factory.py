from models.encoder.PointNet import PointNetEncoder
from models.encoder.an_encoder import AnEncoder
from models.decoder.linear_decoder import LinearDecoder
from models.decoder.mlp_decoder import MLPDecoder
import torch

def get_model(model_type, dim, latent_size):
    """
    Factory function to get the model based on the model type name.
    Returns an instance of the model class.
    """
    if model_type == 'PointNetEncoder':
        return PointNetEncoder(latent_size, dim)
    elif model_type == 'AnEncoder':
        return AnEncoder(latent_size, dim)
    elif model_type == 'LinearDecoder':
        return LinearDecoder(latent_size, dim)
    elif model_type == 'MLPDecoder':
        return MLPDecoder(latent_size, dim)
    else:
        raise NotImplementedError(f'Model {model_type} not implemented')
    

def build_model(model_list, dims, latent_size):
    """
    Builds a sequential model from a list of model names and their corresponding dimensions.
    Args:
        model_list (list): List of model names as strings.
        dims (list): List of dimensions corresponding to each model.
        latent_size (int): Size of the latent space for the models.
    """
    model = torch.nn.Sequential()
    for model_name, dim in zip(model_list, dims):
        model.add_module(model_name, get_model(model_name, dim, latent_size))
    return model