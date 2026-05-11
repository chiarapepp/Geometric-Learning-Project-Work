import torch
import numpy as np
import torch.nn as nn



class Normalizer(nn.Module):
    """
    Normalizer class for normalizing input data.
    """
    def __init__(self, args):
        super(Normalizer, self).__init__()
        self.w = args.frame_dim[0]
        self.h = args.frame_dim[1]
        self.min_values = torch.tensor([0, 0, 0, 0]) #changing in train script based on dataset
        self.max_values = torch.tensor([self.w, self.h, 1, 1]) #changing in train script based on dataset
        self.mean = torch.tensor([self.w/2, self.h/2, 0.5])
        self.std = torch.tensor([self.w/2, self.h/2, 0.5])
        self.normalize_type = args.normalize_type

        if self.normalize_type == 'zero_one':
            self.func_normalize = self.normalize_values_zero_one
            self.func_un_normalize = self.un_normalize_values_zero_one_global
        elif self.normalize_type == 'inplace_normalize_values_zero_one':
            self.func_normalize = self.inplace_normalize_values_zero_one
        elif self.normalize_type == 'zero_one_global':
            self.func_normalize = self.normalize_values_zero_one
            self.func_un_normalize = self.un_normalize_values_zero_one_global
        elif self.normalize_type == 'mean_variance':
            self.func_normalize = self.normalize_values_mean_variance
        elif self.normalize_type == 'mean_variance_local':
            self.func_normalize = self.normalize_values_mean_variance_local
        elif self.normalize_type == 'nothing':
            self.func_normalize = self.normalize_values_nothing
        elif self.normalize_type == 'inplace_normalizer':
            self.func_normalize = self.inplace_normalizer
        elif self.normalize_type == 'heterogeneous':
            self.func_normalize = self.normalize_values_divide
        else:
            raise NotImplementedError
        
    #todo
    def forward(self, input):
        return self.func_normalize(input)

    @staticmethod
    def normalize_values_nothing(input):
            x = input[:, :, 0]
            y = input[:, :, 1]
            t = input[:, :, 2]
            p = input[:, :, 3]
            return x, y, t, p
    
    def normalize_values_zero_one(self, input):
            x = input[:, :, 0]
            y = input[:, :, 1]
            t = input[:, :, 2]
            p = input[:, :, 3]
            max_t = t.max(dim=1)[0][:,None]
            x = x / self.w
            y = y / self.h
            t = t / max_t
            p = p / 1
            return x, y, t, p
    
    def normalize_values_divide(self, input):
            x = input[:, :, 0]
            y = input[:, :, 1]
            t = input[:, :, 2]
            p = input[:, :, 3]
            max_t = t.max(dim=1)[0][:,None]//10
            x = x / self.w
            y = y / self.h
            t = t / max_t
            p = p / 1
            return x, y, t, p
    
    def normalize_values_zero_one_shift(self, input):
            x = input[:, :, 0]
            y = input[:, :, 1]
            t = input[:, :, 2]
            p = input[:, :, 3]
            max_t = t.max(dim=1)[0][:,None]
            x = x / self.w
            y = y / self.h
            t = t / max_t
            p = p / 1
            return x, y, t, p

    def normalize_values_zero_one_global(self, input):
            x = input[:, :, 0]
            y = input[:, :, 1]
            t = input[:, :, 2]
            p = input[:, :, 3]
            x = x / self.max_values[0]
            y = y / self.max_values[1]
            t = t / self.max_values[2]
            p = p / 1
            return x, y, t, p

    def un_normalize_values_zero_one_global(self, input):
            x = input[:, :, 0]
            y = input[:, :, 1]
            t = input[:, :, 2]
            p = input[:, :, 3]
            x = x * self.max_values[0]
            y = y * self.max_values[1]
            t = t * self.max_values[2]
            p = p * 1
            return x, y, t, p
    
    def normalize_values_mean_variance(self, input):
        x = input[:, :, 0]
        y = input[:, :, 1]
        t = input[:, :, 2]
        p = input[:, :, 3]
        x = (x - self.mean[0]) / self.std[0]
        y = (y - self.mean[1]) / self.std[1]
        t = (t - self.mean[2]) / self.std[2]
        p = (p -0.5) / 1
        return x, y, t, p
    
    @staticmethod
    def normalize_values_mean_variance_local(input):
        x = input[:, :, 0]
        y = input[:, :, 1]
        t = input[:, :, 2]
        p = input[:, :, 3]
        x = (x - x.mean(dim=1)[:,None]) / x.std(dim=1)[:,None]
        y = (y - y.mean(dim=1)[:,None]) / y.std(dim=1)[:,None]
        t = (t - t.mean(dim=1)[:,None]) / t.std(dim=1)[:,None]
        p = (p - p.mean(dim=1)[:,None]) / p.std(dim=1)[:,None]

        return x, y, t, p
    
    @staticmethod
    def unnorm_values_mean_variance(input, gt):
        x = input[:, :, 0]
        y = input[:, :, 1]
        t = input[:, :, 2]
        p = input[:, :, 3]
        x = x * gt.std(dim=1)[:,None] + gt.mean(dim=1)[:,None]
        y = y * gt.std(dim=1)[:,None] + gt.mean(dim=1)[:,None]
        t = t * gt.std(dim=1)[:,None] + gt.mean(dim=1)[:,None]
        return x, y, t, p
            
    @staticmethod
    def inplace_normalizer(input):
        means = input.mean(dim=1)
        stds = input.std(dim=1)
        input = (input - means[:,None]) / stds[:,None]
        return input, None, None, None


if __name__ == "__main__":
    example_tensor = torch.rand(8, 5000, 4)

    normalized = Normalizer.normalize_values_mean_variance_local(example_tensor)
    print(normalized)       





