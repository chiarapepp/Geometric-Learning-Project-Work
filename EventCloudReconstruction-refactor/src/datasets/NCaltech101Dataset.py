from datasets.base_datasets import DatasetNpy
import os
from glob import glob
import numpy as np
import utils.data_utils as duts


class NCaltech101Dataset(DatasetNpy):
    def __init__(self, data_path, train, data_dim=(240, 320, 2), N=-1, stride=-1, use_polarity=True, transform=None,
                 normalizer=None):
        super(NCaltech101Dataset, self).__init__(data_path, train, data_dim, N, stride, use_polarity, transform,
                                                 normalizer)

        self.classes = {
            "accordion": 0, "airplanes": 1, "anchor": 2, "ant": 3, "BACKGROUND_Google": 4, "barrel": 5,
            "bass": 6, "beaver": 7, "binocular": 8, "bonsai": 9, "brain": 10, "brontosaurus": 11,
            "buddha": 12, "butterfly": 13, "camera": 14, "cannon": 15, "car_side": 16, "ceiling_fan": 17,
            "cellphone": 18, "chair": 19, "chandelier": 20, "cougar_body": 21, "cougar_face": 22,
            "crab": 23, "crayfish": 24, "crocodile": 25, "crocodile_head": 26, "cup": 27, "dalmatian": 28,
            "dollar_bill": 29, "dolphin": 30, "dragonfly": 31, "electric_guitar": 32, "elephant": 33,
            "emu": 34, "euphonium": 35, "ewer": 36, "Faces_easy": 37, "ferry": 38, "flamingo": 39,
            "flamingo_head": 40, "garfield": 41, "gerenuk": 42, "gramophone": 43, "grand_piano": 44,
            "hawksbill": 45, "headphone": 46, "hedgehog": 47, "helicopter": 48, "ibis": 49, "inline_skate": 50,
            "joshua_tree": 51, "kangaroo": 52, "ketch": 53, "lamp": 54, "laptop": 55, "Leopards": 56,
            "llama": 57, "lobster": 58, "lotus": 59, "mandolin": 60, "mayfly": 61, "menorah": 62,
            "metronome": 63, "minaret": 64, "Motorbikes": 65, "nautilus": 66, "octopus": 67, "okapi": 68,
            "pagoda": 69, "panda": 70, "pigeon": 71, "pizza": 72, "platypus": 73, "pyramid": 74,
            "revolver": 75, "rhino": 76, "rooster": 77, "saxophone": 78, "schooner": 79, "scissors": 80,
            "scorpion": 81, "sea_horse": 82, "soccer_ball": 83, "snoopy": 84, "stapler": 85, "starfish": 86,
            "stegosaurus": 87, "strawberry": 88, "stop_sign": 89, "sunflower": 90, "tick": 91,
            "trilobite": 92, "umbrella": 93, "watch": 94, "water_lilly": 95, "wheelchair": 96, "wild_cat": 97,
            "windsor_chair": 98, "wrench": 99, "yin_yang": 100
        }

    def get_file_list(self):
        return glob(os.path.join(self.data_path, '*/*.npy'))

    def get_item(self, idx):
        file_idx, start_idx, end_idx = self.video_chunks_ids[idx]
        target_name = (self.file_list[file_idx].split('/')[-2])
        target = self.classes[target_name]
        data = duts.read_npy_chunk(self.file_list[file_idx], start_idx, end_idx - start_idx).astype(
            np.float32)  # x, y, p, t

        return data, target