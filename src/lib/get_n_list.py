import os

import numpy as np


def load_off(off_file_name):
    file_handle = open(off_file_name)
    # n_points = int(file_handle.readlines(6)[1].split(' ')[0])
    # all_strings = ''.join(list(islice(file_handle, n_points)))

    file_list = file_handle.readlines()
    n_points = int(file_list[1].split(" ")[0])
    all_strings = "".join(file_list[2 : 2 + n_points])
    array_ = np.fromstring(all_strings, dtype=np.float32, sep="\n")

    all_strings = "".join(file_list[2 + n_points :])
    array_int = np.fromstring(all_strings, dtype=np.int32, sep="\n")

    array_ = array_.reshape((-1, 3))

    return array_, array_int.reshape((-1, 4))[:, 1::]


def get_n_list(mesh_path):
    if not os.path.isdir(mesh_path):
        return [load_off(mesh_path)[0].shape[0]]
    name_list = os.listdir(mesh_path)
    # # for ObjectNet
    # name_list = [n for n in name_list if len(n) == 6]
    name_list = ["%02d.off" % (i + 1) for i in range(len(name_list))]
    out = []
    for n in name_list:
        out.append(load_off(os.path.join(mesh_path, n))[0].shape[0])
    return out

def compute_max_n(config):
    # number of Gaussian features on cuboid
    n_list_set = []
    for class_ in config.dataset.classes:
        mesh_path = f"{config.dataset.paths.root}/{config.dataset.paths.mesh}/{class_}"
        n_list = get_n_list(mesh_path)
        n_list_set.append(n_list[0])
    max_n = max(n_list_set)
    print("Computed max_n:", max_n)
    print("Computed n_list_len:", n_list_set)
    print("Computed sum of n_list_len:", np.sum(n_list_set))
    return max_n, n_list_set

if __name__ == "__main__":
    print(get_n_list("../PASCAL3D/CAD_d4/car/"))
