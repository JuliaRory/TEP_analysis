from h5py import File

def get_data_ui_epochs(filename):
    # files from D:\temp\ICA\Cleaned_epochs
    with File(filename, "r") as h5f:
        epochs = h5f["cleanedResult"]["epochs_clean"][:]  # (n_channels, n_epochs, n_samples)
        tvec = h5f["cleanedResult"]["tvec"][:].T

    if filename.find("session_35") != -1 or filename.find("session_34") != -1:
        tvec = tvec[ 500:]
        epochs = epochs[:, :, 500:]
    return epochs, tvec


def decode_matlab_string_payload(ds):
    import numpy as np
    arr = np.asarray(ds[()]).ravel().astype(np.uint64)

    ndims = int(arr[1])
    shape = tuple(int(x) for x in arr[2:2 + ndims])
    n_items = int(np.prod(shape))

    lengths_start = 2 + ndims
    lengths = [int(x) for x in arr[lengths_start:lengths_start + n_items]]
    packed = arr[lengths_start + n_items:]

    codes = []
    for value in packed:
        codes.extend(
            np.frombuffer(int(value).to_bytes(8, "little"), dtype="<u2").tolist()
        )

    strings = []
    pos = 0
    for length in lengths:
        strings.append("".join(chr(c) for c in codes[pos:pos + length]))
        pos += length

    return np.array(strings, dtype=object).reshape(shape, order="F")


def read_matlab_string(f, dataset_path):
    from numpy import asarray
    desc = asarray(f[dataset_path][()]).ravel()
    mcos_index = int(desc[4]) + 1
    ref = f["#subsystem#/MCOS"][()].ravel()[mcos_index]
    return decode_matlab_string_payload(f[ref])