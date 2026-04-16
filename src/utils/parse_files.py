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