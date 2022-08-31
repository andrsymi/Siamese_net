from util_feature_extractor import extract_features, construct_filters_iir
from scipy.io import wavfile
import numpy as np
import pandas as pd


def main(filename, detect_params, fft_length):
    sr, wave = wavfile.read(filename)
    if np.max(wave) > 10:
        wave = wave / 32768.0

    filt_params = construct_filters_iir(sr)

    csv_filename = filename.split('.wav')[0]
    time_stamps = pd.read_csv("{}.csv".format(csv_filename), header=None)
    sample_stamps = (time_stamps * sr).values[:, 0].astype(int)

    result = []
    for i in range(len(sample_stamps) - 1):
        # extract features
        features = extract_features(wave[sample_stamps[i]:sample_stamps[i + 1]], detect_params, filt_params, fft_length)
        # add class label in first column
        class_data = np.concatenate((np.zeros((features.shape[0], 1)) + i, features), axis=1)

        if i == 0:
            # create dummy empty array with same column size
            data = np.empty((0, class_data.shape[1]))

        # append
        data = np.concatenate((data, class_data), axis=0)

    # columns = ["label"]
    # for i in range(1, data.shape[1]):
    #     columns.append()

    col_names = ["f{}".format(i+1) for i in range(data.shape[1] - 1)]
    col_names.insert(0, "label")

    df = pd.DataFrame(data, columns=col_names)
    df.to_csv("../data.csv", index=False)


if __name__ == "__main__":
    file = "merged.wav"
    length_fft = 1024

    detect_parameters = {'frame_size': 64,
                         'power_thresh': 0.003,  # changed to refer to RMS
                         'dBrat': 3,
                         'look_back_frames': 25,
                         'look_ahead_frames': 1
                         }

    main(file, detect_parameters, length_fft)
