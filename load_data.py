from c3d.c3dReader3 import C3DReader
from compute_delta import compute_delta
import os, sys
import numpy as np


def load_c3d_file(file, get_labels=False, verbose=False):
    """
    Load c3d file of interest and transform it into a list and return labels

    :param file: .c3d file of interest
    :param get_labels:
    :param verbose: print option
    :return: numpy array of all 3D position (x, y, z) for each frames and markers (n_frames, n_markers, 3)
    """

    c3d_object = C3DReader(file)
    if get_labels:
        labels = c3d_object.getparameterDict()['POINT']['LABELS']

    sequence = []
    for f, frame in enumerate(c3d_object.iterFrame()):
        markers = []
        for m, marker in enumerate(frame):
            markers.append([marker.x, marker.y, marker.z])

        sequence.append(markers)

    if verbose:
        print("Finished loading:", np.shape(sequence), "data points from", file)
        print("Array is:", sys.getsizeof(sequence) / 1024 / 1024, "MB")

    if get_labels:
        return np.array(sequence), np.array(labels)
    else:
        return np.array(sequence)


def load_training_seq(path, seq_name, num_markers):
    data = load_c3d_file(os.path.join(path, seq_name))
    if len(np.shape(data)) >= 3:
        if np.shape(data)[1] == num_markers:
            return data
        else:
            print("Warning! A marker is missing", np.shape(data), ", the file", seq_name, "is discarded")
    else:
        print("Warning! The file", seq_name, "has wrong size", np.shape(data), ". The file is discarded")

    return None


def load_training_frames(path, num_markers=45, max_num_seq=None):
    # get all sequences
    sequences_list = []
    for file in os.listdir(path):
        if file.endswith(".c3d"):
            sequences_list.append(file)

    print("found", len(sequences_list), "sequence")

    # get all training frames
    print("loading training data...")
    training_seq = []
    if max_num_seq is None:
        for seq_name in sequences_list:
            seq = load_training_seq(path, seq_name, num_markers)

            if seq is not None:
                training_seq.append(seq)
    else:
        print("Warning!, Using max_num_seq parameter!", max_num_seq)
        for seq_name in sequences_list[:max_num_seq]:
            seq = load_training_seq(path, seq_name, num_markers)

            if seq is not None:
                training_seq.append(seq)
    print("Retaining", len(training_seq), "sequence(s)")

    # get total length
    tot_frames = 0
    for s, seq in enumerate(training_seq):
        tot_frames += np.shape(training_seq[s])[0]

    # declare af and detla_af
    af = np.zeros((tot_frames, np.shape(training_seq[0])[1], np.shape(training_seq[0])[2]))
    delta_af = np.zeros((tot_frames - len(training_seq), np.shape(training_seq[0])[1], np.shape(training_seq[0])[2]))  # remove k=0

    # merge all seq into one tensor and compute deltas
    iterator = 0
    delta_iterator = 0
    for s, seq in enumerate(training_seq):
        shape_seq = np.shape(seq)
        # compute deltas
        deltas = compute_delta(seq, seq[0])

        # merge data
        af[iterator:(iterator+shape_seq[0])] = seq
        delta_af[delta_iterator:(delta_iterator+shape_seq[0]-1)] = deltas[1:]  # remove delta0

        # iterate num frames
        iterator += shape_seq[0]
        delta_iterator += shape_seq[0] - 1  # remove k=0

    print("Finished loading sequences, found a total of", tot_frames, "training frames")
    return af, delta_af


if __name__ == '__main__':
    """
    test loading a c3d file
    
    run: python -m load_data
    """

    path = r"D:\MoCap_Data\David\NewSession_labeled"
    file = 'AngerTrail05.c3d'
    data, labels = load_c3d_file(os.path.join(path, file), get_labels=True, verbose=True)

    print()
    print("labels", len(labels))
    print(labels)
