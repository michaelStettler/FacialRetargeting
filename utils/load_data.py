from c3d.c3dReader3 import C3DReader
from utils.compute_delta import compute_delta
import os, sys
import numpy as np
import logging


def load_c3d_file(file, template_labels=None, get_labels=False, verbose=False):
    """
    Load c3d file of interest and transform it into a list and labels

    The function can take in argument a template labels list that is used to sort the positions accordingly

    :param file: .c3d file of interest
    :param get_labels:
    :param verbose: print option
    :return: numpy array of all 3D position (x, y, z) for each frames and markers (n_frames, n_markers, 3)
    """

    c3d_object = C3DReader(file)
    labels = c3d_object.getparameterDict()['POINT']['LABELS']

    # sorted index to match the sorted template array since c3d file does not ensure the same sequence of markers
    if template_labels is not None:
        # build the matching indexes
        sorted_index = np.zeros(len(template_labels))

        for l, label in enumerate(labels):
            has_been_sorted = False
            for s, sorted_label in enumerate(template_labels):
                if label == sorted_label:
                    sorted_index[s] = l
                    has_been_sorted = True

            if has_been_sorted == False:
                logging.warning("[LOAD C3D] The label {} has not been found! you will likely have dimensionality error".format(label))
        sorted_index = sorted_index.astype(int)

    sequence = []
    for f, frame in enumerate(c3d_object.iterFrame()):
        positions = []
        for m, marker in enumerate(frame):
            positions.append([marker.x, marker.y, marker.z])

        sequence.append(positions)

    if len(np.shape(sequence)) > 1:
        n_markers = np.shape(sequence)[1]
        if template_labels is not None:
            # sort array to match the template
            sequence = np.array(sequence)
            sequence = sequence[:, sorted_index[:n_markers], :]

            labels = np.array(labels)[sorted_index]

    if verbose:
        print("Finished loading:", np.shape(sequence), "data points from", file)
        print("Array is:", sys.getsizeof(sequence) / 1024 / 1024, "MB")

    if get_labels:
        return np.array(sequence), np.array(labels)
    else:
        return np.array(sequence)


def load_training_seq(path, seq_name, num_markers, template_labels=None, get_labels=False):

    if get_labels:
        data, labels = load_c3d_file(os.path.join(path, seq_name), template_labels, get_labels=get_labels)
    else:
        data = load_c3d_file(os.path.join(path, seq_name), template_labels, get_labels=get_labels)

    if len(np.shape(data)) >= 3:
        if np.shape(data)[1] == num_markers:
            if get_labels:
                return data, labels
            else:
                return data
        else:
            logging.warning("A marker is missing {}, the file {} is discarded".format(np.shape(data), seq_name))
    else:
        logging.warning("The file {} has wrong size {}. The file is discarded".format(seq_name, np.shape(data)))

    if get_labels:
        return None, None
    else:
        return None


def load_training_frames(path, num_markers=45, template_labels=None, max_num_seq=None, down_sample_factor=None, get_labels=False):
    # get all sequences
    sequences_list = []
    for file in os.listdir(path):
        if file.endswith(".c3d"):
            sequences_list.append(file)

    print("[loading] found", len(sequences_list), "sequence(s)")

    # get all training frames
    print("[loading] loading training data...")
    if max_num_seq is not None:
        sequences_list = sequences_list[:max_num_seq]
        logging.warning("Using max_num_seq parameter! {}".format(max_num_seq))

    training_seq = []
    for seq_name in sequences_list:
        print("[loading] sequence name:", seq_name)
        if get_labels:
            seq, labels = load_training_seq(path, seq_name, num_markers,
                                            template_labels=template_labels,
                                            get_labels=get_labels)
        else:
            seq = load_training_seq(path, seq_name, num_markers,
                                    template_labels=template_labels,
                                    get_labels=get_labels)

        if seq is not None:
            if down_sample_factor is not None:
                sample = np.arange(0, len(seq), down_sample_factor)
                seq = seq[sample]
            training_seq.append(seq)
    print("[loading] Retaining", len(training_seq), "sequence(s)")
    print("length first sequence", np.shape(training_seq[0]))
    print()

    # concatenate all the sequence into one dimensional frame length
    af = np.concatenate(training_seq, axis=0)
    print("[loading] Finished loading sequences, found a total of", np.shape(af)[0], "training frames")

    if get_labels:
        return af, labels
    else:
        return af


if __name__ == '__main__':
    """
    test loading a c3d file
    
    run: python -m utils.load_data
    """
    template_labels = ['LeftBrow1', 'LeftBrow2', 'LeftBrow3', 'LeftBrow4', 'RightBrow1', 'RightBrow2', 'RightBrow3',
                     'RightBrow4', 'Nose1', 'Nose2', 'Nose3', 'Nose4', 'Nose5', 'Nose6', 'Nose7', 'Nose8',
                     'UpperMouth1', 'UpperMouth2', 'UpperMouth3', 'UpperMouth4', 'UpperMouth5', 'LowerMouth1',
                     'LowerMouth2', 'LowerMouth3', 'LowerMouth4', 'LeftOrbi1', 'LeftOrbi2', 'RightOrbi1', 'RightOrbi2',
                     'LeftCheek1', 'LeftCheek2', 'LeftCheek3', 'RightCheek1', 'RightCheek2', 'RightCheek3',
                     'LeftJaw1', 'LeftJaw2', 'RightJaw1', 'RightJaw2', 'LeftEye1', 'RightEye1', 'Head1', 'Head2',
                     'Head3', 'Head4']

    path = r"D:\MoCap_Data\David\NewSession_labeled"
    file = 'am_Trail27.c3d'
    data, labels = load_c3d_file(os.path.join(path, file),
                                 template_labels=template_labels,
                                 get_labels=True,
                                 verbose=True)
    print("labels", len(labels))
    print(labels)
    print(data[0])
    print()

    file = 'AngerTrail05.c3d'
    data, labels = load_c3d_file(os.path.join(path, file),
                                 template_labels=template_labels,
                                 get_labels=True,
                                 verbose=True)
    print("labels", len(labels))
    print(labels)
    print(data[0])
    print()
