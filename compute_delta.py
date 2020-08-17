def compute_delta(data, ref):
    deltas = []
    for d in data:
        deltas.append(d - ref)

    return deltas