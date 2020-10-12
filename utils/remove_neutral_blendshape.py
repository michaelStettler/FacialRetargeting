def remove_neutral_blendshape(mesh_list, neutral_pose_name):

    ref_index = None
    bs_index = []
    cleaned_mesh_list = []
    for m, mesh in enumerate(mesh_list):
        if mesh == neutral_pose_name:
            ref_index = m
        else:
            bs_index.append(m)
            cleaned_mesh_list.append(mesh)
    if ref_index is None:
        raise ValueError("No Neutral blendshape pose found!")

    return cleaned_mesh_list, bs_index, ref_index