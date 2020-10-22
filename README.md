<h1> Facial Retargeting with Automatic Range of Motion Alignment </h1>
This repository reproduce the paper "Facial Retargeting with Automatic Range of Motion Alignment" (2017) done by Roger Blanco, Eduard Zell, J.P Lewis, Junyong Noh and Mario Botsch. 

<h2> Introduction </h2>
This repository supposes that you already have a character mesh with some blendshapes and a set of Vicon motion capture sequences (in.c3d file). 

To run the complete facial retargeting, you will have to proceed with 8 different steps that will be carried out in your Maya scene and in python. The steps are kept independent to allow faster try and test results without having to re-run all the previous steps. A configuration file is there to help you getting trough all the steps. Therefore, the first thing to do is to set up this config file (config/<your_new_config>.json) with all your different path, names, parameters etc.
You will then need to modify the script to refer to your configuration file as to load your set of parameters. I chose to have this config file to allow to tweak some hyper-parameters and try which one gives the best results. Indeed, while retargeting my sequences, I figured that some expressions where looking better when changing a bit the initial parameters. But this could be part of the extremely low amount of markers used in our setup (41 facial landmarks and 4 references). 

Other than that, the scripts will take care of finding all the blendshapes of your scene, build the nodes, triangulate your mesh and normalize your sequences with a reference frame set in your configuration file.  

<h2> Procedure </h2>
To apply facial retargeting from scratch, all you need to do is to follow the scripts numbers from 01 to 08. Following a exhaustive list of what the scripts are doing.
As I was working on windows, please note that the script "05_build_LdeltaV.py" has been run in python 2.7 as it uses the pymesh library which is not available for Windows. 

<h3> Pre-processing steps (in maya) </h3>
These following steps have to be carried out in Maya console by copy pasting the scripts within it. The scripts are found under the "maya_script" repository.

- 1: Triangulate all your meshes if it's not already done. Simply run the script: 01_triangulate_mesh.
- 2: Attach all the blendshape, this will create the blendshape node where the weights are going to be aplpied to. I would recommend to delete your node if it exists and apply the provided script "02_create_blendshapes_nodes" which will save a mesh_name_list that will be used to ensure the correct indexation of your blendshapes and the final facial retargeting weights.
- 3: Save the vertices' positions that semantically represent your set of markers between your character (avatar) and your actor. To do so, run the script "03_extract_blendshape_pos_and_obj", which will save the positions of all the sparse markers for each blendshape as well as saving the blendshapes into an .obj file which we be use to compute the Laplacian over the full meshes. 

<h3> Facial retargeting (in python) </h3>
These following steps can be carried out directly within python by running: python <script_name>.py

- 4: Save your reference frame positions by running "04_save_neutral_pose.py". This will allow the next script to use this positions as a reference for all the delta. 
- 5: (Python 2.7 and uses pymesh library!) Compute the Laplacian of all your blendshapes. This step can be carried it only once to save computation time. (05_build_LdeltaV.py).
- 6: Compute your set of "personal actor blendshapes". This script implement the section 4 of the paper. 
- 7: Retarget your sequences by computing the blendshape weights. This script implement the section 5 of the paper. 

<h3> Facial retargeting (in Maya) </h3>

- 8: Finally, run the script "08_apply_weights.py" in your maya console to apply the weights on your scene.

<h3> Final Note </h3>
I would like to thanks Eduard Zell for his very valuable help throughout this implementation. I have learned a lot to implement this paper and I have been pleased by the results.
