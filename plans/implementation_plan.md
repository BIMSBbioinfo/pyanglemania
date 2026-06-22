# anglemania in python and based on GPU

# Motivation:
The anglemania R package which we developed is working nicely but we were thinking of ways to speed it up and decided to re-write it in python and integrate it into the scanpy and more recently the rapids-singlecell environment/architecture. For this purpose, the R package shouldn't be completely recreated but rather the main idea/main points should be transferred into the new python package and integrated into the scanpy/rapids-singlecell structure.

# main differences we want in python compared to the current R package
We want to recreate the idea of anglemania but in the python architecture and with GPU support.
Currently, we create correlation matrices per batch and build this z score matrix and store these matrices on disk. Only in the end we compute the mean and SD or signal-to-noise ratio. We now want to just compute the mean and the SD on the fly, and not store the individual matrices in the meanwhile.

# task
Please rewrite the R package into python keeping the idea of anglemania but make it fit into the the scanpy/rapids-singlecell archtitecture, keeping in mind the Motivation and the differences we want to implement.