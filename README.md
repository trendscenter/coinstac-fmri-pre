All the raw fMRI data will only be stored on the user's local machine and never leave their machine. Pre-processing will happen on the local machine and outputs are stored on the local machine. 

FMRI pre-processing involves taking subjects in their native space and running pre-processing pipeline on those images. Steps include:
Number of volumes to ignore from beginning (default=0)
Distortion correction if set in the options
Realignment
Slicetiming
Normalizing
Smoothing (10,10,10 kernel is used by default which can be changed). 

It uses spm12 toolbox(MAtlab) for pre-processing. They are normalized to MNI space and uses SPM12’s T1 template from the SPM12 toolbox. Framewise displacement (motion qc) is calculated as a motion quality control metric if users want to check the pre-processing outputs and threshold their data in some way.
