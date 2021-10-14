#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This layer runs the pre-processing fmri pipeline in SPM12 based on the inputs
from interface adapter layer. This layer uses entities layer to modify
nodes of the pipeline as needed.
"""
import contextlib,traceback,sys


@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr
    e.g.:
    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    """

    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        # print(dest_file)
        # print()
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()


import sys, os, glob, shutil, math, base64, warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
import ujson as json
from nipype import config

config.set('execution', 'remove_unnecessary_outputs', 'false')

# Load bids layout interface for parsing bids data to extract T1w scans,subject names etc.
from bids import BIDSLayout

import nibabel as nib
import nipype.pipeline.engine as pe
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import fmri_entities_layer


#Stop printing nipype.workflow info to stdout
from nipype import logging
logging.getLogger('nipype.workflow').setLevel('CRITICAL')

def setup_pipeline(data='', write_dir='', data_type=None, **template_dict):
    """setup the pre-processing pipeline on fMRI scans
        Args:
            data (array) : Input data
            write_dir (string): Directory to write outputs
            data_type (string): BIDS, niftis, dicoms
            template_dict ( dictionary) : Dictionary that stores all the paths, \
                                          file names, software locations
        Returns:
            computation_output (json): {"output": {
                                                  "success": {
                                                    "type": "boolean"
                                                  },
                                                   "message": {
                                                    "type": "string",
                                                  },
                                                   "download_outputs": {
                                                    "type": "string",
                                                  },
                                                   "display": {
                                                    "type": "string",
                                                  }
                                                  }
                                        }
        Comments:
            After setting up the pipeline here , the pipeline is run with run_pipeline function
        """
    try:

        # Create pipeline nodes from fmri_entities_layer.py and pass them to run_pipeline function

        [realign, slicetiming, datasink, fmri_preprocess] = create_pipeline_nodes(
            **template_dict)
        if data_type == 'bids':
            # Runs the pipeline on each subject serially
            layout = BIDSLayout(data)
            smri_data = layout.get(
                datatype='func', extensions='.nii.gz')
            return run_pipeline(
                write_dir,
                smri_data,
                realign,
                slicetiming,
                datasink,
                fmri_preprocess,
                data_type='bids',
                **template_dict)
        elif data_type == 'nifti':
            # Runs the pipeline on each nifti file serially
            smri_data = data
            return run_pipeline(
                write_dir,
                smri_data,
                realign,
                slicetiming,
                datasink,
                fmri_preprocess,
                data_type='nifti',
                **template_dict)
        elif data_type == 'dicoms':
            # Runs the pipeline on each nifti file serially
            smri_data = data
            return run_pipeline(
                write_dir,
                smri_data,
                realign,
                slicetiming,
                datasink,
                fmri_preprocess,
                data_type='dicoms',
                **template_dict)

    except Exception as e:
        sys.stdout.write(
            json.dumps({
                "output": {
                    "message": str(e)
                },
                "cache": {},
                "success": True
            }))


def remove_tmp_files(fmri_out):
    """this function removes any tmp files in the docker"""

    for a in glob.glob('/var/tmp/*'):
        os.remove(a)

    #    for b in glob.glob(os.getcwd() + '/crash*'):
    for b in glob.glob('/crash*'):
        os.remove(b)

    #    for c in glob.glob(os.getcwd() + '/tmp*'):

    # Copy the /tmp files to results directory to retain nipype logs

    logdir = fmri_out + '/nipype_logs'
    c = glob.glob('/tmp/tmp*/')

    for i in c:
        #  try:
        shutil.copytree(i, logdir + i)
        #  except OSError as e:
        #    if e.errno == 17:
        os.chmod(logdir + i, 0o755)
    for (dirname, dirs, files) in os.walk(logdir):
        for file in files:
            if file.endswith('.nii'):
                source_file = os.path.join(dirname, file)
                os.remove(source_file)

    # Now remove /tmp/tmp* directories

    #    for c in glob.glob(os.getcwd() + '/tmp*'):
    for c in glob.glob('/tmp/tmp*'):
        shutil.rmtree(c, ignore_errors=True)

    for d in glob.glob(os.getcwd() + '/__pycache__'):
        shutil.rmtree(d, ignore_errors=True)

    shutil.rmtree(os.getcwd() + '/fmri_preprocess', ignore_errors=True)

    if os.path.exists(os.getcwd() + '/pyscript.m'):
        os.remove(os.getcwd() + '/pyscript.m')

    # And as long as we're here, let's move over the graph visualization files
    gr = glob.glob('/tmp/graph*png')
    # gt = glob.glob('/computation/graph*dot')
    for i in gr:
        shutil.move(i, fmri_out)
    # for i in gt:
    #  os.remove(i)


def write_readme_files(write_dir='', data_type=None, **template_dict):
    """This function writes readme files"""

    # Write a text file with info on each of the output nifti files
    if data_type == 'bids':
        with open(
                os.path.join(write_dir, template_dict['outputs_manual_name']),
                'w') as fp:
            fp.write(template_dict['bids_outputs_manual_content'])
            fp.close()
    elif data_type == 'nifti':
        with open(
                os.path.join(write_dir, template_dict['outputs_manual_name']),
                'w') as fp:
            fp.write(template_dict['nifti_outputs_manual_content'])
            fp.close()
    elif data_type == 'dicoms':
        with open(
                os.path.join(write_dir, template_dict['outputs_manual_name']),
                'w') as fp:
            fp.write(template_dict['dicoms_outputs_manual_content'])
            fp.close()

    # Write a text file with info on quality control correlation coefficent
    with open(os.path.join(write_dir, template_dict['qc_readme_name']),
              'w') as fp:
        fp.write(template_dict['qc_readme_content'])
        fp.close()


def calculate_FD(rp_text_file, **template_dict):
    """Calculates framewise displacement from realignment parameters.
       Realignment (movement) parameters are calculated from realignment of nifti.
           Args:
               realignment parameters.txt file
           Returns:
               Mean of RMS of Framewise displacement
           Comments:
               Framewise Displacement of a time series is defined as the sum of the \
               absolute values of the derivatives of the six realignment parameters.
               Realignment displacements are converted from degrees to millimeters by \
               calculating displacement on the surface of a sphere of radius 50 mm.
               (Power et al, 2012)
            """
    realignment_parameters = np.loadtxt(rp_text_file)

    ### plot translational motion parameters
    fig = plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(realignment_parameters[:, 0], label='x')
    plt.plot(realignment_parameters[:, 1], label='y')
    plt.plot(realignment_parameters[:, 2], label='z')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    plt.ylabel('Translational Movement (mm)')
    plt.legend()
    write_path = os.path.dirname(rp_text_file)
    plt.savefig(os.path.join(write_path, template_dict['fmri_trans_move_filename']), bbox_inches='tight')
    plt.close()

    ### plot rotational motion parameters
    fig = plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(realignment_parameters[:, 3], label='pitch')
    plt.plot(realignment_parameters[:, 4], label='roll')
    plt.plot(realignment_parameters[:, 5], label='yaw')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    plt.ylabel('Rotational Movement (deg)')
    plt.legend()
    write_path = os.path.dirname(rp_text_file)
    plt.savefig(os.path.join(write_path, template_dict['fmri_rot_move_filename']), bbox_inches='tight')
    plt.close()

    ### estimate fd
    rot_indices = range(3, 6)
    rad = 50
    # assume head radius of 50mm
    rot = realignment_parameters[:, rot_indices]
    rdist = rad * np.tan(rot)
    realignment_parameters[:, rot_indices] = rdist
    diff = np.diff(realignment_parameters, axis=0)
    FD_rms = np.sqrt(np.sum(diff ** 2, axis=1))
    FD_rms_mean = np.mean(FD_rms)
    write_path = os.path.dirname(rp_text_file)

    with open(
            os.path.join(write_path, template_dict['fmri_qc_filename']),
            'w') as fp:
        fp.write("%3.2f\n" % (FD_rms_mean))
        fp.close()


def nii_to_image_converter(write_dir, label, **template_dict):
    """This function converts nifti to base64 string"""
    import nibabel as nib
    from nilearn import plotting, image
    import os, base64

    file = glob.glob(os.path.join(write_dir, template_dict['display_nifti']))
    mask = image.index_img(file[0], int(
        (image.load_img(file[0]).shape[3]) / 2))
    new_data = mask.get_data()

    clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)

    plotting.plot_anat(
        clipped_img,
        cut_coords=(0, 0, 0),
        annotate=False,
        draw_cross=False,
        output_file=os.path.join(write_dir,
                                 template_dict['display_image_name']),
        display_mode='ortho',
        title=label + ' ' + template_dict['display_pngimage_name'],
        colorbar=False)

    #print()


#### Remove dummy scans, if number of scans to be removed is specified in input.json

def discard_dummy_scans(fmri_out, nifti_file, nt, **template_dict):
    import nipype.interfaces.fsl as fsl

    # print("Discarding dummy scans")
    #
    # print()
    # print("Input file is", nifti_file)
    # print()

    fslroi_rmDumScns = pe.Node(interface=fsl.ExtractROI(), name="fslroi_rmDumScns")

    fslroi_rmDumScns.inputs.in_file = nifti_file
    fslroi_rmDumScns.inputs.z_min = template_dict['num_vols_to_remove']
    fslroi_rmDumScns.inputs.z_size = nt - template_dict['num_vols_to_remove']

    with stdchannel_redirected(sys.stderr, os.devnull):
        fslroi_rmDumScns.run()

    croi = glob.glob('/tmp/tmp*/fslroi_rmDumScns/*roi.nii')
    if len(croi) == 1:
        write_path = os.path.join(fmri_out, template_dict['fmri_output_dirname'])
        shutil.copy(croi[0], write_path)
        nifti_file = croi[0]
    #     print("Input file is", nifti_file)
    else:
        sys.exit()
    #     print("No fslroi file found")
    #     print()

    return nifti_file


#### Attach SBRef, if specified in json file.

def attachSBRef(fmri_out, smri_data, nifti_file, **template_dict):
    import nipype.interfaces.fsl as fsl

    # print()
    # print("Attaching SBRef")
    # print()
    # print("Input file is", nifti_file)
    # print()

    if smri_data.endswith(".nii.gz"):
        filebase = os.path.basename(smri_data)
        sbref_dir = glob.glob(os.path.join(template_dict['input_dir'], filebase.split(".")[0] + '_[sS][bB][rR]*'))[0]
        sbref = glob.glob(os.path.join(sbref_dir, '*.nii*'))[0]
        # print("SBRef file is", sbref)
        # print()

    merge = pe.Node(interface=fsl.Merge(), name="merge")

    merge.inputs.in_files = [sbref, nifti_file]
    merge.inputs.dimension = 't'

    with stdchannel_redirected(sys.stderr, os.devnull):
        merge.run()

    cmerge = glob.glob('/tmp/tmp*/merge/*merged.nii*')
    if len(cmerge) == 1:
        write_path = os.path.join(fmri_out, template_dict['fmri_output_dirname'])
        shutil.copy(cmerge[0], write_path)
        nifti_file = cmerge[0]
        # print("New input file is", nifti_file)
    else:
        sys.exit()
        # print("No SBRef merged file found")
        # print()

    return nifti_file


#### Remove SBRef, if specified in json file.

def rm_SBRef(fmri_out, nifti_file, **template_dict):
    import nipype.interfaces.fsl as fsl
    import nibabel as nib

    # print("Removing SBRef")

    fslroi_rmFirstVol = pe.Node(interface=fsl.ExtractROI(), name="fslroi_rmFirstVol")

    smfmri_file = glob.glob(os.path.join(fmri_out, template_dict['fmri_output_dirname'],'sw*merged*.nii*'))[0]

    smfmri = nib.load(smfmri_file)
    fmri_data = smfmri.get_fdata()
    nt = fmri_data.shape[3]

    fslroi_rmFirstVol.inputs.in_file = smfmri_file
    fslroi_rmFirstVol.inputs.z_min = 1
    fslroi_rmFirstVol.inputs.z_size = nt - 1

    with stdchannel_redirected(sys.stderr, os.devnull):
        fslroi_rmFirstVol.run()

    croi = glob.glob('/tmp/tmp*/fslroi_rmFirstVol/*roi.nii')
    if len(croi) == 1:
        write_path = os.path.join(fmri_out, template_dict['fmri_output_dirname'])
        shutil.copy(croi[0], write_path)
        smoothed = os.path.basename(glob.glob(os.path.join(write_path, 's*_roi.nii'))[0])
        # print(smoothed)
        smoothed_re = smoothed.replace("roi", "noSBRef")
        os.rename(os.path.join(write_path, smoothed), os.path.join(write_path, smoothed_re))

    return


#### Run distortion correction, if specified by path to fieldmaps.

def distortionCorrection(fmri_out, nifti_file, **template_dict):
    # if template_dict['dist_corr'] is not None:

    def run_topup(write_path):

        from nipype.interfaces.fsl import TOPUP
        topup_fmri = pe.Node(interface=TOPUP(), name="topup_fmri")

        topup_fmri.inputs.in_file = os.path.join(write_path, 'distortion_corr.nii')
        topup_fmri.inputs.encoding_file = os.path.join(write_path, 'acq_params_fmri_allN.txt')
        topup_fmri.inputs.config = '/opt/fsl-6.0.3/etc/flirtsch/b02b0.cnf'

        with stdchannel_redirected(sys.stderr, os.devnull):
            topup_fmri.run()

        ctp = glob.glob('/tmp/tmp*/topup_fmri/distortion_corr*')
        for file in ctp:
            shutil.copy(file, write_path)

        return ('Success')

    def run_applyTopup(write_path, nifti_file):

        from nipype.interfaces.fsl import ApplyTOPUP
        applyTopup_fmri = pe.Node(interface=ApplyTOPUP(), name="applyTopup_fmri")

        applyTopup_fmri.inputs.in_files = nifti_file
        applyTopup_fmri.inputs.encoding_file = os.path.join(write_path, 'acq_params_fmri_allN.txt')
        applyTopup_fmri.inputs.in_index = [1]
        applyTopup_fmri.inputs.in_topup_fieldcoef = os.path.join(write_path, 'distortion_corr_base_fieldcoef.nii')
        applyTopup_fmri.inputs.in_topup_movpar = os.path.join(write_path, 'distortion_corr_base_movpar.txt')
        applyTopup_fmri.inputs.interp = 'spline'
        applyTopup_fmri.inputs.method = 'jac'

        with stdchannel_redirected(sys.stderr, os.devnull):
            applyTopup_fmri.run()

        catp = glob.glob('/tmp/tmp*/applyTopup_fmri/*_corrected.nii')
        if len(catp) == 1:
            shutil.copy(catp[0], write_path)
            nifti_file = catp[0]
            # print("New input file is", nifti_file)

        return (nifti_file)

    # print("Running distortion correction")
    #
    # print("Input file to distortion correction is ", nifti_file)
    # print()

    ### Create concatenated fieldmap file from individual series runs       #

    fileList = []
    for file in glob.glob(os.path.join(template_dict['input_dir'], '*dist*[aA][pP]*/*[aA][pP]*.nii*')):
        fileList.append(file)

    for file in glob.glob(os.path.join(template_dict['input_dir'], '*dist*[pP][aA]*/*[pP][aA]*.nii*')):
        fileList.append(file)

    ### Create acquisition parameters file from json files           #

    jfileList = []

    for file in glob.iglob(os.path.join(template_dict['input_dir'], '*dist*[aA][pP]*/*[aA][pP]*.json')):
        jfileList.append(file)

    for file in glob.iglob(os.path.join(template_dict['input_dir'], '*dist*[pP][aA]*/*[pP][aA]*.json')):
        jfileList.append(file)

    datajson = glob.glob(os.path.join(template_dict['data_dir'], '*.json'))[0]
    with open(datajson) as json_file:
        dj = json.load(json_file)
        if dj['PhaseEncodingDirection'] == 'j':
            fileList.sort(reverse=True)
            jfileList.sort(reverse=True)
            # print("Phase encode sequence is PAAP")
        else:
            sys.exit()
    #         print("Phase encode sequance is APPA")
    #
    # print(fileList)
    # print()
    # print(jfileList)
    # print()

    #### Create concatenated distortion image
    images = []
    for i in range(len(fileList)):
        fname = 'images' + str(i)
        images.append(fname)

    for i in range(len(fileList)):
        with stdchannel_redirected(sys.stderr, os.devnull):
            images[i] = nib.load(fileList[i])


    image_all = nib.concat_images(images, check_affines=False, axis=3)
    write_path = os.path.join(fmri_out, template_dict['fmri_output_dirname'])
    nib.save(image_all, os.path.join(write_path, 'distortion_corr.nii'))

    #### Create acquisition parameters file from json files
    write_path = os.path.join(fmri_out, template_dict['fmri_output_dirname'])
    with open(os.path.join(write_path, 'acq_params_fmri_allN.txt'), 'w') as outfile:
        for i in range(len(jfileList)):
            with open(jfileList[i]) as json_file:
                jfl = json.load(json_file)
                if jfl['PhaseEncodingDirection'] == 'j-':
                    fline = '0 -1 0 ' + str(jfl['TotalReadoutTime'])
                elif jfl['PhaseEncodingDirection'] == 'j':
                    fline = '0 1 0 ' + str(jfl['TotalReadoutTime'])
                elif jfl['PhaseEncodingDirection'] == 'i-':
                    fline = '1 0 0 ' + str(jfl['TotalReadoutTime'])
                else:
                    fline = '-1 0 0 ' + str(jfl['TotalReadoutTime'])
                for j in range(images[i].shape[3]):
                    outfile.write(fline)
                    outfile.write("\n")

    #### Run topup and applytopup

    status = run_topup(write_path)
    if status == 'Success':
        with stdchannel_redirected(sys.stderr, os.devnull):
            dcnifti_file = run_applyTopup(write_path, nifti_file)

    nifti_file = dcnifti_file
    #print("NEW NIFTI FILE", nifti_file)
    return (nifti_file)


def create_pipeline_nodes(**template_dict):
    """This function creates and modifies nodes of the pipeline from entities layer with nipype
           smooth.node.inputs.fwhm: (a list of from 3 to 3 items which are a float or a float)
               3-list of fwhm for each dimension.
           This is the size of the Gaussian (in mm) for smoothing the preprocessed data by. This
               is typically between 4mm and 12mm.
       """

    # SPM Pipeline

    # 1 Slicetiming Node and settings #
    slicetiming = fmri_entities_layer.Slicetiming(**template_dict)

    # 2 Realign node and settings #
    realign = fmri_entities_layer.Realign(**template_dict)

    # 3 Normalize Node and settings #
    normalize = fmri_entities_layer.Normalize(**template_dict)

    # 4 Smoothing Node & Settings #
    smooth = fmri_entities_layer.Smooth(**template_dict)

    # 5 Datsink Node that collects swa files and writes to temp_write_dir #
    datasink = fmri_entities_layer.Datasink()

    ## 6 Create the pipeline/workflow and connect the nodes created above ##
    fmri_preprocess = pe.Workflow(name="fmri_preprocess")

    if template_dict['stc_flag']:

        fmri_preprocess.connect([

            # create_workflow_input(
            #    source=realign.node,
            #    target=slicetiming.node,
            #    source_output='realigned_files',
            #    source_output='mean_image',
            #    target_input='in_files'),
            # create_workflow_input(
            #    source=realign.node,
            #    target=normalize.node,
            #    source_output='realigned_files',
            #    source_output='mean_image',
            #    target_input='image_to_align'),
            create_workflow_input(
                source=slicetiming.node,
                target=realign.node,
                source_output='timecorrected_files',
                target_input='in_files'),
            create_workflow_input(
                source=realign.node,
                target=normalize.node,
                source_output='realigned_files',
                target_input='apply_to_files'),
            create_workflow_input(
                source=realign.node,
                target=normalize.node,
                source_output='realigned_files',
                #    source_output='mean_image',
                target_input='image_to_align'),
            create_workflow_input(
                source=normalize.node,
                target=smooth.node,
                source_output='normalized_files',
                target_input='in_files'),
            # create_workflow_input(
            #    source=realign.node,
            #    target=datasink.node,
            # source_output='mean_image',
            #    source_output='modified_in_files',
            #    target_input=template_dict['fmri_output_dirname']),
            create_workflow_input(
                source=realign.node,
                target=datasink.node,
                source_output='realigned_files',
                target_input=template_dict['fmri_output_dirname'] + '.@1'),
            create_workflow_input(
                source=realign.node,
                target=datasink.node,
                source_output='realignment_parameters',
                target_input=template_dict['fmri_output_dirname'] + '.@2'),
            create_workflow_input(
                source=slicetiming.node,
                target=datasink.node,
                source_output='timecorrected_files',
                target_input=template_dict['fmri_output_dirname'] + '.@3'),
            create_workflow_input(
                source=normalize.node,
                target=datasink.node,
                source_output='normalized_files',
                target_input=template_dict['fmri_output_dirname'] + '.@4'),
            create_workflow_input(
                source=smooth.node,
                target=datasink.node,
                source_output='smoothed_files',
                target_input=template_dict['fmri_output_dirname'] + '.@5')
        ])

    else:

        #print("No slice time correction")

        fmri_preprocess.connect([
            #    create_workflow_input(
            #        source=slicetiming.node,
            #        target=realign.node,
            #        source_output='timecorrected_files',
            #        target_input='in_files'),
            create_workflow_input(
                source=realign.node,
                target=normalize.node,
                source_output='realigned_files',
                target_input='apply_to_files'),
            create_workflow_input(
                source=realign.node,
                target=normalize.node,
                source_output='realigned_files',
                #    source_output='mean_image',
                target_input='image_to_align'),
            create_workflow_input(
                source=normalize.node,
                target=smooth.node,
                source_output='normalized_files',
                target_input='in_files'),
            # create_workflow_input(
            #    source=realign.node,
            #    target=datasink.node,
            # source_output='mean_image',
            #    source_output='modified_in_files',
            #    target_input=template_dict['fmri_output_dirname']),
            create_workflow_input(
                source=realign.node,
                target=datasink.node,
                source_output='realigned_files',
                target_input=template_dict['fmri_output_dirname'] + '.@1'),
            create_workflow_input(
                source=realign.node,
                target=datasink.node,
                source_output='realignment_parameters',
                target_input=template_dict['fmri_output_dirname'] + '.@2'),
            create_workflow_input(
                source=normalize.node,
                target=datasink.node,
                source_output='normalized_files',
                target_input=template_dict['fmri_output_dirname'] + '.@3'),
            create_workflow_input(
                source=smooth.node,
                target=datasink.node,
                source_output='smoothed_files',
                target_input=template_dict['fmri_output_dirname'] + '.@4')
        ])

    fmri_preprocess.write_graph(dotfilename='/tmp/graph.dot', graph2use='exec', format='png', simple_form=True)
    return [realign, slicetiming, datasink, fmri_preprocess]


def create_workflow_input(source, target, source_output, target_input):
    """This function collects pipeline nodes and their connections
    and returns them in appropriate format for nipype pipeline workflow
    """
    return (source, target, [(source_output, target_input)])


def run_pipeline(write_dir,
                 smri_data,
                 realign,
                 slicetiming,
                 datasink,
                 fmri_preprocess,
                 data_type=None,
                 **template_dict):
    """This function runs pipeline"""

    write_dir = write_dir + '/' + template_dict[
        'output_zip_dir']  # Store outputs in this directory for zipping the directory
    error_log = dict()  # dict for storing error log

    # print('fmri pipeline is setting up...')
    # print()
    # print("write_dir is", write_dir)
    each_sub = smri_data

    try:

        # Extract subject id and name of nifti file
        if data_type == 'bids':
            write_dir = write_dir + '/' + template_dict[
                'output_zip_dir']  # Store outputs in this directory for zipping the directory
            sub_id = 'sub-' + each_sub.subject
            session_id = getattr(each_sub, 'session', None)

            if session_id is not None:
                session = 'ses-' + getattr(each_sub, 'session', None)
            else:
                session = ''

            nii_output = ((each_sub.filename).split('/')[-1]).split('.gz')[0]
            with stdchannel_redirected(sys.stderr, os.devnull):
                n1_img = nib.load(each_sub.filename)

        if data_type == 'nifti':
            sub_id = template_dict['subject']
            session = template_dict['session']
            nii_output = ((each_sub).split('/')[-1]).split('.gz')[0]
            with stdchannel_redirected(sys.stderr, os.devnull):
                n1_img = nib.load(each_sub)

        if data_type == 'dicoms':
            sub_id = template_dict['subject']
            session = template_dict['session']
            fmri_out = os.path.join(write_dir, sub_id, session, 'func')
            os.makedirs(fmri_out, exist_ok=True)

            ## This code runs the dicom to nifti conversion here
            from nipype.interfaces.dcm2nii import Dcm2niix
            dcm_nii_convert = Dcm2niix()
            dcm_nii_convert.inputs.source_dir = each_sub
            dcm_nii_convert.inputs.output_dir = fmri_out
            with stdchannel_redirected(sys.stderr, os.devnull):
                dcm_nii_convert.run()
            with stdchannel_redirected(sys.stderr, os.devnull):
                n1_img = nib.load(glob.glob(os.path.join(fmri_out, '*.nii*'))[0])
                nii_output = ((glob.glob(os.path.join(fmri_out, '*.nii*'))[0]).split('/')[-1]).split('.gz')[0]

        # Directory in which fmri outputs will be written
        fmri_out = os.path.join(write_dir, sub_id, session, 'func')

        # print("fmri_out is", fmri_out)
        # print()

        # Create output dir for sub_id
        os.makedirs(fmri_out, exist_ok=True)

        if n1_img:

            """
            Save nifti file from input data into output directory only if data_type !=dicoms 
                 because the dcm_nii_convert in the previous step saves the nifti file to 
                 output directory
            """

            nib.save(n1_img, os.path.join(fmri_out, nii_output))
            # os.remove(glob.glob(os.path.join(fmri_out, '*.gz'))[0])

            # Create fmri_spm12 dir under the specific sub-id/func
            os.makedirs(
                os.path.join(fmri_out, template_dict['fmri_output_dirname']),
                exist_ok=True)

            nifti_file = glob.glob(os.path.join(fmri_out, '*.nii'))[0]
            orig_nifti = smri_data

            nt = n1_img.shape[3]

            # Dump nifti header contents to stdout :)

            hdr = n1_img.header
            # print()
            # print(hdr)
            # print()

            # Dump TR and number of slices

            TR = n1_img.header.get_zooms()[-1]
            num_slices = n1_img.shape[2]
            # print("TR = ", TR)
            # print("Number of slices is ", num_slices)
            # print()

            # Discard dummy scans, if number of volumes to remove is specified in input.json
            if template_dict['num_vols_to_remove'] != 0:
                # print(nifti_file)
                nf = discard_dummy_scans(fmri_out, nifti_file, nt, **template_dict)
                # print()
                # print(nifti_file)
                # print("new nifti ", nf)
                # print()
                nifti_file = nf

            # Attach SBRef, if specified in input.json file
            if template_dict['SBRef_file']:
                # print(nifti_file)
                nf = attachSBRef(fmri_out, smri_data, nifti_file, **template_dict)
                # print()
                # print(nifti_file)
                # print("new nifti ", nf)
                # print()
                nifti_file = nf

            # Run distortion correction, if specified in input.json file
            if template_dict['dist_corr']:
                nf = distortionCorrection(fmri_out, nifti_file, **template_dict)
                # print()
                # print(nifti_file)
                # print("new nifti ", nf)
                # print()
                nifti_file = nf

            # Edit realign node inputs

            realign.node.inputs.in_files = nifti_file
            # realign.node.inputs.out_file = fmri_out + "/" + template_dict['fmri_output_dirname'] + "/Re.nii"
            # realign.node.run()

            # Edit slicetiming node inputs
            slicetiming.node.inputs.in_files = nifti_file
            slicetiming.node.inputs.num_slices = num_slices
            slicetiming.node.inputs.time_repetition = TR
            time_for_one_slice = TR / num_slices
            slicetiming.node.inputs.time_acquisition = TR - time_for_one_slice
            # odd = range(1, num_slices + 1, 2)
            # even = range(2, num_slices + 1, 2)
            # acq_order = list(odd) + list(even)
            # slicetiming.node.inputs.slice_order = acq_order

            if template_dict['slicetime_ref_slice'] is not None:
                # print(template_dict['slicetime_ref_slice'])
                slicetiming.node.inputs.ref_slice = template_dict['slicetime_ref_slice']
            # else: slicetiming.node.inputs.ref_slice = int(num_slices / 2)

            if template_dict['slicetime_acq_order'] is not None:
                # print()
                # print(template_dict['slicetime_acq_order'])
                slicetiming.node.inputs.slice_order = template_dict['slicetime_acq_order']
            # else: slicetiming.node.inputs.slice_order = acq_order

            # Edit datasink node inputs
            datasink.node.inputs.base_directory = fmri_out

            # print()
            # print("SPM12 pipeline is running....")
            # print()

            # Run the nipype pipeline
            with stdchannel_redirected(sys.stderr, os.devnull):
                fmri_preprocess.run()

            # Motion quality control: Calculate Framewise Displacement
            calculate_FD(glob.glob(os.path.join(fmri_out, template_dict['fmri_output_dirname'],'rp*.txt'))[0], **template_dict)

            # Write readme files
            write_readme_files(write_dir, data_type, **template_dict)

            # Remove SBRef file, if specified in json file.
            if template_dict['SBRef_file'] ==False:
                rm_SBRef(fmri_out, nifti_file, **template_dict)

    except Exception as e:
        # If the above code fails for any reason update the error log for the subject id
        # ex: the nifti file is not a nifti file
        # the input file is not a brain scan
        error_log.update({sub_id: str(e) + str(traceback.format_exc())})

    finally:
        remove_tmp_files(fmri_out)

    output_message = "FMRI preprocessing completed."

    if bool(error_log):
        output_message = " Error log:" + str(error_log)
    return json.dumps({
        "output": {
            "message": output_message
        },
        "cache": {},
        "success": True
    })
