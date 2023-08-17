import numpy as np
from nibabel.freesurfer.mghformat import load
from tqdm import tqdm
from scipy.stats import pearsonr as corr
import os 

def median_squared_noisenorm_correlation(lh_fmri_val_pred, 
                                         rh_fmri_val_pred,
                                         lh_fmri_val,
                                         rh_fmri_val,
                                         data_dir,
                                         ncsnr_dir,
                                         images_trials_dir,
                                         idxs_val):
    ## Compute the correlation between the predicted and actual fMRI data ##
    print('Computing the correlation between the predicted and actual fMRI data...')
    # Empty correlation array of shape: (LH vertices)
    lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])
    # Correlate each predicted LH vertex with the corresponding ground truth vertex
    for v in tqdm(range(lh_fmri_val_pred.shape[1])):
        lh_correlation[v] = corr(lh_fmri_val_pred[:,v], lh_fmri_val[:,v])[0] # 0 per selezionare valore e non p-value

    # Empty correlation array of shape: (RH vertices)
    rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])
    # Correlate each predicted RH vertex with the corresponding ground truth vertex
    for v in tqdm(range(rh_fmri_val_pred.shape[1])):
        rh_correlation[v] = corr(rh_fmri_val_pred[:,v], rh_fmri_val[:,v])[0]

    ## Evaluate the model ##
    # NCSNR
    lh_ncsnr = load(os.path.join(ncsnr_dir, 'lh.ncsnr.mgh'))
    rh_ncsnr = load(os.path.join(ncsnr_dir, 'rh.ncsnr.mgh'))
    lh_ncsnr_all_vertices = lh_ncsnr.get_fdata()[:,0,0]
    rh_ncsnr_all_vertices = rh_ncsnr.get_fdata()[:,0,0]
    # fsaverage
    hemisphere = ['left', 'right'] #@param ['left', 'right'] {allow-input: true}
    # Load the brain surface map of all vertices
    roi_dir = os.path.join(data_dir, 'roi_masks',
        hemisphere[0][0]+'h.all-vertices_fsaverage_space.npy')
    lh_fsaverage_all_vertices = np.load(roi_dir)
    roi_dir = os.path.join(data_dir, 'roi_masks',
        hemisphere[1][0]+'h.all-vertices_fsaverage_space.npy')
    rh_fsaverage_all_vertices = np.load(roi_dir)
    # NCSNR for challenge vertices
    lh_ncsnr_challenge_vertices = lh_ncsnr_all_vertices[np.where(lh_fsaverage_all_vertices)[0]]
    rh_ncsnr_challenge_vertices = rh_ncsnr_all_vertices[np.where(rh_fsaverage_all_vertices)[0]]
    # TRIALS
    image_trial_number = np.load(os.path.join(images_trials_dir, 'train_images_trials.npy'))
    image_trial_number_val = image_trial_number[idxs_val]
    # Compute Noise Ceiling from NCSNR and TRIALS
    A = len(image_trial_number_val[image_trial_number_val == 3])
    B = len(image_trial_number_val[image_trial_number_val == 2])
    C = len(image_trial_number_val[image_trial_number_val == 1])
    lh_noise_ceiling = (lh_ncsnr_challenge_vertices ** 2) / ((lh_ncsnr_challenge_vertices ** 2) + ((A/3 + B/2 + C/1) / (A + B + C)))
    rh_noise_ceiling = (rh_ncsnr_challenge_vertices ** 2) / ((rh_ncsnr_challenge_vertices ** 2) + ((A/3 + B/2 + C/1) / (A + B + C)))
    # Compute Noise Normalized Squared Correlation
    
    # "Xh_correlation" are 1-D vectors with the correlation scores of all vertices
    # of a given Challenge subject (each component corresponds to the correlation
    # score of a vertex).

    # "Xh_noise_ceiling" are 1-D vectors with the noise ceiling values of all
    # vertices of a given Challenge subject (each component corresponds to the noise
    # ceiling value of a vertex).
    
    # Set negative correlation values to 0, so to keep the noise-normalized
    # encoding accuracy positive
    lh_correlation[lh_correlation<0] = 0
    rh_correlation[rh_correlation<0] = 0
    # Square the correlation values
    lh_correlation = lh_correlation ** 2
    rh_correlation = rh_correlation ** 2
    # Add a very small number to noise ceiling values of 0, otherwise the
    # noise-normalized encoding accuracy cannot be calculated (division by 0 is
    # not possible)
    lh_noise_ceiling[lh_noise_ceiling==0] = 1e-14
    rh_noise_ceiling[rh_noise_ceiling==0] = 1e-14
    # Compute the noise-normalized encoding accuracy
    lh_noise_norm_corr = np.divide(lh_correlation, lh_noise_ceiling)
    rh_noise_norm_corr = np.divide(rh_correlation, rh_noise_ceiling)
    # Set the noise-normalized encoding accuracy to 1 (100% accuracy) for those
    # vertices in which the correlation is higher than the noise ceiling, to prevent
    # encoding accuracy values higher than 100%
    lh_noise_norm_corr[lh_noise_norm_corr>1] = 1
    rh_noise_norm_corr[rh_noise_norm_corr>1] = 1
    
    return lh_noise_norm_corr, rh_noise_norm_corr


