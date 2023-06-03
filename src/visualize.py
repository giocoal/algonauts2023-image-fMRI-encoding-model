from time import strftime
import os
import os.path as osp

import numpy as np
import pandas as pd
# import kaleido
import plotly.express as px
#import plotly.io as pio


def classiy_correlation_by_roi(roi_path, lh_correlation, rh_correlation):
    # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
                         'mapping_floc-faces.npy', 'mapping_floc-places.npy',
                         'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(osp.join(roi_path, 'roi_masks', r),
                                     allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
                              'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
                              'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
                              'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
                              'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
                              'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
                              'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for l, r in zip(lh_challenge_roi_files, rh_challenge_roi_files):
        lh_challenge_rois.append(
            np.load(osp.join(roi_path, 'roi_masks', l)))
        rh_challenge_rois.append(
            np.load(osp.join(roi_path, 'roi_masks', r)))

    # Select the correlation results vertices of each ROI
    roi_names = []
    lh_roi_correlation = []
    rh_roi_correlation = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0:  # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                rh_roi_correlation.append(rh_correlation[rh_roi_idx])
    roi_names.append('All vertices')
    lh_roi_correlation.append(lh_correlation)
    rh_roi_correlation.append(rh_correlation)

    return roi_names, lh_roi_correlation, rh_roi_correlation


def histogram(roi_path, lh_correlation, rh_correlation, title, save=None):
    """
        Visualize the correlation result
        Args:
            roi_path,           str, path to ROI masks
            lh_correlation,     np.ndarray, left hemisphere correlation
            rh_correlation,     np.ndarray, right hemishphere correlation
            title,              str, title of the figure
            save,               str, where to save the graph
    """

    roi_names, lh_roi_correlation, rh_roi_correlation = classiy_correlation_by_roi(
        roi_path, lh_correlation, rh_correlation)

    lh_median_roi_correlation = [
        np.median(corr) if len(corr) > 0 else np.nan for corr in lh_roi_correlation]
    rh_median_roi_correlation = [
        np.median(corr) if len(corr) > 0 else np.nan for corr in rh_roi_correlation]

    df = pd.DataFrame({"ROIs": roi_names + roi_names, "Median Noise Normalized Encoding Accuracy": lh_median_roi_correlation + rh_median_roi_correlation,
                       "Hemisphere": ["Left"] * len(lh_roi_correlation) + ["Right"] * len(rh_roi_correlation)})
    # draw the diagram
    fig = px.histogram(df, x="ROIs", y="Median Noise Normalized Encoding Accuracy", color="Hemisphere",
                       hover_data=df.columns.tolist(), barmode="group", width=1500, height=500)
    fig.update_xaxes(categoryorder='array',
                     categoryarray=roi_names, tickangle=45)
    fig.update_layout(title_text=title, yaxis=dict(range=[0.0, 1.0]), yaxis_title="Median Noise Normalized Encoding Accuracy", xaxis_title="ROIs")

    if save is not None:

        # if not osp.isdir(save):
        #     os.makedirs(save)

        to_save = osp.join(save, "histogram_pearson_{}".format(
            strftime("%Y%m%d%H%M%S")))

        fig.write_html(to_save+".html")
        fig.write_image(to_save+".png")
        # pio.write_image(fig, to_save+".png")
        
        # Save roi correlation data
        df.apply(lambda row: "{} {}: {:.2f}".format(row['ROIs'], row['Hemisphere'], row['Median Noise Normalized Encoding Accuracy']), axis=1).to_csv(osp.join(save,'scores_subj_hemishphere_roi.txt'), index=False, header=False)

    return fig


def box_plot(roi_path, lh_correlation, rh_correlation, title, save=None):
    """
        Visualize the correlation result
        Args:
            roi_path,           str, path to ROI masks
            lh_correlation,     np.ndarray, left hemisphere correlation
            rh_correlation,     np.ndarray, right hemishphere correlation
            title,              str, title of the figure
            save,               str, where to save the graph
    """

    roi_names, lh_roi_correlation, rh_roi_correlation = classiy_correlation_by_roi(
        roi_path, lh_correlation, rh_correlation)

    r = list()
    p = list()
    h = list()
    for i in range(len(roi_names)):
        r += [roi_names[i]] * len(lh_roi_correlation[i])
        p += lh_roi_correlation[i].tolist()
        h += ["Left"] * len(lh_roi_correlation[i])

        r += [roi_names[i]] * len(rh_roi_correlation[i])
        p += rh_roi_correlation[i].tolist()
        h += ["Right"] * len(rh_roi_correlation[i])

    df = pd.DataFrame({"ROIs": r, "Noise Normalized Encoding Accuracy": p, "Hemisphere": h})

    # draw the diagram
    fig = px.box(df, x="ROIs", y="Noise Normalized Encoding Accuracy", color="Hemisphere",
                 hover_data=df.columns.tolist(), width=1500, height=500)
    fig.update_xaxes(categoryorder='array',
                     categoryarray=roi_names, tickangle=45)
    fig.update_layout(title_text=title)

    if save is not None:

        # if not osp.isdir(save):
        #     os.makedirs(save)

        to_save = osp.join(save, "box_pearson_{}".format(
            strftime("%Y%m%d%H%M%S")))

        fig.write_html(to_save+".html")
        fig.write_image(to_save+".png")
        # pio.write_image(fig, to_save+".png")

    return fig

def noise_norm_corr_ROI(roi_path, lh_correlation, rh_correlation, save):
    """
        Save the median correlation result to a dataframe
        indipendently for each ROI and each hemisphere
    """

    roi_names, lh_roi_correlation, rh_roi_correlation = classiy_correlation_by_roi(
        roi_path, lh_correlation, rh_correlation)

    lh_median_roi_correlation = [
        np.median(corr) if len(corr) > 0 else np.nan for corr in lh_roi_correlation]
    rh_median_roi_correlation = [
        np.median(corr) if len(corr) > 0 else np.nan for corr in rh_roi_correlation]

    # Save the median correlation for each ROI and each hemisphere
    df = pd.DataFrame({"ROIs": roi_names + roi_names, "Median Noise Normalized Encoding Accuracy": lh_median_roi_correlation + rh_median_roi_correlation,
                       "Hemisphere": ["Left"] * len(lh_roi_correlation) + ["Right"] * len(rh_roi_correlation)})
    
    return df

def final_subj_corr_dataframe_boxplot_istograms(noise_norm_corr_dict, title, save=None):
    # Create lists for the dataframe columns
    subject_list = []
    hemi_list = []
    acc_list = []
    
    # Loop attraverso le chiavi e i valori del dizionario
    for key, value in noise_norm_corr_dict.items():
        # Estrai il numero di soggetto dalla chiave e assegnalo alla variabile Subject
        subject = int(key.split('_')[1])
        
        # Estrai le prime due lettere della chiave per determinare l'emisfero e assegnalo alla variabile Hemisphere
        hemisphere = key.split('_')[0]
        
        # Loop attraverso ogni valore del vettore e aggiungi una riga al dataframe per ogni valore
        for val in value:
            # Aggiungi Subject, Hemisphere e il valore corrente alla rispettiva lista
            subject_list.append(subject)
            hemi_list.append(hemisphere)
            acc_list.append(val)

    # Creare un dizionario dei dati utilizzando le liste create in precedenza
    data = {'Subject': subject_list,
            'Hemisphere': hemi_list,
            'Noise Normalized Encoding Accuracy': acc_list}

    # Creare il dataframe utilizzando il dizionario dei dati
    data = pd.DataFrame(data)
    
    ### Boxplot showing the distribution of the noise normalized encoding accuracy for each subject and hemisphere
    fig = px.box(data, x="Subject", y="Noise Normalized Encoding Accuracy", color="Hemisphere",
                 hover_data=data.columns.tolist(), width=1500, height=500)
    # fig.update_xaxes(categoryorder='array',
    #                  categoryarray=roi_names, tickangle=45)
    fig.update_layout(title_text=title)

    if save is not None:

        # if not osp.isdir(save):
        #     os.makedirs(save)

        to_save = osp.join(save, "subj-wise_correlation_boxplot")

        fig.write_html(to_save+".html")
        fig.write_image(to_save+".png")
        # pio.write_image(fig, to_save+".png")
        
    # Histogram showing the the median noise normalized encoding accuracy for each subject and hemisphere
    
    # Create grouped dataframe on hemisphere and find median noise normalized encoding accuracy for each subject and hemisphere
    grouped_df = data.groupby(['Subject', 'Hemisphere'])['Noise Normalized Encoding Accuracy'].median().reset_index().rename(columns={'Noise Normalized Encoding Accuracy': 'Median Noise Normalized Encoding Accuracy'})
    fig = px.histogram(grouped_df, x="Subject", y="Median Noise Normalized Encoding Accuracy", color="Hemisphere",
                       hover_data=grouped_df.columns.tolist(), barmode="group", width=1500, height=500)
    # fig.update_xaxes(categoryorder='array',
    #                  categoryarray=roi_names, tickangle=45)
    fig.update_layout(title_text=title, yaxis=dict(range=[0.0, 1.0]), yaxis_title="Median Noise Normalized Encoding Accuracy", xaxis_title="Subject")
    
    if save is not None:

        # if not osp.isdir(save):
        #     os.makedirs(save)

        to_save = osp.join(save, "subj-wise_correlation_histogram")

        fig.write_html(to_save+".html")
        fig.write_image(to_save+".png")
        # pio.write_image(fig, to_save+".png")
        
