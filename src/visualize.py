from time import strftime
import os
import os.path as osp

import numpy as np
import pandas as pd
# import kaleido
import plotly.express as px
#import plotly.io as pio
import json

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

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
    
    # Add All Vertices Correlations
    
    roi_names.append('All vertices')
    lh_roi_correlation.append(lh_correlation)
    rh_roi_correlation.append(rh_correlation)
    
    # Add Unknown Functional/Stream/Both ROIs Correlations (using the generated files)
        
    lh_unknown_roi_files = ['lh.unknown_functional_ROI_masks_challenge_space.npy', 
                            'lh.unknown_streams_ROI_masks_challenge_space.npy',
                            'lh.unknown_ROI_masks_challenge_space.npy']
    rh_unknown_roi_files = ['rh.unknown_functional_ROI_masks_challenge_space.npy', 
                            'rh.unknown_streams_ROI_masks_challenge_space.npy',
                            'rh.unknown_ROI_masks_challenge_space.npy']
    
    roi_names.append('Unknown ROI')
    roi_names.append('Unknown Stream')
    roi_names.append('Unknown')
    
    for l, r in zip(lh_unknown_roi_files, rh_unknown_roi_files):
        lh_challenge_unknown = np.load(osp.join(roi_path, 'roi_masks_enhanced', 
                                                'unknown_masks', l))
        lh_roi_correlation.append(lh_correlation[lh_challenge_unknown != 0])
        rh_challenge_unknown = np.load(osp.join(roi_path, 'roi_masks_enhanced', 
                                                'unknown_masks', r))
        rh_roi_correlation.append(rh_correlation[rh_challenge_unknown != 0])

    
    return roi_names, lh_roi_correlation, rh_roi_correlation

def histogram(roi_path, lh_correlation, rh_correlation, title, save=None, filename=None):
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
        np.nanmedian(corr) if len(corr) > 0 else np.nan for corr in lh_roi_correlation]
    rh_median_roi_correlation = [
        np.nanmedian(corr) if len(corr) > 0 else np.nan for corr in rh_roi_correlation]

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

        fig.write_html(to_save+str([filename if filename is not None else ""][0])+".html")
        fig.write_image(to_save+str([filename if filename is not None else ""][0])+".png")
        # pio.write_image(fig, to_save+".png")
        
        # Save roi correlation data
        df.apply(lambda row: "{} {}: {:.2f}".format(row['ROIs'], row['Hemisphere'], row['Median Noise Normalized Encoding Accuracy']), axis=1).to_csv(osp.join(save,'scores_subj_hemishphere_roi.txt'), index=False, header=False)

    return fig


def box_plot(roi_path, lh_correlation, rh_correlation, title, save=None, filename=None):
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

        fig.write_html(to_save+ str([filename if filename is not None else ""][0])+".html")
        fig.write_image(to_save+ str([filename if filename is not None else ""][0])+".png")
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
        np.nanmedian(corr) if len(corr) > 0 else np.nan for corr in lh_roi_correlation]
    rh_median_roi_correlation = [
        np.nanmedian(corr) if len(corr) > 0 else np.nan for corr in rh_roi_correlation]

    # Save the median correlation for each ROI and each hemisphere
    df = pd.DataFrame({"ROIs": roi_names + roi_names, "Median Noise Normalized Encoding Accuracy": lh_median_roi_correlation + rh_median_roi_correlation,
                       "Hemisphere": ["Left"] * len(lh_roi_correlation) + ["Right"] * len(rh_roi_correlation)})
    
    return df

def noise_norm_corr_ROI_df(roi_path, lh_correlation, rh_correlation, model_layer_id):
    """
        Create a dataframe with the correlation results classified by ROI
    """
    roi_names, lh_roi_correlation, rh_roi_correlation = classiy_correlation_by_roi(
        roi_path, lh_correlation, rh_correlation)

    # I concatenate lh and rh correlation for each ROI to choose a single layer for each ROI
    roi_correlation = []
    if len(lh_roi_correlation) == len(rh_roi_correlation):
        # Ciclo per concatenare gli elementi corrispondenti delle due liste
        for vettore1, vettore2 in zip(lh_roi_correlation, rh_roi_correlation):
            vettore_concatenato = np.concatenate((vettore1, vettore2), axis=0)
            roi_correlation.append(vettore_concatenato)
    else:
        print("Lists have different length")
    
    median_roi_correlation = [[
        np.nanmedian(corr) if len(corr) > 0 else np.nan for corr in roi_correlation]]

    # Save the median correlation for each ROI
    median_roi_correlation_df = pd.DataFrame(median_roi_correlation, columns=roi_names, index=[model_layer_id])

    return median_roi_correlation_df

def find_best_performing_layer(median_roi_correlation_df, parent_config_dir, save):
    """
        Find the best performing layer for each ROI and save it in a dictionary
    """	
    # find the index, for each roi, associated with the maximum value (model+layer)
    max_indices = median_roi_correlation_df.idxmax()
    final_dict = {col: idx.split('+') if not pd.isna(idx) else np.NaN for col, idx in max_indices.items()}
    # If a layer is a concatenation of more than one layer, split it into a list instead of a string
    for chiave, valore in final_dict.items():
        # Se il valore non è una lista, continuiamo senza apportare modifiche
        if not isinstance(valore, list):
            continue
        # Controlliamo se il secondo elemento contiene "&" (quindi è sono più di un layer concatenati)
        if valore is not np.NaN:
            print(valore)
            if '&' in str(valore[1]):
                # Se contiene "&", splittiamolo in una lista di stringhe
                final_dict[chiave][1] = str(valore[1]).split('&')
    if save:
        average_value = median_roi_correlation_df.max().mean()
        filename = "config_" + str(average_value.round(5)) + ".json"
        full_path = os.path.join(parent_config_dir, filename)

        # Salvataggio come JSON
        with open(full_path, 'w') as file:
            json.dump(final_dict, file, indent=4)
     
    #print(median_roi_correlation_df)
    #print('\n')
    #print(json.dumps(final_dict, indent=4))
    return final_dict

def json_config_to_feature_extraction_dict_5(config_dict):
    new_dict = {}
    for key, value in config_dict.items():
        # If the value is not a list or is NaN, skip this iteration
        if not isinstance(value, list) or (isinstance(value, float) and math.isnan(value)):
            continue
        
        # model, layer/s, transform, regression_type
        first_string, second_string, third_string, fourth_string, fifth_string = value
        
        # If the first string is not a key in the new dictionary, add an empty list
        if first_string not in new_dict:
            new_dict[first_string] = []
        
        # Add the second string to the corresponding list only if it's not already present
        if [second_string, third_string, fourth_string, fifth_string] not in new_dict[first_string]:
            new_dict[first_string].append([second_string, third_string, fourth_string, fifth_string])

    # Sort the second strings in each list alphabetically
    # for first_string in new_dict:
    #     new_dict[first_string].sort()
    return new_dict

def json_config_to_feature_extraction_dict_6(config_dict):
    new_dict = {}
    for key, value in config_dict.items():
        # If the value is not a list or is NaN, skip this iteration
        if not isinstance(value, list) or (isinstance(value, float) and math.isnan(value)):
            continue
        
        # model, layer/s, transform, regression_type
        first_string, second_string, third_string, fourth_string, fifth_string, sixth_string = value
        
        # If the first string is not a key in the new dictionary, add an empty list
        if first_string not in new_dict:
            new_dict[first_string] = []
        
        # Add the second string to the corresponding list only if it's not already present
        if [second_string, third_string, fourth_string, fifth_string, sixth_string] not in new_dict[first_string]:
            new_dict[first_string].append([second_string, third_string, fourth_string, fifth_string, sixth_string])

    # Sort the second strings in each list alphabetically
    # for first_string in new_dict:
    #     new_dict[first_string].sort()
    return new_dict

def json_config_to_feature_extraction_dict(config_dict):
    new_dict = {}
    for key, value in config_dict.items():
        # If the value is not a list or is NaN, skip this iteration
        if not isinstance(value, list) or (isinstance(value, float) and math.isnan(value)):
            continue
        
        # model, layer/s, transform, regression_type
        first_string, second_string, third_string, fourth_string = value
        
        # If the first string is not a key in the new dictionary, add an empty list
        if first_string not in new_dict:
            new_dict[first_string] = []
        
        # Add the second string to the corresponding list only if it's not already present
        if [second_string, third_string, fourth_string] not in new_dict[first_string]:
            new_dict[first_string].append([second_string, third_string, fourth_string])

    # Sort the second strings in each list alphabetically
    # for first_string in new_dict:
    #     new_dict[first_string].sort()
    return new_dict

def json_config_test_models_layers(config_dict):
    new_dict = {}
    for key, value in config_dict.items():
        # If the value is not a list or is NaN, skip this iteration
        if not isinstance(value, list) or (isinstance(value, float) and math.isnan(value)):
            continue
        
        # model, layer/s, transform, regression_type
        first_string, second_string, third_string, fourth_string = value
        
        # If the first string is not a key in the new dictionary, add an empty list
        if first_string not in new_dict:
            new_dict[first_string] = []
        
        # Add the second string to the corresponding list only if it's not already present
        if [second_string, third_string, fourth_string] not in new_dict[first_string]:
            new_dict[first_string].append([second_string, third_string, fourth_string])

    # Sort the second strings in each list alphabetically
    # for first_string in new_dict:
    #     new_dict[first_string].sort()
    return new_dict

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

def median_squared_noisenorm_correlation_dataframe_results(csv_file_path, noise_norm_corr_dict, subj):
    lh_value = np.nanmedian(noise_norm_corr_dict[f'lh_{subj}'])*100
    rh_value = np.nanmedian(noise_norm_corr_dict[f'rh_{subj}'])*100
    total_value = np.nanmedian(np.concatenate((noise_norm_corr_dict[f'lh_{subj}']*100,noise_norm_corr_dict[f'rh_{subj}']*100)))
    if not os.path.exists(csv_file_path):
        data = {'LH': [lh_value], 'RH': [rh_value], 'Total': [total_value]}
        df = pd.DataFrame(data, index=["Subj0" + subj])
        df.to_csv(csv_file_path)
        print(f"File {csv_file_path} created.")
    else:
        # Se il file CSV esiste, carica il DataFrame esistente e aggiungi una riga
        df = pd.read_csv(csv_file_path, index_col=0)
        if "Subj0" + subj not in df.index:
            new_row = pd.DataFrame({'LH': [lh_value], 'RH': [rh_value], 'Total': [total_value]}, index=["Subj0" + subj])
            df = pd.concat([df, new_row])
            df.to_csv(csv_file_path)
            # print(f"Riga aggiunta al file {csv_file_path}.")
        #else:
            # print(f"La riga con indice 'Subj{subj}' esiste già nel file {csv_file_path}.")
      
# def find_best_performing_layer(median_roi_correlation_df, parent_config_dir, save, regression_type):
#     """
#         Find the best performing layer for each ROI and save it in a dictionary
#     """	
#     # find the index, for each roi, associated with the maximum value (model+layer)
#     max_indices = median_roi_correlation_df.idxmax()
#     final_dict = {col: idx.split('+') if not pd.isna(idx) else np.NaN for col, idx in max_indices.items()}
#     # If a layer is a concatenation of more than one layer, split it into a list instead of a string
#     for chiave, valore in final_dict.items():
#         # Se il valore non è una lista, continuiamo senza apportare modifiche
#         if not isinstance(valore, list):
#             continue
#         # Controlliamo se il secondo elemento contiene "&" (quindi è sono più di un layer concatenati)
#         if valore is not np.NaN:
#             print(valore)
#             if '&' in str(valore[1]):
#                 # Se contiene "&", splittiamolo in una lista di stringhe
#                 final_dict[chiave][1] = str(valore[1]).split('&')
#     if save:
#         average_value = median_roi_correlation_df.max().mean()
#         filename = "config_" + regression_type + "_" + str(average_value.round(5)) + ".json"
#         full_path = os.path.join(parent_config_dir, filename)

#         # Salvataggio come JSON
#         with open(full_path, 'w') as file:
#             json.dump(final_dict, file, indent=4)
     
#     print(median_roi_correlation_df)
#     print('\n')
#     print(json.dumps(final_dict, indent=4))
#     return final_dict

# def json_config_to_feature_extraction_dict(config_dict):
#     new_dict = {}
#     for key, value in config_dict.items():
#         # If the value is not a list or is NaN, skip this iteration
#         if not isinstance(value, list) or (isinstance(value, float) and math.isnan(value)):
#             continue
#         first_string, second_string = value
        
#         # If the first string is not a key in the new dictionary, add an empty list
#         if first_string not in new_dict:
#             new_dict[first_string] = []
        
#         # Add the second string to the corresponding list only if it's not already present
#         if second_string not in new_dict[first_string]:
#             new_dict[first_string].append(second_string)

#     # Sort the second strings in each list alphabetically
#     # for first_string in new_dict:
#     #     new_dict[first_string].sort()
#     return new_dict