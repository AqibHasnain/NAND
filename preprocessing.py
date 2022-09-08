import numpy as np
import pandas as pd
from  sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
import csv
import sys

def snapshots_from_df(df):

    strains = ['wt','icar','phlf','nand']
    tps = ['5','18']
    temps = ['30','37']
    inducers = ['00','10','01','11']

    # create a dictionary where you specify strain, temp, and inducers as keys to grab the snapshot matrices
    snapshot_dict = {}
    for strain in strains: 
        snapshot_dict[strain] = {}
        for temp in temps: 
            snapshot_dict[strain][temp] = {}
            for inducer in inducers: 
                snapshot_dict[strain][temp][inducer] = {} # keys are to be Yf and Yp
                # get substring of colname that identifies the group (so everything except rep number)
                yp_colname = strain + '_' + inducer + temp + tps[0]
                # get list of indices that correspond to this group (in short, grabbing inds for all replicates)
                yp_col_inds = [ii for ii, this_col in enumerate(list(df.columns)) if yp_colname in this_col]
                snapshot_dict[strain][temp][inducer]['Yp'] = np.array(df.iloc[:,yp_col_inds])
                # do the same for the 18 hours timepoint i.e. Yf
                yf_colname = strain + '_' + inducer + temp + tps[1]
                yf_col_inds = [ii for ii, this_col in enumerate(list(df.columns)) if yf_colname in this_col]
                snapshot_dict[strain][temp][inducer]['Yf'] = np.array(df.iloc[:,yf_col_inds])
                
    return snapshot_dict

def get_unpaired_samples(df):
    # filter the samples that don't have a timepoint pair due to low sequencing depth
    unpaired_samples = []
    for sample in df.columns: 
        if '5' in sample:
            if sample.replace('5','18') not in df.columns:
                unpaired_samples.append(sample)
        elif '18' in sample: 
            if sample.replace('18','5') not in df.columns:
                unpaired_samples.append(sample)
    return unpaired_samples

def apply_normalizer(Yp,Yf):
    # normalize each datapoint to have unit norm
    transformer1 = Normalizer().fit(Yp.T)
    Yp_normed = transformer1.transform(Yp.T).T

    transformer2 = Normalizer().fit(Yf.T)
    Yf_normed = transformer2.transform(Yf.T).T
    return Yp_normed,Yf_normed

def apply_biased_StandardScaler(Yp,Yf):
    '''
    First each gene is given zero mean and unit variance (feature scaling rather than observation scaling).
    Then the entire dataset is shifted to be positive by adding the minimum value to each gene 
    '''
    # this function will only work for two-timepoint case. for more than two timepoints, i.e. where all but two timepoint is shared in Yp and Yf, need to reconsider
    Y = np.hstack((Yp,Yf))
    Ymean = np.mean(Y,axis=1)[:,np.newaxis]
    Ystd = np.std(Y,axis=1)[:,np.newaxis]
    Y_normed = (Y - Ymean)/Ystd
    Y_normed = Y_normed + (-np.nanmin(Y_normed))
    ndatapts = Yp.shape[1]
    Yp_normed = Y_normed[:,:ndatapts]
    Yf_normed = Y_normed[:,ndatapts:]
    return Yp_normed,Yf_normed

def apply_MinMaxScaler(Yp,Yf):
    '''
    Each gene will have range [0,1] (feature scaling rather than observation scaling).
    '''
    # this function will only work for two-timepoint case. for more than two timepoints, i.e. where all but two timepoint is shared in Yp and Yf, need to reconsider
    Y = np.hstack((Yp,Yf))
    Ymin = np.min(Y,axis=1)[:,np.newaxis]
    Ymax = np.max(Y,axis=1)[:,np.newaxis]
    Y_normed = (Y - Ymin)/(Ymax-Ymin)
    ndatapts = Yp.shape[1]
    Yp_normed = Y_normed[:,:ndatapts]
    Yf_normed = Y_normed[:,ndatapts:]
    return Yp_normed,Yf_normed