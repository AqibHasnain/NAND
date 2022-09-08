import numpy as np
import pandas as pd

def get_ara_lac_genes(all_genes_filter):
    '''
    all_genes_filter: list of gene names in same order as df_tpm_filter
    '''
    lac_inds = [ii for ii,this_gene in enumerate(all_genes_filter) if 'lac' in this_gene]
    ara_inds = [ii for ii,this_gene in enumerate(all_genes_filter) if 'ara' in this_gene]
    lac_genes = [this_gene for ii,this_gene in enumerate(all_genes_filter) if 'lac' in this_gene]
    ara_genes = [this_gene for ii,this_gene in enumerate(all_genes_filter) if 'ara' in this_gene]
    my_genes = ara_genes + lac_genes
    my_inds = ara_inds + lac_inds
    return my_genes, my_inds

def get_rpoS_genes(all_genes_filter):

    rpoS_list = ['rpoS','acs','adhE','aldB','alkA','appY','cpxA','cpxR','dps','ftsQ','ftsA','ftsZ','katG',\
                'narZ','narY','narW','narV','nhaA','osmC','osmY','proP']
    my_inds = [ii for ii,this_gene in enumerate(all_genes_filter) if this_gene in rpoS_list]
    my_genes = [this_gene for ii,this_gene in enumerate(all_genes_filter) if this_gene in rpoS_list]
    
    return my_genes, my_inds

def get_DE_genes(res_dir_list,all_genes_filter,p_thresh=0.05,fc_thresh=2): 
    '''
    res_dir_list: list of directories for where to find DE results from DESeq2    

    '''
    res_list = []
    for res_dir in res_dir_list:  
        res_df = pd.read_csv(res_dir)
        # first filter by padj
        filter_p = res_df.loc[res_df.padj <  p_thresh]
        # next by FC
        res_filter_df  = filter_p.loc[np.abs(filter_p.log2FoldChange)>=\
                                                    np.log2(fc_thresh)]
        res_list.append(res_filter_df)

    # take the union of the genes remaining in each condition
    genes_DE = set()
    for res in res_list: 
        genes_DE = genes_DE.union(res.iloc[:,0]) # the first row contains gene names
    genes_DE = list(genes_DE)

    my_inds = [ii for ii,this_gene in enumerate(all_genes_filter) \
               for jj,DE_gene in enumerate(genes_DE) if  DE_gene == this_gene]
    my_genes = [all_genes_filter[ii] for ii in my_inds]

    return my_genes, my_inds

def get_acrR_genes(all_genes_filter):
    acrR_list = ['acrR','acrA','acrB']
    my_inds = [ii for ii,this_gene in enumerate(all_genes_filter) if this_gene in acrR_list]
    my_genes = [this_gene for ii,this_gene in enumerate(all_genes_filter) if this_gene in acrR_list]
    
    return my_genes, my_inds

def get_argR_genes(all_genes_filter):
    acrR_list = ['argR','argC','argB','argH','argD','argE','argF','argI','carA','carB']
    my_inds = [ii for ii,this_gene in enumerate(all_genes_filter) if this_gene in acrR_list]
    my_genes = [this_gene for ii,this_gene in enumerate(all_genes_filter) if this_gene in acrR_list]
    
    return my_genes, my_inds

def get_xylR_genes(all_genes_filter):
    xylR_list = ['xylR','xylA','xylB','xylF','xylG','xylH']
    my_inds = [ii for ii,this_gene in enumerate(all_genes_filter) if this_gene in xylR_list]
    my_genes = [this_gene for ii,this_gene in enumerate(all_genes_filter) if this_gene in xylR_list]
    
    return my_genes, my_inds

    
def get_lac_xyl_genes(all_genes_filter):
    '''
    all_genes_filter: list of gene names in same order as df_tpm_filter
    '''
    lac_inds = [ii for ii,this_gene in enumerate(all_genes_filter) if 'lac' in this_gene]
    xyl_inds = [ii for ii,this_gene in enumerate(all_genes_filter) if 'xyl' in this_gene]
    lac_genes = [this_gene for ii,this_gene in enumerate(all_genes_filter) if 'lac' in this_gene]
    xyl_genes = [this_gene for ii,this_gene in enumerate(all_genes_filter) if 'xyl' in this_gene]
    my_genes = lac_genes + xyl_genes
    my_inds = lac_inds + xyl_inds
    
    return my_genes, my_inds    