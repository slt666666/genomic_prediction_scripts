import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import StratifiedKFold


def make_data(family, family_list, trait_name, genotype, phenotype, CV=5, seed=1024, plot=True, log=True):
    # read dataset
    phenotype = pd.read_csv(phenotype, index_col=0)
    genotype = pd.read_csv(genotype)
    genotype = genotype.loc[:, genotype.isna().sum() < genotype.shape[0]/2] # remove many NA columns
    # extract RILs derived from selected family
    RIL_ids = family_list.loc[family_list.family.isin(family), "id"].values
    RIL_ids = list(set(RIL_ids) & set(phenotype["Line"]) & set(genotype.columns[3:]))
    RIL_ids.sort()
    # extract phenotype & genotype data of RILs
    phenotype = phenotype[phenotype.Line.isin(RIL_ids)]
    genotype_columns = ["chr", "pos", "SNP_type"]
    genotype_columns.extend(RIL_ids)
    genotype = genotype.loc[:, genotype_columns]
    # make phenotype & genotype data
    phenotype_data = phenotype.loc[:, ["Line", trait_name]]
    phenotype_data = phenotype_data.set_index("Line")
    position_data = genotype.loc[:, ["chr", "pos", "SNP_type"]]
    genotype = genotype.drop(genotype.columns[0:3], axis=1)
    genotype = genotype.T
    if log:
        print("The phenotype data:", phenotype_data[trait_name].notna().sum(), "The genotype data:", genotype.shape[0])
    # merge phenotype & genotype data
    merge_data = pd.concat([phenotype_data, genotype], axis=1, join="inner")
    merge_data_index = merge_data.index.values
    # remove lines without phenotype values & common genotype SNP across all lines
    merge_data = np.array(merge_data)
    merge_data_index = merge_data_index[~np.isnan(merge_data)[:, 0]]
    merge_data = merge_data[~np.isnan(merge_data)[:, 0]]
    merge_data = pd.DataFrame(merge_data)
    merge_data.index = merge_data_index
    if log:
        print("The merge data:", merge_data.shape)

    del genotype, phenotype, phenotype_data

    # separate dataset to train, test.
    train_data = []
    test_data = []
    skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=seed)
    for train, test in skf.split(range(merge_data.shape[0]),  merge_data.index.str[:3].values):
        train_data.append([merge_data.iloc[train, 1:], merge_data.iloc[train, 0]])
        test_data.append([merge_data.iloc[test, 1:], merge_data.iloc[test, 0]])
    
    del merge_data
    
    if plot:
        fig = plt.figure(figsize=(15,3))
        for i in range(CV):
            ax1 = fig.add_subplot(1, 5, i+1)
            sns.histplot(train_data[i][1], stat="density", color="r", label="train", ax=ax1)
            sns.histplot(test_data[i][1], stat="density", color="b", label="test", ax=ax1)
        plt.legend()
        plt.show()

    return test_data, train_data, position_data

# deprecated
def make_data_for_Tuning(family, trait_name, genotype, phenotype, CV=5, seed=1024, plot=True, log=True):
    phenotype = pd.read_csv(phenotype, index_col=0)
    phenotype = phenotype[phenotype.Line.str.contains("|".join(RIL_set.value[family]))]
    genotype = pd.read_csv(genotype)
    genotype = genotype.loc[:, genotype.isna().sum() < genotype.shape[0]/2] # remove many NA lines if genotype is selected HBs
    # genotype = genotype.fillna(1)
    genotype_columns = ["chr", "pos", "SNP_type"]
    line_ids = list(set(genotype.columns[3:]) & set(phenotype["Line"]))
    line_ids.sort()
    genotype_columns.extend(line_ids)
    genotype = genotype.loc[:, genotype_columns]

    # make phenotype & genotype data
    phenotype_data = phenotype.loc[:, ["Line", trait_name]]
    phenotype_data = phenotype_data.set_index("Line")
    position_data = genotype.loc[:, ["chr", "pos", "SNP_type"]]
    genotype = genotype.drop(genotype.columns[0:3], axis=1)
    genotype = genotype.T
    if log:
        print("The phenotype data:", phenotype_data.shape[0], "The genotype data:", genotype.shape[0])

    merge_data = pd.concat([phenotype_data, genotype], axis=1, join="inner")
    merge_data_index = merge_data.index.values
    # remove lines without phenotype values & common genotype SNP across all lines
    merge_data = np.array(merge_data)
    merge_data_index = merge_data_index[~np.isnan(merge_data)[:, 0]]
    merge_data = merge_data[~np.isnan(merge_data)[:, 0]]
    # merge_data = merge_data[:, sum(np.isnan(merge_data)) == 0]
    merge_data = pd.DataFrame(merge_data)
    merge_data.index = merge_data_index
    if log:
        print("The merge data:", merge_data.shape)

    del genotype, phenotype, phenotype_data
        
    return merge_data.iloc[:, 1:], merge_data.iloc[:, 0]

