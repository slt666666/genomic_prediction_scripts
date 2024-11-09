import pickle
import pandas as pd
import numpy as np

def get_all_accuracy(pic_file):
    with open(pic_file, 'rb') as p:
        summary_all_traits = pickle.load(p)

    r2_summary = []
    for summary in summary_all_traits:
        trait_name = summary[0]
        summary = summary[1]
        for i, each_CV in enumerate(summary):
            for each_model in each_CV:
                r2_summary.append([trait_name, i, each_model[0], each_model[2]])
    r2_summary = pd.DataFrame(r2_summary)
    r2_summary.columns = ["trait", "CV", "model", "r2"]
    r2_summary = r2_summary.groupby(["trait", "model"]).mean()

    models = r2_summary.index.get_level_values('model')[:10]
    traits = r2_summary.index.get_level_values('trait').unique()

    r2_summary = pd.DataFrame([r2_summary[r2_summary.index.get_level_values('trait') == i].r2.values for i in traits])
    r2_summary.index = traits
    r2_summary.columns = models
    return r2_summary

def get_RILs_accuracy(pic_file, trait_name):

    with open(pic_file, 'rb') as p:
        summary_all_traits = pickle.load(p)

    for summary in summary_all_traits:
        if summary[0] == trait_name:
            summary = summary[1]
            acc_RILs = np.repeat(0, len(summary[0])*len(summary[0][0][3][1])).reshape(len(summary[0]), len(summary[0][0][3][1]))
            for i, each_CV in enumerate(summary):
                each_acc_RILs = []
                models = []
                for each_model in each_CV:
                    models.append(each_model[0])
                    each_acc_RILs.append(each_model[3][0])
                each_acc_RILs = np.array(each_acc_RILs)
                acc_RILs = acc_RILs + each_acc_RILs
            acc_RILs = pd.DataFrame(acc_RILs / len(summary))
            acc_RILs.columns = each_model[3][1]
            acc_RILs.index = models
    return acc_RILs

def get_imps(pic_file, trait_name):

    with open(pic_file, 'rb') as p:
        summary_all_traits = pickle.load(p)

    for summary in summary_all_traits:
        if summary[0] == trait_name:
            summary = summary[1]
            imps = np.repeat(0, len(summary[0])*len(summary[0][2][4])).reshape(len(summary[0]), len(summary[0][2][4]))
            for i, each_CV in enumerate(summary):
                models = []
                each_imps = []
                for each_model in each_CV:
                    models.append(each_model[0])
                    if each_model[0] == "GBLUP":
                        imp = each_model[4].iloc[:, 3].values
                    elif each_model[0] == "LGBM":
                        imp = np.mean(each_model[4], axis=0)
                    else:
                        imp = each_model[4]
                    each_imps.append(imp)
                each_imps = np.array(each_imps)
                imps = imps + each_imps
            imps = pd.DataFrame(imps / len(summary))
            imps.index = models
    return imps