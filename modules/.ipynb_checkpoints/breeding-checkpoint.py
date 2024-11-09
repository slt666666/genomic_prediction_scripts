import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import seaborn as sns
sns.set()

def make_regional_info(imp_data, divide, genotype_path, pop):
    genotype_data = pd.read_csv(genotype_path)
    pos_array = genotype_data.pos.str.split("_", expand=True).values
    pos = np.array([[int(each_pos[0]), int(each_pos[0])] if each_pos[1] == None else [int(each_pos[0]), int(each_pos[1])] for each_pos in pos_array])
    imp_data["start"] = pos[:, 0]
    imp_data["end"] = pos[:, 1]
    
    genotype_data["start"] = pos[:, 0]
    genotype_data["end"] = pos[:, 1]

    tmp_imp_data = imp_data.loc[imp_data[pop] == "2"]
    tmp_genotype_data = genotype_data.loc[imp_data[pop] == "2"]

    for tmp_idx, trait_imp in enumerate(["GN_imps", "PN_imps", "GS_imps", "sum_zscore"]):

        regional_info_all = []
        chr_pos = []
        chr_start = [0]
        for each_chr in genotype_data.chr.unique():
            chr_SNP_imps = tmp_imp_data[tmp_imp_data["chr"] == each_chr]
            chr_end_pos = genotype_data[genotype_data["chr"] == each_chr].iloc[-1, -1]
            for i in range(chr_end_pos//divide+1):
                regional_imps = chr_SNP_imps[(chr_SNP_imps.start >= i*divide) & (chr_SNP_imps.start < (i+1)*divide)]
                regional_geno = tmp_genotype_data[(tmp_genotype_data.chr == each_chr) & (tmp_genotype_data.start >= i*divide) & (tmp_genotype_data.start < (i+1)*divide)]
                regional_info = [each_chr, i*divide, (i+1)*divide, regional_imps.loc[:, "GN_imps"].sum(), regional_imps.loc[:, "PN_imps"].sum(), regional_imps.loc[:, "GS_imps"].sum(), regional_imps.loc[:, "sum_zscore"].sum()]
                if regional_geno.shape[0] < 1:
                    regional_info.extend(np.repeat(0, sum(tmp_genotype_data.columns.str.contains(pop))))
                else:
                    regional_info.extend((regional_geno.loc[:, regional_geno.columns.str.contains(pop)].sum() / (regional_geno.shape[0]*2)).values)
                regional_info_all.append(regional_info)

    regional_info_all = pd.DataFrame(regional_info_all)
    regional_columns = ["chr", "start", "end", "GN_imps", "PN_imps", "GS_imps", "sum_zscore"]
    regional_columns.extend(regional_geno.columns[regional_geno.columns.str.contains(pop)].values)
    regional_info_all.columns = regional_columns
    return regional_info_all

def get_top_regions(regional_info_all, target, kind, num):
    blocks = []
    current_block = [0]
    for i in range(1, regional_info_all.shape[0]):
        if (regional_info_all.loc[current_block[0], target] * regional_info_all.loc[i, target] <= 0) | \
           (regional_info_all.loc[current_block[-1], "chr"] != regional_info_all.loc[i, "chr"]) | \
           (i == regional_info_all.shape[0] - 1):
            if len(current_block) > 4:
                score = regional_info_all.loc[np.array(current_block)[[0, 1, 2, 3]], target].sum()
                block = np.array(current_block)[[0, 1, 2, 3]]
                for j in range(1, len(current_block) - 3):
                    tmp_score = regional_info_all.loc[np.array(current_block)[[j, j+1, j+2, j+3]], target].sum()
                    if ((kind == "top") & (tmp_score > score)) | ((kind == "worst") & (tmp_score < score)):
                        score = tmp_score
                        block = list(np.array(current_block)[[j, j+1, j+2, j+3]])
                blocks.append([block, score])
            else:
                blocks.append([current_block, regional_info_all.loc[current_block, target].sum()])
            current_block = [i]
        else:
            current_block.append(i)
            
    if kind == "top":
        top_inds = pd.DataFrame(blocks).sort_values(by=1, ascending=False).iloc[:num, 0].values
    else:
        top_inds = pd.DataFrame(blocks).sort_values(by=1, ascending=True).iloc[:num, 0].values
    top_regions = [[regional_info_all.loc[i[0], "chr"], regional_info_all.loc[i[0], "start"], regional_info_all.loc[i[-1], "end"]] for i in top_inds]
    return top_regions

def calc_outlier(importances, distance):
    # Q3+3IQR & Q1-3IQR
    q1=importances.quantile(0.25)
    q3=importances.quantile(0.75)
    iqr=q3-q1
    up=q3+(distance*iqr)
    bottom=q1-(distance*iqr)    
    return up, bottom

def get_target_region(regional_info):
    target_regions = []
    for row in regional_info.itertuples():
        target_regions.append([row[1], row[2], row[3]])
    return target_regions

def calc_genetic_effect(target_regions, imp_data, genotype, trait):
    if target_regions == "all":
        target_indices = imp_data.index.values
    else:
        target_indices = []
        for target_region in target_regions:
            target_index = imp_data.loc[(imp_data.chr == target_region[0]) & (imp_data.start >= target_region[1]) & (imp_data.start < target_region[2]), :].index.values
            target_indices.extend(target_index)
    target_indices = np.sort(np.unique(target_indices))
    target_imp_data = imp_data.loc[target_indices, :]
    target_genotype = genotype.loc[target_indices, :]
    return (target_genotype.T * target_imp_data[trait]).sum(axis=1)    

def plot_regional_info(regional_info_all, trait, color, figsize=(15,2)):
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=figsize)
    
    effects = regional_info_all.loc[:, trait].values
    plt.bar(np.arange(0, len(effects)), effects, color=color, edgecolor=color, label=trait)
    
    for i in [1,3,5,7,9,11]:
        start_pos = regional_info_all[regional_info_all.chr == "chr{}".format(str(i).zfill(2))].index.values[-1]
        last_pos = regional_info_all[regional_info_all.chr == "chr{}".format(str(i+1).zfill(2))].index.values[-1]
        plt.axvspan(start_pos+0.5, last_pos+0.5, color="gray", alpha=0.3)        

    xlabel_ori = []
    xlabel_new = []
    for i in range(1, 13):
        start_pos = regional_info_all[regional_info_all.chr == "chr{}".format(str(i).zfill(2))].index.values[0]
        last_pos = regional_info_all[regional_info_all.chr == "chr{}".format(str(i).zfill(2))].index.values[-1]
        xlabel_ori.extend(np.arange(start_pos, last_pos, 10))
        xlabel_new.extend(np.arange(0, (last_pos - start_pos), 10))
    plt.xticks(xlabel_ori, xlabel_new, rotation=90)
    plt.show()
    
    
def plot_regional_info_with_target(regional_info_all, trait, color, target_plus, target_minus):
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(15, 2))
    plt.axes().set_facecolor('white')
    effects = regional_info_all.loc[:, trait].values
    
    for i in [1,3,5,7,9,11]:
        start_pos = regional_info_all[regional_info_all.chr == "chr{}".format(str(i).zfill(2))].index.values[-1]
        last_pos = regional_info_all[regional_info_all.chr == "chr{}".format(str(i+1).zfill(2))].index.values[-1]
        plt.axvspan(start_pos+0.5, last_pos+0.5, color="gray", alpha=0.3)        

    for each_plus in target_plus:
        start = regional_info_all[(regional_info_all.chr == each_plus[0]) & (regional_info_all.start == each_plus[1])].index.values[-1]
        end = regional_info_all[(regional_info_all.chr == each_plus[0]) & (regional_info_all.end == each_plus[2])].index.values[-1]
        plt.axvspan(start-0.5, end+0.5, color="red", alpha=0.3, linewidth=0)
    for each_minus in target_minus:
        start = regional_info_all[(regional_info_all.chr == each_minus[0]) & (regional_info_all.start == each_minus[1])].index.values[-1]
        end = regional_info_all[(regional_info_all.chr == each_minus[0]) & (regional_info_all.end == each_minus[2])].index.values[-1]
        plt.axvspan(start-0.5, end+0.5, color="blue", alpha=0.3, linewidth=0)
        
    xlabel_ori = []
    xlabel_new = []
    for i in range(1, 13):
        start_pos = regional_info_all[regional_info_all.chr == "chr{}".format(str(i).zfill(2))].index.values[0]
        last_pos = regional_info_all[regional_info_all.chr == "chr{}".format(str(i).zfill(2))].index.values[-1]
        xlabel_ori.extend(np.arange(start_pos, last_pos, 10))
        xlabel_new.extend(np.arange(0, (last_pos - start_pos), 10))

    plt.bar(np.arange(0, len(effects)), effects, color=color, edgecolor=color, label=trait)

    plt.xticks(xlabel_ori, xlabel_new, rotation=90)
    plt.show()

def plot_regional_info_with_genotype(regional_info_all, trait, color, target_plus, target_minus, all_minus, target_RILs):
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(15, 2))
    
    height_ratios = [5, 5]
    height_ratios.extend(np.repeat(1, len(target_RILs)))
    fig, ax = plt.subplots(2+len(target_RILs), 1, figsize=(15, 7*sum(height_ratios)/20), sharex="all", gridspec_kw={'height_ratios': height_ratios})  
    effects = regional_info_all.loc[:, trait].values
    ax[0].bar(np.arange(0, len(effects)), effects, color=color, edgecolor=color, label=trait)
    sum_effects = regional_info_all.loc[:, "sum_zscore"].values
    ax[1].bar(np.arange(0, len(sum_effects)), sum_effects, color="gray", edgecolor="gray", label="sum_zscore")
    
    for i in [1,3,5,7,9,11]:
        start_pos = regional_info_all[regional_info_all.chr == "chr{}".format(str(i).zfill(2))].index.values[-1]
        last_pos = regional_info_all[regional_info_all.chr == "chr{}".format(str(i+1).zfill(2))].index.values[-1]
        ax[0].axvspan(start_pos+0.5, last_pos+0.5, color="gray", alpha=0.3)        
        ax[1].axvspan(start_pos+0.5, last_pos+0.5, color="gray", alpha=0.3)        

    for each_plus in target_plus:
        start = regional_info_all[(regional_info_all.chr == each_plus[0]) & (regional_info_all.start == each_plus[1])].index.values[-1]
        end = regional_info_all[(regional_info_all.chr == each_plus[0]) & (regional_info_all.end == each_plus[2])].index.values[-1]
        ax[0].axvspan(start-0.5, end+0.5, color="red", alpha=0.3, linewidth=0)
    for each_minus in target_minus:
        start = regional_info_all[(regional_info_all.chr == each_minus[0]) & (regional_info_all.start == each_minus[1])].index.values[-1]
        end = regional_info_all[(regional_info_all.chr == each_minus[0]) & (regional_info_all.end == each_minus[2])].index.values[-1]
        ax[0].axvspan(start-0.5, end+0.5, color="blue", alpha=0.3, linewidth=0)
    for each_minus in all_minus:
        start = regional_info_all[(regional_info_all.chr == each_minus[0]) & (regional_info_all.start == each_minus[1])].index.values[-1]
        end = regional_info_all[(regional_info_all.chr == each_minus[0]) & (regional_info_all.end == each_minus[2])].index.values[-1]
        ax[1].axvspan(start-0.5, end+0.5, color="blue", alpha=0.3, linewidth=0)
        
    xlabel_ori = []
    xlabel_new = []
    for i in range(1, 13):
        start_pos = regional_info_all[regional_info_all.chr == "chr{}".format(str(i).zfill(2))].index.values[0]
        last_pos = regional_info_all[regional_info_all.chr == "chr{}".format(str(i).zfill(2))].index.values[-1]
        xlabel_ori.extend(np.arange(start_pos, last_pos, 10))
        xlabel_new.extend(np.arange(0, (last_pos - start_pos), 10))
    
    for i, target_RIL in enumerate(target_RILs):
        genotype_color = []
        for j in regional_info_all[target_RIL]:
            if j == 0:
                genotype_color.append((230/255,159/255,0,1))
            else:
                if color == "#0072B2":
                    genotype_color.append((0,114/255,178/255,1*j))
                else:
                    genotype_color.append((0,158/255,115/255,1*j))

        ax[i+2].bar(np.arange(0, len(effects)), np.repeat(1, len(effects)), color=genotype_color, edgecolor=genotype_color)
        ax[i+2].set_xticks([])
        ax[i+2].set_yticks([])
        ax[i+2].set_ylabel(target_RIL, rotation=0)
    plt.xticks(xlabel_ori, xlabel_new, rotation=90)
    plt.show()
    
def target_effect_summary(imp_data, genotype_path, Hitome_per_path, pop, trait, target_plus, target_minus, total_minus):
    genotype_data = pd.read_csv(genotype_path)
    pos_array = genotype_data.pos.str.split("_", expand=True).values
    pos = np.array([[int(each_pos[0]), int(each_pos[0])] if each_pos[1] == None else [int(each_pos[0]), int(each_pos[1])] for each_pos in pos_array])
    genotype_data["start"] = pos[:, 0]
    genotype_data["end"] = pos[:, 1]

    plus, plus_num, plus_ratios, minus, minus_ratios = [], [], [], [], []
    plus_all, minus_all, minus_all_ratios = [], [], []
    for check_RIL in genotype_data.columns[genotype_data.columns.str.contains(pop)]:
        pos_indices = []
        plus_regions = 0
        plus_region_ratio = []
        for i in target_plus:
            tmp_genotype_data = genotype_data[(genotype_data.chr == i[0]) & (genotype_data.start >= i[1]) & (genotype_data.end <= i[2])]
            tmp_imp_data = imp_data[(imp_data.chr == i[0]) & (imp_data.start >= i[1]) & (imp_data.end <= i[2])]
            tmp_genotype_data = tmp_genotype_data[(tmp_genotype_data[check_RIL] == 1) | (tmp_genotype_data[check_RIL] == 2)]
            pos_indices.extend(tmp_genotype_data.index.values)
            if len(tmp_genotype_data.index.values) > 0:
                plus_regions+=1
                plus_region_ratio.append(np.round(tmp_genotype_data.shape[0] / (tmp_imp_data[check_RIL[:3]] == "2").sum(), 3))
            else:
                plus_region_ratio.append(0)
        plus_num.append(plus_regions)
        plus_ratios.append(f"{plus_region_ratio[0]}, {plus_region_ratio[1]}, {plus_region_ratio[2]}")
        plus.append(imp_data.loc[pos_indices, :].sum()["{}_imps".format(trait)])
        plus_all.append(imp_data.loc[pos_indices, :].sum()["sum_zscore"])
        pos_indices = []
        minus_region_ratio = []
        for i in target_minus:
            tmp_genotype_data = genotype_data[(genotype_data.chr == i[0]) & (genotype_data.start >= i[1]) & (genotype_data.end <= i[2])]
            tmp_genotype_data = tmp_genotype_data[(tmp_genotype_data[check_RIL] == 1) | (tmp_genotype_data[check_RIL] == 2)]
            tmp_imp_data = imp_data[(imp_data.chr == i[0]) & (imp_data.start >= i[1]) & (imp_data.end <= i[2])]
            pos_indices.extend(tmp_genotype_data.index.values)
            if len(tmp_genotype_data.index.values) > 0:
                minus_region_ratio.append(np.round(tmp_genotype_data.shape[0] / (tmp_imp_data[check_RIL[:3]] == "2").sum(), 3))
            else:
                minus_region_ratio.append(0)
        minus.append(imp_data.loc[pos_indices, :].sum()["{}_imps".format(trait)])
        minus_ratios.append(f"{minus_region_ratio[0]}, {minus_region_ratio[1]}, {minus_region_ratio[2]}")
        minus_all_ratio = []
        for i in total_minus:
            tmp_genotype_data = genotype_data[(genotype_data.chr == i[0]) & (genotype_data.start >= i[1]) & (genotype_data.end <= i[2])]
            tmp_genotype_data = tmp_genotype_data[(tmp_genotype_data[check_RIL] == 1) | (tmp_genotype_data[check_RIL] == 2)]
            tmp_imp_data = imp_data[(imp_data.chr == i[0]) & (imp_data.start >= i[1]) & (imp_data.end <= i[2])]
            pos_indices.extend(tmp_genotype_data.index.values)
            if len(tmp_genotype_data.index.values) > 0:
                minus_all_ratio.append(np.round(tmp_genotype_data.shape[0] / (tmp_imp_data[check_RIL[:3]] == "2").sum(), 3))
            else:
                minus_all_ratio.append(0)
        minus_all.append(imp_data.loc[pos_indices, :].sum()["sum_zscore"])
        minus_all_ratios.append(f"{minus_all_ratio[0]}, {minus_all_ratio[1]}, {minus_all_ratio[2]}")
    summary = pd.DataFrame({"RIL": genotype_data.columns[genotype_data.columns.str.contains(pop)],
                            "{}_plus".format(trait):plus,
                            "{}_plus_regions".format(trait):plus_num,
                            "{}_plus_region_ratios".format(trait):plus_ratios,
                            "{}_minus".format(trait):minus,
                            "{}_minus_region_ratios".format(trait):minus_ratios,
                            "{}_merge".format(trait):np.array(plus)+np.array(minus),
                            "zscore_minus":minus_all,
                            "zscore_minus_ratios":minus_all_ratios,
                            })
    Hitome_per = pd.read_csv(Hitome_per_path, index_col=0)
    summary["Hitome_per"] = Hitome_per.loc[summary.RIL, :].values
    return summary.sort_values(by="{}_merge".format(trait), ascending=False)
    
    
def visualize_line_genotype(genotype, chr_ends, hitome_color, other_color, dpi=80):
    
    plt.rcParams['figure.dpi'] = dpi
    fig = plt.figure(figsize=(100,10))
    ax = plt.axes()
    for i, each_chr in enumerate(genotype.chr.unique()):
        if i == 0:
            r = patches.Rectangle(xy=(0, 0), height=8000000, width=chr_ends[i], ec='gray', fc=hitome_color, linewidth=3)
        else:
            r = patches.Rectangle(xy=(sum(chr_ends[:i]), 0), height=8000000, width=chr_ends[i], ec='gray', fc=hitome_color, linewidth=3)
        ax.add_patch(r)
        chr_genotype = genotype[genotype["chr"] == each_chr]
        chr_genotype = chr_genotype[chr_genotype.iloc[:, 3] >= 1]
        for j in range(chr_genotype.shape[0]):
            start = chr_genotype.iloc[j, :].start
            end = chr_genotype.iloc[j, :].end
            if i == 0:
                r = patches.Rectangle(xy=(start, 0), height=8000000, width=(end-start), ec='gray', fc=other_color, linewidth=0)
            else:
                r = patches.Rectangle(xy=(sum(chr_ends[:i])+start, 0), height=8000000, width=(end-start), ec='gray', fc=other_color, linewidth=0)
            ax.add_patch(r)

    r = patches.Rectangle(xy=(0, 0), height=8000000, width=sum(chr_ends), ec='black', fill=False, linewidth=20)
    ax.add_patch(r)
    
    plt.axis('scaled')
    plt.axis('off')
    plt.tight_layout()
    ax.set_aspect('equal')

    plt.show()
    

def visualize_RIL_genotype(genotype, chr_ends, hitome_color, other_color, target_plus, target_minus, dpi=80):
    
    plt.rcParams['figure.dpi'] = dpi
    fig = plt.figure(figsize=(5,12))
    ax = plt.axes()
    for i, each_chr in enumerate(genotype.chr.unique()):
        r = patches.Rectangle(xy=(i*16000000, 0), width=6000000, height=-chr_ends[i]*3, ec='gray', fc=hitome_color, linewidth=3)
        ax.add_patch(r)
        chr_genotype = genotype[genotype["chr"] == each_chr]
        chr_genotype = chr_genotype[chr_genotype.iloc[:, 3] >= 1]
        for j in range(chr_genotype.shape[0]):
            start = chr_genotype.iloc[j, :].start
            end = chr_genotype.iloc[j, :].end
            r = patches.Rectangle(xy=(i*16000000, -start*3), width=6000000, height=-(end-start)*3, ec='gray', fc=other_color, linewidth=0)
            ax.add_patch(r)
        r = patches.Rectangle(xy=(i*16000000, 0), width=6000000, height=-chr_ends[i]*3, ec='gray', fill=False, linewidth=3)
        ax.add_patch(r)

    for plus_region in target_plus:
        i = int(plus_region[0][3:]) - 1
        start = plus_region[1]
        end = plus_region[2]
        r = patches.Rectangle(xy=(i*16000000+8000000, -start*3), width=3000000, height=-(end-start)*3, ec='gray', fc="red", alpha=0.3, linewidth=0)
        ax.add_patch(r)

    for minus_region in target_minus:
        i = int(minus_region[0][3:]) - 1
        start = minus_region[1]
        end = minus_region[2]
        r = patches.Rectangle(xy=(i*16000000+8000000, -start*3), width=3000000, height=-(end-start)*3, ec='gray', fc="blue", alpha=0.3, linewidth=0)
        ax.add_patch(r)

    plt.axis('scaled')
    plt.axis('off')
    plt.tight_layout()
    ax.set_aspect('equal')

    plt.show()
    
    
def visualize_F3_genotype(genotype, chr_ends, hitome_color, GN_target, PN_target, dpi=80, save=False, name=""):
    
    plt.rcParams['figure.dpi'] = dpi
    fig = plt.figure(figsize=(5,12))
    ax = plt.axes()
    for i, each_chr in enumerate(genotype.chr.unique()):
        r = patches.Rectangle(xy=(i*16000000, 0), width=6000000, height=-chr_ends[i]*3, ec='gray', fc=hitome_color, linewidth=3)
        ax.add_patch(r)
        chr_genotype = genotype[genotype["chr"] == each_chr]
        chr_genotype = chr_genotype[chr_genotype.iloc[:, 3] >= 1]
        for j in range(chr_genotype.shape[0]):
            start = chr_genotype.iloc[j, :].start
            end = chr_genotype.iloc[j, :].end
            color = chr_genotype.iloc[j, :].color
            if chr_genotype.iloc[j, 3] == 1:
                r = patches.Rectangle(xy=(i*16000000, -start*3), width=3600000, height=-(end-start)*3, ec='gray', fc=color, linewidth=0)
            else:
                r = patches.Rectangle(xy=(i*16000000, -start*3), width=6000000, height=-(end-start)*3, ec='gray', fc=color, linewidth=0)
            ax.add_patch(r)
        r = patches.Rectangle(xy=(i*16000000, 0), width=6000000, height=-chr_ends[i]*3, ec='gray', fill=False, linewidth=3)
        ax.add_patch(r)

    for plus_region in GN_target:
        i = int(plus_region[0][3:]) - 1
        start = plus_region[1]
        end = plus_region[2]
        r = patches.Rectangle(xy=(i*16000000+8000000, -start*3), width=3000000, height=-(end-start)*3, ec='gray', fc="purple", alpha=0.3, linewidth=0)
        ax.add_patch(r)

    for minus_region in PN_target:
        i = int(minus_region[0][3:]) - 1
        start = minus_region[1]
        end = minus_region[2]
        r = patches.Rectangle(xy=(i*16000000+8000000, -start*3), width=3000000, height=-(end-start)*3, ec='gray', fc="darkgreen", alpha=0.3, linewidth=0)
        ax.add_patch(r)

    plt.axis('scaled')
    plt.axis('off')
    plt.tight_layout()
    ax.set_aspect('equal')
    if save:
        plt.savefig(f"{name}.png", format="png", dpi=300)
    plt.show()