import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import seaborn as sns
sns.set()


def make_regional_info(imp_data, divide, genotype_path, pop):
    
    # add position data to genetic effet of each HB
    genotype_data = pd.read_csv(genotype_path)
    pos_array = genotype_data.pos.str.split("_", expand=True).values
    pos = np.array([[int(each_pos[0]), int(each_pos[0])] if each_pos[1] == None else [int(each_pos[0]), int(each_pos[1])] for each_pos in pos_array])
    imp_data["start"] = pos[:, 0]
    imp_data["end"] = pos[:, 1]
    
    genotype_data["start"] = pos[:, 0]
    genotype_data["end"] = pos[:, 1]
    
    # extract specific donor HBs
    tmp_imp_data = imp_data.loc[imp_data[pop] == "2"]
    tmp_genotype_data = genotype_data.loc[imp_data[pop] == "2"]

    # calculate genetic effect for each 1Mbp
    for tmp_idx, trait_imp in enumerate(["GN_imps", "PN_imps", "GS_imps", "sum_zscore"]):

        regional_info_all = []
        chr_pos = []
        chr_start = [0]
        
        # calculate for each chromosome
        for each_chr in genotype_data.chr.unique():
            chr_SNP_imps = tmp_imp_data[tmp_imp_data["chr"] == each_chr]
            chr_end_pos = genotype_data[genotype_data["chr"] == each_chr].iloc[-1, -1]
            
            # calculate for each region
            for i in range(chr_end_pos//divide+1):
                regional_imps = chr_SNP_imps[(chr_SNP_imps.start >= i*divide) & (chr_SNP_imps.start < (i+1)*divide)]
                regional_geno = tmp_genotype_data[(tmp_genotype_data.chr == each_chr) & (tmp_genotype_data.start >= i*divide) & (tmp_genotype_data.start < (i+1)*divide)]
                regional_info = [each_chr, i*divide, (i+1)*divide, regional_imps.loc[:, "GN_imps"].sum(), regional_imps.loc[:, "PN_imps"].sum(), regional_imps.loc[:, "GS_imps"].sum(), regional_imps.loc[:, "sum_zscore"].sum()]
                if regional_geno.shape[0] < 1:
                    regional_info.extend(np.repeat(0, sum(tmp_genotype_data.columns.str.contains(pop))))
                else:
                    regional_info.extend((regional_geno.loc[:, regional_geno.columns.str.contains(pop)].sum() / (regional_geno.shape[0]*2)).values)
                regional_info_all.append(regional_info)
    
    # make dataframe of genetic effect of regions
    regional_info_all = pd.DataFrame(regional_info_all)
    regional_columns = ["chr", "start", "end", "GN_imps", "PN_imps", "GS_imps", "sum_zscore"]
    regional_columns.extend(regional_geno.columns[regional_geno.columns.str.contains(pop)].values)
    regional_info_all.columns = regional_columns
    return regional_info_all


# extract top X or worst X genomic regions
def get_top_regions(regional_info_all, target, kind, num):
    
    blocks = []
    current_block = [0]
    
    # check genetic effect & compare previous blocks
    for i in range(1, regional_info_all.shape[0]):
        if (regional_info_all.loc[current_block[0], target] * regional_info_all.loc[i, target] <= 0) | \
           (regional_info_all.loc[current_block[-1], "chr"] != regional_info_all.loc[i, "chr"]) | \
           (i == regional_info_all.shape[0] - 1):
            
            # block >= 4 Mbp or not
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
    
    # top or worst
    if kind == "top":
        top_inds = pd.DataFrame(blocks).sort_values(by=1, ascending=False).iloc[:num, 0].values
    else:
        top_inds = pd.DataFrame(blocks).sort_values(by=1, ascending=True).iloc[:num, 0].values
    top_regions = [[regional_info_all.loc[i[0], "chr"], regional_info_all.loc[i[0], "start"], regional_info_all.loc[i[-1], "end"]] for i in top_inds]
    return top_regions


# visualize genetic effect of each genomic regions   
def plot_regional_info_with_target(regional_info_all, trait, color, target_plus, target_minus):
    
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(15, 2))
    plt.axes().set_facecolor('white')
    
    # visualize each chromosome color
    effects = regional_info_all.loc[:, trait].values
    for i in [1,3,5,7,9,11]:
        start_pos = regional_info_all[regional_info_all.chr == "chr{}".format(str(i).zfill(2))].index.values[-1]
        last_pos = regional_info_all[regional_info_all.chr == "chr{}".format(str(i+1).zfill(2))].index.values[-1]
        plt.axvspan(start_pos+0.5, last_pos+0.5, color="gray", alpha=0.3)        
    
    # visualize top3 regions
    for each_plus in target_plus:
        start = regional_info_all[(regional_info_all.chr == each_plus[0]) & (regional_info_all.start == each_plus[1])].index.values[-1]
        end = regional_info_all[(regional_info_all.chr == each_plus[0]) & (regional_info_all.end == each_plus[2])].index.values[-1]
        plt.axvspan(start-0.5, end+0.5, color="red", alpha=0.3, linewidth=0)
    
    # visualize worst3 regions
    for each_minus in target_minus:
        start = regional_info_all[(regional_info_all.chr == each_minus[0]) & (regional_info_all.start == each_minus[1])].index.values[-1]
        end = regional_info_all[(regional_info_all.chr == each_minus[0]) & (regional_info_all.end == each_minus[2])].index.values[-1]
        plt.axvspan(start-0.5, end+0.5, color="blue", alpha=0.3, linewidth=0)
        
    # add label
    xlabel_ori = []
    xlabel_new = []
    for i in range(1, 13):
        start_pos = regional_info_all[regional_info_all.chr == "chr{}".format(str(i).zfill(2))].index.values[0]
        last_pos = regional_info_all[regional_info_all.chr == "chr{}".format(str(i).zfill(2))].index.values[-1]
        xlabel_ori.extend(np.arange(start_pos, last_pos, 10))
        xlabel_new.extend(np.arange(0, (last_pos - start_pos), 10))

    # visualize each genomic regions (ex. 1 Mbp)
    plt.bar(np.arange(0, len(effects)), effects, color=color, edgecolor=color, label=trait)

    plt.xticks(xlabel_ori, xlabel_new, rotation=90)
    plt.show()


# summarize genetic effect of top/worst genomic regions
def target_effect_summary(imp_data, genotype_path, Hitome_per_path, pop, trait, target_plus, target_minus, total_minus):
    
    # get genotype & position data
    genotype_data = pd.read_csv("../data/PART7_Select_useful_RILs/Haplotype_high_imp_HBs_for_3traits.csv")
    pos_array = genotype_data.pos.str.split("_", expand=True).values
    pos = np.array([[int(each_pos[0]), int(each_pos[0])] if each_pos[1] == None else [int(each_pos[0]), int(each_pos[1])] for each_pos in pos_array])
    genotype_data["start"] = pos[:, 0]
    genotype_data["end"] = pos[:, 1]

    # calculate sum of genetic effects in top/worst genomic regions for each RIL
    plus, plus_num, plus_ratios, minus, minus_ratios = [], [], [], [], []
    plus_all, minus_all, minus_all_ratios = [], [], []
    for check_RIL in genotype_data.columns[genotype_data.columns.str.contains(pop)]:
        
        # check genetic effects of beneficial effect of target trait
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
                plus_region_ratio.append(np.round(tmp_genotype_data[check_RIL].sum() / ((tmp_imp_data[check_RIL[:3]] == "2").sum() * 2), 3))
            else:
                plus_region_ratio.append(0)
        plus_num.append(plus_regions)
        if len(target_plus) == 3:
            plus_ratios.append(f"{plus_region_ratio[0]}, {plus_region_ratio[1]}, {plus_region_ratio[2]}")
        else:
            plus_ratios.append(f"{plus_region_ratio[0]}, {plus_region_ratio[1]}")
        plus.append((imp_data.loc[pos_indices, "{}_imps".format(trait)] * genotype_data.loc[pos_indices, check_RIL]).sum())
        plus_all.append((imp_data.loc[pos_indices, "sum_zscore"] * genotype_data.loc[pos_indices, check_RIL]).sum())
        
        # check genetic effects of detrimental effect of target trait & yield
        pos_indices = []
        minus_region_ratio = []
        for i in target_minus:
            tmp_genotype_data = genotype_data[(genotype_data.chr == i[0]) & (genotype_data.start >= i[1]) & (genotype_data.end <= i[2])]
            tmp_genotype_data = tmp_genotype_data[(tmp_genotype_data[check_RIL] == 1) | (tmp_genotype_data[check_RIL] == 2)]
            tmp_imp_data = imp_data[(imp_data.chr == i[0]) & (imp_data.start >= i[1]) & (imp_data.end <= i[2])]
            pos_indices.extend(tmp_genotype_data.index.values)
            if len(tmp_genotype_data.index.values) > 0:
                minus_region_ratio.append(np.round(tmp_genotype_data[check_RIL].sum() / ((tmp_imp_data[check_RIL[:3]] == "2").sum() * 2), 3))
            else:
                minus_region_ratio.append(0)
        minus.append((imp_data.loc[pos_indices, "{}_imps".format(trait)] * genotype_data.loc[pos_indices, check_RIL]).sum())
        minus_ratios.append(f"{minus_region_ratio[0]}, {minus_region_ratio[1]}, {minus_region_ratio[2]}")
        minus_all_ratio = []
        for i in total_minus:
            tmp_genotype_data = genotype_data[(genotype_data.chr == i[0]) & (genotype_data.start >= i[1]) & (genotype_data.end <= i[2])]
            tmp_genotype_data = tmp_genotype_data[(tmp_genotype_data[check_RIL] == 1) | (tmp_genotype_data[check_RIL] == 2)]
            tmp_imp_data = imp_data[(imp_data.chr == i[0]) & (imp_data.start >= i[1]) & (imp_data.end <= i[2])]
            pos_indices.extend(tmp_genotype_data.index.values)
            if len(tmp_genotype_data.index.values) > 0:
                minus_all_ratio.append(np.round(tmp_genotype_data[check_RIL].sum() / ((tmp_imp_data[check_RIL[:3]] == "2").sum() * 2), 3))
            else:
                minus_all_ratio.append(0)
        minus_all.append((imp_data.loc[pos_indices, "sum_zscore"] * genotype_data.loc[pos_indices, check_RIL]).sum())
        minus_all_ratios.append(f"{minus_all_ratio[0]}, {minus_all_ratio[1]}, {minus_all_ratio[2]}")
    
    # make dataframe of effects
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
    
    # add ratio of Hitomebore derived regions
    Hitome_per = pd.read_csv(Hitome_per_path, index_col=0)
    summary["Hitome_per"] = Hitome_per.loc[summary.RIL, :].values
    return summary.sort_values(by="{}_merge".format(trait), ascending=False)
    

# visualize graphical genotype of RILs
def visualize_RIL_genotype(genotype, chr_ends, hitome_color, other_color, target_plus, target_minus, dpi=80):
    
    plt.rcParams['figure.dpi'] = dpi
    fig = plt.figure(figsize=(5,12))
    ax = plt.axes()
    
    # visualize chromosomes
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
    
    # put color to benefical regions
    for plus_region in target_plus:
        i = int(plus_region[0][3:]) - 1
        start = plus_region[1]
        end = plus_region[2]
        r = patches.Rectangle(xy=(i*16000000+8000000, -start*3), width=3000000, height=-(end-start)*3, ec='gray', fc="red", alpha=0.3, linewidth=0)
        ax.add_patch(r)

    # put color to detrimental regions
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