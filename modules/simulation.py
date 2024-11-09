import numpy as np
import pandas as pd
import itertools
import random
import copy
from scipy.stats import gaussian_kde


def get_input_for_recombination(recombination_info):
    input_data = []
    for i in range(1,13):
        tmp_chr = "chr{}".format(str(i).zfill(2))
        chr_recombination_info = recombination_info[recombination_info.chromosome == tmp_chr]
        recombination_points = list(itertools.chain.from_iterable(chr_recombination_info.recombination_point.values))
        recombination_points.sort()
        average_count = chr_recombination_info["count"].mean()
        input_data.append([recombination_points, average_count])
    input_data = pd.DataFrame(input_data)
    input_data.columns = ["pos", "ave_num"]
    input_data.index = ["chr{}".format(str(i).zfill(2)) for i in range(1, 13)]
    return input_data

def get_recombination_points(recombination_points, average_recom_num):
    np.random.seed(seed=random.randint(0, 10000))
    # simulate recombination number
    recom_num = np.random.poisson(average_recom_num / 2, 1)[0]
    # simulate recombination points
    kernel = gaussian_kde(recombination_points)
    recom_points = kernel.resample(recom_num, seed=random.randint(0, 10000))[0]
    return np.sort(recom_points)

def convert_genotype_to_pair(genotype_values, line):
    pair = []
    for i in genotype_values:
        if i == 0:
            pair.append([0, 0, "Hitomebore", "Hitomebore"])
        elif i == 1:
            pair.append([0, 1, "Hitomebore", line[:3]])
        else:
            pair.append([1, 1, line[:3], line[:3]])
    pair = pd.DataFrame(pair, columns=["{}_1".format(line), "{}_2".format(line), "{}_1_color".format(line), "{}_2_color".format(line)])
    return pair

# cross simulation
# columns of genotype must be "chr" & "start" & "end" & "parent1_1" & "parent1_2" & "parent1_1_color" & "parent1_2_color"
def cross(parent1_genotype, parent2_genotype, parent1, parent2, progeny, input_recom_data):
    gametes = parent1_genotype.loc[:, ["chr", "start", "end"]]
    # generate possible gametes for each parent
    for parent in [[parent1, parent1_genotype], [parent2, parent2_genotype]]:
        parent_name = parent[0]
        parent_genotype = parent[1]
        each_parent_gamete = pd.DataFrame()
        # generate possible gamate for each chromosome
        for each_chr in parent_genotype.chr.unique():
            chr_genotype = parent_genotype.loc[parent_genotype.chr == each_chr, :]
            parent_gamete1 = chr_genotype.loc[:, [f"{parent_name}_1", f"{parent_name}_1_color"]]
            parent_gamete1.columns = ["genotype", "color"]
            parent_gamete2 = chr_genotype.loc[:, [f"{parent_name}_2", f"{parent_name}_2_color"]]
            parent_gamete2.columns = ["genotype", "color"]
            # simulate recombination events
            recombination_points = get_recombination_points(input_recom_data.loc[each_chr, "pos"], input_recom_data.loc[each_chr, "ave_num"])
            recombination_num = len(recombination_points)
            # no recombination -> which gamete
            if recombination_num == 0:
                if random.randint(0, 1) == 0:
                    chr_gamete = parent_gamete1
                else:
                    chr_gamete = parent_gamete2
            else:
                # recombination -> simulate exchange regions
                change_region = np.repeat(False, chr_genotype.shape[0])
                for j in range(recombination_num // 2 + 1):
                    if j == 0:
                        change_region[chr_genotype["start"] < recombination_points[j]] = True
                    elif (j == recombination_num // 2) & (recombination_num % 2 == 0):
                        change_region[chr_genotype["start"] >= recombination_points[j*2-1]] = True
                    else:
                        change_region[(chr_genotype["start"] >= recombination_points[j*2-1]) & (chr_genotype["start"] < recombination_points[j*2])] = True
                # possible gamete genotypes                
                gamete1 = copy.copy(parent_gamete1)
                gamete1.loc[change_region, :] = parent_gamete2.loc[change_region, :]
                gamete2 = copy.copy(parent_gamete2)
                gamete2.loc[change_region, :] = parent_gamete1.loc[change_region, :]
                possible_gametes = [gamete1, gamete2]
                chr_gamete = possible_gametes[random.randint(0, 1)]
            each_parent_gamete = pd.concat([each_parent_gamete, chr_gamete])
        gametes = pd.concat([gametes, each_parent_gamete], axis=1)
    gametes.columns = ["chr", "start", "end", f"{progeny}_1", f"{progeny}_1_color", f"{progeny}_2", f"{progeny}_2_color"]
    return gametes

def check_target_regions(check_genotype, name, target_regions):
    target = False
    target_Homo = False
    target_exist = (check_genotype.loc[target_regions, [f"{name}_1", f"{name}_2"]] == 1).sum(axis=1)
    if (target_exist == 0).sum() == 0:
        target = True
    if (target_exist < 2).sum() == 0:
        target_Homo = True
    target_Homo_ratio = (target_exist == 2).sum() / len(target_exist)
    return target, target_Homo, target_Homo_ratio

def merge_overlapped_regions(tmp_genotype, current_region, regions):
    if tmp_genotype.shape[0] == 0:
        if len(current_region) > 0:
            regions.append(current_region)
        return regions
    else:
        if len(current_region) == 0:
            current_region = [tmp_genotype.iloc[0, :].start, tmp_genotype.iloc[0, :].end]
            return merge_overlapped_regions(tmp_genotype, current_region, regions)
        else:
            not_overlapped = (tmp_genotype.end < current_region[0]) | (tmp_genotype.start > current_region[1])
            overlapped_genotype = tmp_genotype[~not_overlapped]
            tmp_genotype = tmp_genotype[not_overlapped]
            if overlapped_genotype.shape[0] > 0:
                current_region = [min(current_region[0], overlapped_genotype.start.min()), max(current_region[1], overlapped_genotype.end.max())]
                return merge_overlapped_regions(tmp_genotype, current_region, regions)
            else:
                regions.append(current_region)
                return merge_overlapped_regions(tmp_genotype, [], regions)

def check_approximate_Hitomebore_ratio(check_genotype, name, total_length):
    check_genotype = check_genotype.sort_values(by="start")
    # check length of regions with variants
    Not_Hitomebore_region_length = 0
    for one_genotype in [f"{name}_1", f"{name}_2"]:
        one_genotype = check_genotype[check_genotype[one_genotype] == 1]
        for each_chr in one_genotype.chr.unique():
            chr_genotype = one_genotype[one_genotype.chr == each_chr]
            merged_regions = merge_overlapped_regions(chr_genotype, [], [])
            for i in merged_regions:
                Not_Hitomebore_region_length+=(i[1]-i[0])
    return (total_length*2 - Not_Hitomebore_region_length) / (total_length*2)