import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform
import allel
from tqdm import tqdm


def merge_same_genotype_position(SNP_set, position, SNPtype_RIL_ids):
    # extract index in the same genotype SNPs in specific population
    SNPtype_genotype = SNP_set.loc[:, SNPtype_RIL_ids]
    indices = []
    for i in range(SNPtype_genotype.shape[0]):
        if i == SNPtype_genotype.shape[0] - 1:
            indices.append([i])
        elif np.array_equal(SNPtype_genotype.iloc[i, :].values, SNPtype_genotype.iloc[i+1, :].values, equal_nan=True):
            indices.append([i, i+1])
        else:
            indices.append([i])

    after = []
    for i, j in enumerate(indices):
        if i == 0:
            after.append(j)
        else:
            if j[0] in after[-1]:
                if len(j) > 1:
                    after[-1].append(j[1])
                else:
                    pass
            else:
                after.append(j)

    range_position = []
    new_indices = []
    for i in after:
        if len(i) > 1:
            first_pos = position.iloc[i[0], 1]
            last_pos = position.iloc[i[-1], 1]
            range_pos = "{}_{}".format(first_pos, last_pos)
            range_position.append(range_pos)
            new_indices.append(i[0])
        else:
            range_position.append(position.iloc[i[0], 1])
            new_indices.append(i[0])

    new_genotype = SNP_set.iloc[new_indices, :]
    new_position = position.iloc[new_indices, :].copy()
    new_position["pos"] = range_position
    new_genotype = pd.concat([new_position, new_genotype], axis=1)
    return new_genotype

def parse_start_position(pos):
    pos = str(pos)
    if pos.find("_") > 0:
        return pos[:pos.find("_")]
    else:
        return pos

def parse_end_position(pos):
    pos = str(pos)
    if pos.find("_") > 0:
        return pos[pos.rfind("_")+1:]
    else:
        return pos

def extract_High_LD_block(genotype_data, haplotype_blocks, SNPtype_RIL_ids, threshold=0.9):
    # merge SNP set that showed linkage disequilibrium > 0.9
    SNP_num = genotype_data.shape[0]
    SNP_type = genotype_data.iloc[0, 2]
    if SNP_num <= 1:
        haplotype_blocks.append(genotype_data.values)
        return haplotype_blocks
    else:
        for i in range(SNP_num):
            gn = np.array(genotype_data.iloc[[0, i], :].loc[:, SNPtype_RIL_ids].dropna(how='any', axis=1))
            r = allel.rogers_huff_r(gn)
            if squareform(r ** 2)[0, 1] < threshold:
                chrom = genotype_data.iloc[0, 0]
                start = parse_start_position(genotype_data.iloc[0, 1])
                end = parse_end_position(genotype_data.iloc[i-1, 1])
                mode_genotype = genotype_data.iloc[:i, 3:].mode().values[0]
                haplotype_block = [chrom, "{}_{}".format(start, end), SNP_type]
                haplotype_block.extend(list(mode_genotype))
                haplotype_blocks.append(haplotype_block)
                if i == SNP_num - 1:
                    haplotype_blocks.append(genotype_data.iloc[i, :].values)
                    return haplotype_blocks
                else:
                    return extract_High_LD_block(genotype_data.iloc[i:, :], haplotype_blocks, SNPtype_RIL_ids)
            else:
                if i == SNP_num - 1:
                    chrom = genotype_data.iloc[0, 0]
                    start = parse_start_position(genotype_data.iloc[0, 1])
                    end = parse_end_position(genotype_data.iloc[i, 1])
                    mode_genotype = genotype_data.iloc[:, 3:].mode().values[0]
                    haplotype_block = [chrom, "{}_{}".format(start, end), SNP_type]
                    haplotype_block.extend(list(mode_genotype))
                    haplotype_blocks.append(haplotype_block)
                    return haplotype_blocks
                else:
                    pass

def make_Haplotype_block_df(genotype, SNPtype_RIL_ids, threshold=0.9):
    Haplotype_block = extract_High_LD_block(genotype, [], SNPtype_RIL_ids, threshold)
    Haplotype_block = pd.DataFrame(Haplotype_block)
    Haplotype_block.columns = genotype.columns
    return Haplotype_block


def make_Haplotype_block(family, family_list, genotype_path_list, out_path):

    Haplotype = pd.DataFrame()
    all_RIL_ids = family_list.loc[family_list.family.isin(family), "id"]
    
    for genotype_path in genotype_path_list:
        # read original data
        genotype = pd.read_csv(genotype_path, sep="\t")
        # extract position
        position = genotype.loc[:, ["chr", "pos"]]
        # select parent lines & RIL lines
        parental = genotype.loc[:, family]
        genotype = genotype.loc[:, all_RIL_ids]
        # extract Haplotype block
        chr_Haplotype = pd.DataFrame()
        # classify SNP type (which parental lines have this SNP?)
        for SNP_type in tqdm(np.unique(parental.values, axis=0)):
            if sum(SNP_type) == 0:
                pass
            else:
                # extract SNP set of each SNP type
                SNP_set_indices = [np.array_equal(each_line, SNP_type) for each_line in parental.values]
                SNP_set_position = position.loc[SNP_set_indices, :]
                SNP_set = genotype.loc[SNP_set_indices, :]
                # identify RIL ids dereived from parents of each SNP type
                SNPtype_parents = parental.columns[SNP_type == 2].values
                SNPtype_RIL_ids = family_list.loc[family_list.family.isin(SNPtype_parents), "id"].values
                # merge SNP set that showed same genotypes in RIL population
                SNP_set = merge_same_genotype_position(SNP_set, SNP_set_position, SNPtype_RIL_ids)
                # add SNP type column
                SNP_type = pd.Series(np.repeat("".join([str(n) for n in SNP_type]), SNP_set.shape[0]), name="SNP_type")
                SNP_set = SNP_set.reset_index(drop=True)
                SNP_set = pd.concat([SNP_set.iloc[:, :2], SNP_type, SNP_set.iloc[:, 2:]], axis=1)
                # merge SNP set that showed LD > 0.9
                if SNP_set.shape[0] > 1:
                    Haplotype_blocks = make_Haplotype_block_df(SNP_set, SNPtype_RIL_ids)
                else:
                    Haplotype_blocks = SNP_set
                chr_Haplotype = pd.concat([chr_Haplotype, Haplotype_blocks])

        sorted_index = [int(i[:i.find("_")]) if i.find("_") > 0 else int(i) for i in chr_Haplotype.pos.astype("str").values]
        chr_Haplotype = chr_Haplotype.iloc[np.argsort(sorted_index), :]

        Haplotype = pd.concat([Haplotype, chr_Haplotype])

    print(Haplotype.shape)
    
    Haplotype.to_csv(out_path, index=None)
    
    
def make_Haplotype_block_other_pop(other_genotype, NAM_HB_genotype, family_list):
    family_list = pd.read_csv(family_list)
    family = family_list.family.unique()
    other_genotype = pd.read_csv(other_genotype, sep="\t")
    
    # Get SNP type information
    SNP_types = []
    for chr_num in tqdm(range(1, 13)):
        chromosome = "chr{}".format(str(chr_num).zfill(2))
        other_genotype_chr = other_genotype[other_genotype.chr == chromosome]
        # read original data
        parental = pd.read_csv("../data/genotype_data/NAM_all_imputed_all_filtered_commpos_ALL_genotype_{}.txt.gz".format(chromosome), usecols=range(25), sep="\t")
        parental = parental.loc[parental.pos.isin(other_genotype_chr.pos), :]
        # extract paretal lines
        parental = parental.iloc[:, 2:23]
        # select family lines
        parental = parental.loc[:, family]
        SNP_type = ["".join(i) for i in parental.values.astype("str")]
        SNP_types.extend(SNP_type)
    other_genotype["SNP_type"] = SNP_types
    
    # Get genotype from each haplotype block
    Haplotype_block = pd.read_csv(NAM_HB_genotype)
    haplotype_blocks = []
    for SNP_type in tqdm(other_genotype.SNP_type.unique()):
        other_genotype_SNP_type = other_genotype[other_genotype["SNP_type"] == SNP_type]
        HB_SNP_type = Haplotype_block[Haplotype_block["SNP_type"] == SNP_type]
        for each_chr in other_genotype_SNP_type.chr.unique():
            other_genotype_SNP_type_chr = other_genotype_SNP_type[other_genotype_SNP_type["chr"] == each_chr]
            HB_SNP_type_chr = HB_SNP_type[HB_SNP_type["chr"] == each_chr]
            for each_region in HB_SNP_type_chr.pos.values:
                each_region = each_region.split("_")
                if len(each_region) > 1:
                    start = int(each_region[0])
                    end = int(each_region[1])
                    mode_value = other_genotype_SNP_type_chr[(other_genotype_SNP_type_chr["pos"] >= start) & (other_genotype_SNP_type_chr["pos"] <= end)].mode().values[0]
                    mode_value = mode_value[2:-1]
                    haplotype_block = [each_chr, "{}_{}".format(start, end), SNP_type]
                    haplotype_block.extend(list(mode_value))
                    haplotype_blocks.append(haplotype_block)
                else:
                    mode_value = other_genotype_SNP_type_chr[other_genotype_SNP_type_chr["pos"] == int(each_region[0])].values[0]
                    mode_value = mode_value[2:-1]
                    haplotype_block = [each_chr, "{}".format(each_region[0]), SNP_type]
                    haplotype_block.extend(list(mode_value))
                    haplotype_blocks.append(haplotype_block)
    other_haplotype_blocks = pd.DataFrame(haplotype_blocks)
    columns = ["chr", "pos", "SNP_type"]
    columns.extend(list(Inov.columns[2:-1]))
    other_haplotype_blocks.columns = columns
    
    # fill na value to No genotype data regions
    final_other_haplotype_blocks = pd.DataFrame()
    for i in tqdm(range(Haplotype_block.shape[0])):
        tmp_chr = Haplotype_block.iloc[i, 0]
        tmp_pos = Haplotype_block.iloc[i, 1]
        tmp_SNP_type = Haplotype_block.iloc[i, 2]
        tmp = other_haplotype_blocks[(other_haplotype_blocks["chr"] == tmp_chr) & (other_haplotype_blocks["pos"] == tmp_pos)]
        if tmp.shape[0] > 0:
            final_other_haplotype_blocks = pd.concat([final_other_haplotype_blocks, tmp])
        else:
            tmp = [tmp_chr, tmp_pos, tmp_SNP_type]
            tmp.extend(np.repeat(np.nan, other_haplotype_blocks.shape[1]-3))
            tmp = pd.DataFrame(tmp).T
            tmp.columns = other_haplotype_blocks.columns
            final_other_haplotype_blocks = pd.concat([final_other_haplotype_blocks, tmp])
    return final_other_haplotype_blocks