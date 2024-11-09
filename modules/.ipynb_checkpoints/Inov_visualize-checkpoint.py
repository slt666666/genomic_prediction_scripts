from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import seaborn as sns
sns.set()


class Inov(Enum):

    params = {
        "colors":{
            "N05":"blue",
            "N11":"green",
            "N12":"red",
            "N14":"lightgreen",
            "N18":"skyblue",
            "NXX":"gray"
        },
        "Inov_01":[
             [8835, 8880, 'N11'],
             [15821, 15882, 'N11'],
             [15910, 16060, 'NXX'],
             [21098, 21216, 'NXX'],
             [21272, 22274, 'N11'],
             [34577, 34696, 'NXX']
        ],
        "Inov_02":[
             [798, 1331, 'N14'],
             [34249, 34590, 'N05']
        ],
        "Inov_03":[
             [22193, 23126, 'N05'],
             [24815, 25958, 'N05'],
             [31593, 35061, 'N05'],
             [42745, 45367, 'N05'],
        ],
        "Inov_04":[
             [234, 1949, 'N14'],
             [13062, 14387, 'N14'],
             [22027, 22281, 'N14'],
             [25407, 29118, 'N14'],
             [45098, 45742, 'N14'],
             [52078, 52525, 'N14'],
        ],
        "Inov_05":[
             [36, 1948, 'N11'],
             [13060, 13456, 'N14'],
             [15821, 15992, 'N12'],
             [16009, 16081, 'NXX'],
             [22225, 25119, 'N11'],
             [26234, 27994, 'N14'],
             [28208, 28572, 'NXX'],
             [28575, 29426, 'N11'],
             [30398, 30732, 'NXX'],
             [30735, 35061, 'N05'],
             [39671, 42401, 'N11'],
             [56507, 56910, 'N11'],
        ],
        "Inov_06":[
             [892, 2437, 'N14'],
             [7735, 8567, 'N12'],
             [15495, 16054, 'N12'],
             [16449, 16949, 'N05'],
             [22027, 23388, 'N14'],
             [23391, 24049, 'N12'],
             [26234, 29208, 'N14'],
             [30418, 31193, 'N05'],
             [42745, 43930, 'N12']
        ],
        "Inov_07":[
             [6960, 7243, 'N14'],
             [15495, 16942, 'N12'],
             [18537, 18667, 'N12'],
             [22936, 23126, 'N05'],
             [23354, 23850, 'N12'],
             [30398, 35061, 'N05'],
             [42773, 43930, 'N12']
        ],
        "Inov_08":[
             [234, 1949, 'N14'],
             [8835, 8880, 'N11'],
             [13060, 14390, 'N14'],
             [15821, 16081, 'NXX'],
             [21098, 21432, 'NXX'],
             [22027, 22155, 'N14'],
             [22160, 22274, 'N11'],
             [25416, 29118, 'N14'],
             [34577, 34696, 'N11'],
             [39601, 40199, 'N14'],
             [45153, 45795, 'N14'],
             [52078, 52525, 'N14'],
             [55510, 55788, 'NXX'],
        ],
        "Inov_09":[
             [0, 1429, 'N18'],
             [7718, 8404, 'N18'],
             [13563, 14280, 'N18'],
             [15832, 16942, 'N12'],
             [21381, 21674, 'N18'],
             [24107, 24709, 'N18'],
             [44924, 45518, 'N12'],
             [54096, 54333, 'N12']
        ],
        "Inov_10":[
             [234, 2437, 'N14'],
             [6995, 7238, 'N14'],
             [13060, 13456, 'N14'],
             [22027, 22283, 'N14'],
             [24927, 29893, 'N14'],
             [30398, 31611, 'N05'],
             [32964, 35061, 'N05'],
             [37062, 38476, 'N14'],
             [39643, 40190, 'N14'],
             [42758, 43618, 'N05'],
             [55300, 56476, 'N05']
        ],
        "Inov_11":[
             [234, 2437, 'N14'],
             [7735, 8568, 'N12'],
             [15495, 16445, 'N12'],
             [16449, 16949, 'N05'],
             [23354, 23796, 'N12'],
             [26235, 29893, 'N14'],
             [30398, 30809, 'N05'],
             [33206, 35061, 'N05'],
             [42745, 43930, 'N12']
        ],
        "Inov_12":[
             [0, 1290, 'N18'],
             [7718, 8076, 'N18'],
             [13568, 14280, 'N18'],
             [15944, 16939, 'N12'],
             [16950, 17076, 'N18'],
             [21381, 21674, 'N18'],
             [24788, 25959, 'N05'],
             [32964, 35056, 'N05'],
             [37936, 38493, 'N12'],
             [44159, 45518, 'N12'],
             [54096, 54333, 'N12'],
        ],
        "Inov_13":[
             [234, 891, 'N14'],
             [7011, 7238, 'N14'],
             [13060, 13607, 'N14'],
             [15832, 15976, 'N12'],
             [16449, 16949, 'N05'],
             [22027, 22283, 'N14'],
             [22936, 23126, 'N05'],
             [26234, 27994, 'N14'],
             [34472, 35061, 'N05'],
             [38247, 38493, 'N12'],
             [39522, 40057, 'N14'],
             [42745, 45367, 'N05']
        ],
        "Inov_14":[
             [234, 788, 'N14'],
             [13060, 13607, 'N14'],
             [15832, 15960, 'N12'],
             [16450, 16949, 'N05'],
             [22027, 22283, 'N14'],
             [22936, 23126, 'N05'],
             [26234, 27994, 'N14'],
             [34472, 35061, 'N05'],
             [38123, 38493, 'N12'],
             [42745, 45367, 'N05']
        ],
        "Inov_15":[
             [1295, 2431, 'N14'],
             [6995, 7238, 'N14'],
             [15495, 16432, 'N12'],
             [16449, 16949, 'N05'],
             [18537, 18667, 'N12'],
             [22027, 22283, 'N14'],
             [23354, 23796, 'N12'],
             [26234, 29893, 'N14'],
             [30398, 30799, 'N05'],
             [33185, 35061, 'N05'],
             [42745, 43930, 'N12']
        ],
        "Inov_16":[
             [234, 2431, 'N14'],
             [6988, 7243, 'N14'],
             [7735, 8567, 'N12'],
             [15495, 16378, 'N12'],
             [16449, 16949, 'N05'],
             [22027, 22188, 'N14'],
             [23354, 23796, 'N12'],
             [26234, 28201, 'N14'],
             [33206, 35061, 'N05'],
             [42745, 43925, 'N12']
        ],
        "Inov_17":[
             [6960, 7243, 'N14'],
             [8565, 8568, 'N12'],
             [15495, 16942, 'N12'],
             [18537, 18667, 'N12'],
             [22027, 22283, 'N14'],
             [23735, 23796, 'N12'],
             [26234, 29965, 'N14'],
             [30398, 30809, 'N05'],
             [33206, 35061, 'N05'],
             [42745, 43930, 'N12']
        ],
        "Inov_18":[
             [7735, 8532, 'N12'],
             [15495, 16942, 'N12'],
             [18537, 18667, 'N12'],
             [22936, 23126, 'N05'],
             [23354, 23850, 'N12'],
             [26234, 28965, 'N14'],
             [30398, 30799, 'N05'],
             [34143, 35061, 'N05'],
        ],
        "Inov_19":[
             [7735, 8567, 'N12'],
             [15495, 15825, 'N12'],
             [16449, 16949, 'N05'],
             [18537, 18667, 'N12'],
             [22936, 23126, 'N05'],
             [23354, 23850, 'N12'],
             [26234, 28159, 'N14'],
             [30398, 30809, 'N05'],
             [34143, 35061, 'N05'],
             [42745, 43930, 'N12']
        ],
        "Inov_20":[
             [8, 1772, 'N11'],
             [1837, 1993, 'N14'],
             [6960, 7243, 'N14'],
             [13060, 13630, 'N14'],
             [13637, 14249, 'N11'],
             [15821, 16081, 'NXX'],
             [23691, 24356, 'N11'],
             [24776, 25287, 'N11'],
             [26234, 28572, 'N14'],
             [30398, 30732, 'NXX'],
             [30735, 30809, 'N05'],
             [31072, 31618, 'N05'],
             [32964, 35061, 'N05'],
             [36467, 38189, 'N11'],
             [39671, 41916, 'N11'],
             [42747, 44855, 'N11'],
             [52759, 53091, 'N11'],
             [56507, 56910, 'N11']
        ],
        "Inov_21":[
             [8, 1772, 'N11'],
             [6960, 7243, 'N14'],
             [13060, 13719, 'N14'],
             [13723, 14249, 'N11'],
             [15821, 15992, 'NXX'],
             [21702, 21845, 'N11'],
             [28208, 29426, 'N11'],
             [32964, 35061, 'N05'],
             [36467, 36693, 'N11'],
             [37062, 37572, 'N14'],
             [38264, 39516, 'N11'],
             [40684, 42401, 'N11'],
             [52759, 53091, 'N11'],
             [56507, 56910, 'N11']
        ],        
    }
    

def get_regions(line, Inov_genotype_with_SNP_types):
    
    select_columns = ["chr","pos","SNP_type","N05","N11","N12","N14","N18"]
    select_columns.append(line)
    
    tmp_Inov_genotype = Inov_genotype_with_SNP_types.loc[Inov_genotype_with_SNP_types.loc[:, line].isin([1,2]), select_columns]
    regions = []
    for each_chr in tmp_Inov_genotype.chr.unique():
        tmp_chr_Inov_genotype = tmp_Inov_genotype[tmp_Inov_genotype["chr"] == each_chr]
        region_cand = []
        for i, k in enumerate(tmp_chr_Inov_genotype.index.values):
            if i == 0:
                region_cand.append([k])
            else:
                if k - region_cand[-1][-1] < 15:
                    region_cand[-1].append(k)
                else:
                    region_cand.append([k])

        for k in region_cand:
            tmp_region_Inov_genotype = tmp_Inov_genotype.loc[k[0]:k[-1], :]
            donar = tmp_region_Inov_genotype.columns[tmp_region_Inov_genotype.sum() == "2"*tmp_region_Inov_genotype.shape[0]]
            if len(donar) != 1:
                regions.append([k[0], k[-1], "NXX"])
            else:
                regions.append([k[0], k[-1], donar[0]])
    return regions


def get_hetero_regions(line, regions, Inov_genotype_with_SNP_types):
    
    select_columns = ["chr","pos","SNP_type","N05","N11","N12","N14","N18"]
    select_columns.append(line)
    
    tmp_Inov_genotype = Inov_genotype_with_SNP_types.loc[Inov_genotype_with_SNP_types.loc[:, line].isin([1,2]), select_columns]
    
    hetero_regions = []
    for region in regions:
        tmp_region_Inov_genotype = tmp_Inov_genotype.loc[region[0]:region[1], :]
        tmp_region_index = tmp_region_Inov_genotype.index.values
        tmp_hetero = []
        for i, j in enumerate(tmp_region_Inov_genotype[line].values):
            if j == 1:
                if len(tmp_hetero) == 0:
                    tmp_hetero.append(i)
                    if i == len(tmp_region_Inov_genotype[line].values) - 1:
                        hetero_regions.append([tmp_region_index[tmp_hetero[0]], tmp_region_index[tmp_hetero[-1]]])
                else:
                    tmp_hetero.append(i)
                    if i == len(tmp_region_Inov_genotype[line].values) - 1:
                        hetero_regions.append([tmp_region_index[tmp_hetero[0]], tmp_region_index[tmp_hetero[-1]]])
            else:
                if len(tmp_hetero) != 0:
                    hetero_regions.append([tmp_region_index[tmp_hetero[0]], tmp_region_index[tmp_hetero[-1]]])
                    tmp_hetero = []
                
    return hetero_regions

def visualize_Inov_genotype(Inov_genotype, chr_ends, regions, hetero_regions, hitome_color, colors, dpi=80):
    plt.rcParams['figure.dpi'] = dpi
    fig = plt.figure(figsize=(5,12))
    ax = plt.axes()
    for i, each_chr in enumerate(Inov_genotype.chr.unique()):
        r = patches.Rectangle(xy=(i*12000000, 0), width=6000000, height=-chr_ends[i]*2.5, ec='gray', fc=hitome_color, linewidth=0)
        ax.add_patch(r)

    for region in regions:
        region_genotype = Inov_genotype[(Inov_genotype.index >= region[0]) & (Inov_genotype.index <= region[1])]
        region_chr = region_genotype.chr.unique()[0]
        i = int(region_chr[-2:])-1
        start = region_genotype.start.min()
        end = region_genotype.end.max()
        color = colors[region[2]]
        r = patches.Rectangle(xy=(i*12000000, -start*2.5), width=6000000, height=-(end-start)*2.5, ec='gray', fc=color, linewidth=0)
        ax.add_patch(r)

    for hetero_region in hetero_regions:
        region_genotype = Inov_genotype[(Inov_genotype.index >= hetero_region[0]) & (Inov_genotype.index <= hetero_region[1])]
        region_chr = region_genotype.chr.unique()[0]
        i = int(region_chr[-2:])-1
        start = region_genotype.start.min()
        end = region_genotype.end.max()
        r = patches.Rectangle(xy=(i*12000000, -start*2.5), width=3000000, height=-(end-start)*2.5, ec='gray', fc=hitome_color, linewidth=0)
        ax.add_patch(r)

    for i, each_chr in enumerate(Inov_genotype.chr.unique()):
        r = patches.Rectangle(xy=(i*12000000, 0), width=6000000, height=-chr_ends[i]*2.5, ec='gray', fill=False, linewidth=3)
        ax.add_patch(r)

    plt.axis('scaled')
    plt.axis('off')
    plt.tight_layout()
    ax.set_aspect('equal')
    plt.show()