import numpy as np
import os

#------------------------------------------------------------ SUB ------------------------------------------------------------
# # # without iran
test_weight_path_0='Results_sub_woiran/FT_dofabAx0.1_aux_attcp_exact_sub_max1500_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_5'

# # # without usfs-lc
# # 1 branch
test_weight_path_1='Results_sub_wousfslc/weak/FT_dofabAx0.1_attcp_exact_sub_max1500_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_5'
# # with aux-decoder
test_weight_path_2='Results_sub_wousfslc/weak/FT_dofabAx0.1_aux_attcp_exact_sub_max1500_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_5'

# # # with usfs-lc
# # with separate aux-decoders + fixed dofa + noisy label input loss
test_weight_path_3='Results_sub_wusfslc/FT_dofab_auxl_attcp_exact_sub_max1500_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_6'
# # with separate aux-decoders for each dataset + fixed dofa
test_weight_path_4='Results_sub_wusfslc/FT_dofab_sepaux_attcp_exact_sub_max1500_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_4'
# # 1 branch + trained dofa
test_weight_path_5='Results_sub_wusfslc/FT_dofabAx0.1_aux_attcp_exact_sub_max1500_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_5'
# # with separate aux-decoders + trained dofa + noisy label input loss
test_weight_path_6='Results_sub_wusfslc/FT_dofabAx0.1_auxl_attcp_exact_sub_max1500_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_5'
# # with separate aux-decoders for each dataset + trained dofa
test_weight_path_7='Results_sub_wusfslc/FT_dofabAx0.1_sepaux_attcp_exact_sub_max1500_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_4'


#------------------------------------------------------------ ALL ------------------------------------------------------------
# without usfs-lc
# 1 branch
test_weight_path0='Results_weak_wousfslc/FT_dofabAx0.1_attcp_exact_max15000_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_12'
# with aux-decoder
test_weight_path1='Results_weak_wousfslc/FT_dofabAx0.1_aux_attcp_exact_max15000_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_15'
# sepaux
test_weight_path2='Results_weak_wousfslc/FT_dofabAx0.1_sepaux_attcp_exact_max15000_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_12'


# with usfs-lc
# 1 branch
test_weight_path3='Results_weak/FT_dofabAx0.1_attcp_exact_max15000_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_15'
# with aux-decoder
test_weight_path4='Results_weak/FT_dofabAx0.1_aux_attcp_exact_max15000_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_15'
# # sepaux
test_weight_path5='Results_weak/FT_dofabAx0.1_sepaux_attcp_exact_max15000_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_12'
#------------------------------------------------------------ ALL ------------------------------------------------------------


#----------------------ALL-new with openearthmap=1024------------------------------------------
# lr=1e-4
# single branch
fp0='Results_all/FT_dofabAx0.1_attcp_exact_max15000_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_12'
# aux branch
fp1='Results_all/FT_dofabAx0.1_aux_attcp_exact_max15000_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_12'
# sep branch
fp2='Results_all/FT_dofabAx0.1_sepaux_attcp_exact_max15000_deI_mcloss/0F_1D_1C_lr1e4_cos_w5_cos95/seed_42_bs_12'

# lr=5e-4
# single branch
fp3='Results_all/FT_dofabAx0.1_attcp_exact_max15000_deI_mcloss/0F_1D_1C_lr5e4_cos_w5_cos95/seed_42_bs_12'
# aux branch
fp4='Results_all/FT_dofabAx0.1_aux_attcp_exact_max15000_deI_mcloss/0F_1D_1C_lr5e4_cos_w5_cos95/seed_42_bs_12'
# sep branch
fp5='Results_all/FT_dofabAx0.1_sepaux_attcp_exact_max15000_deI_mcloss/0F_1D_1C_lr5e4_cos_w5_cos95/seed_42_bs_12'


test_weight_paths = [fp0, fp1, fp2, fp3, fp4, fp5]


# test_weight_paths = [# test_weight_path_0, test_weight_path_1, test_weight_path_2, test_weight_path_3, 
#                      # test_weight_path_4, test_weight_path_5, test_weight_path_6, test_weight_path_7, 
#                      test_weight_path0, test_weight_path1, test_weight_path2, 
#                      test_weight_path3, test_weight_path4, test_weight_path5]
pn_acc = 'accuracy'
datasets = ['potsdam', 'loveda', 'nyc', 'dw', 'osm', 'multisenge'] 

EPs = [99]
print_img_wise = False
for d in datasets:
    print(f'=========={d}==========')
    for ep in EPs:
        for i, path in enumerate(test_weight_paths):
            # fp = os.path.join(parent_dir, path, f'pred_EP{ep}_{d}.npy')
            fp = os.path.join(path, pn_acc, f'pred_EP{ep}_{d}.npy')
            pred = np.load(fp)
            tmp = [float(f'{p*100:.2f}') for p in pred[0]]
            if print_img_wise: tmp2 = [f'{p*100:.2f}' for p in pred[-1]]
            if d in ['potsdam', 'loveda']:
                endi = -2 if d == 'potsdam' else -1
                tmp.append(float(f'{np.mean(pred[0][1:endi])*100:.2f}'))
                if print_img_wise: tmp2.append(f'{np.mean(pred[-1][1:endi])*100:.2f}')
                n = 3
            else:
                n = 2
            print(f'{ep}-{i}: {tmp[:-n]} {tmp[-n:]}')
            if print_img_wise: print(f'{ep}-{i}: {tmp2[:-n]} {tmp2[-n:]}')
