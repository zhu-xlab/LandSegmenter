import os

dataset_names = ['dw', 'dynamicearthnet', 'openearthmap']
weak_lmdb_dir = '/p/scratch/hai_ws4eo/liu_ch/data/WeakFT/weak_lmdbs'

datasets_settings_train = {
    'dw': {
        'data_path': '/p/project1/hai_dm4eo/liu_ch/data/SSL4EO-S2/downstream_lmdb_small_remove/dw_train_lmdb',
        'mode': 's2',
        'n_channels': 13,
        },

    'dynamicearthnet': {
        'data_path': '/p/project1/hai_ws4eo/liu_ch/data/WeakFT/exact/dynamicearthnet_seasonal.lmdb',
        },
    
    'openearthmap': {
        'data_path': '/p/project1/hai_ws4eo/liu_ch/data/WeakFT/exact/openearthmap_train.lmdb',
        },
    
    'iran': {
        'data_path': os.path.join(weak_lmdb_dir, 'iran.lmdb'),
        'N': 4000,
        },
    
    'ghsl': {
        'data_path': os.path.join(weak_lmdb_dir, 'ghsl_select.lmdb'),
        'N': -1,
        },

    'worldcover': {
        'data_path_v100': os.path.join(weak_lmdb_dir, 'worldcover_v100.lmdb'),
        'data_path_v200': os.path.join(weak_lmdb_dir, 'worldcover_v200.lmdb'),
        'N': 15000,
        'mode': 'rand',
        'n_bands': 12,
    },

    'nlcd': {
        'data_path': os.path.join(weak_lmdb_dir, 'nlcd.lmdb'),
        'N': 8000,
    },

    'usfs': {
        'data_path': os.path.join(weak_lmdb_dir, 'usfs.lmdb'),
        'N': 8000,
    },

    'sbtn': {
        'data_path': os.path.join(weak_lmdb_dir, 'sbtn.lmdb'),
        'N': 15000,
    },

}


datasets_settings_test = {
    'dw': {
        'data_path': '/p/project1/hai_dm4eo/liu_ch/data/SSL4EO-S2/downstream_lmdb_small_remove/dw_test_lmdb',
        'mode': 's2',
        'n_channels': 13,
    },

    'osm': {
        'data_path': '/p/scratch/hai_ws4eo/liu_ch/data/WeakFT/test_data/dwosm/osm_test_lmdb',
        'mode': 's2',
        'n_channels': 13,
    },

    'openearthmap': {
        'data_path': '/p/project1/hai_ws4eo/liu_ch/data/WeakFT/exact/openearthmap_val.lmdb',
    },

    'multisenge': {
        'data_path': '/p/scratch/hai_ws4eo/liu_ch/data/WeakFT/test_data/multisenge/multisenge.lmdb',
    },
}


class_names = {
    'dynamicearthnet': {
        0: ['developed impervious area', 'built-up impervious area'],
        1: ['agricultural land', 'crop', 'crop land', 'arable land and permanent crop'],
        2: ['forest, grass, shrub, pasture and artificial vegetation', 'tree, herbaceous vegetation and scrub and artificial vegetation', 'tree, herb, shrub and artificial vegetation', 'tree, herbaceous vegetation, shrub and artificial vegetation', 'tree, grass, shrub, pasture and artificial vegetation', 'forest, herbaceous vegetation, scrub and artificial vegetation', 'forest, herb, shrub and artificial vegetation'],
        3: ['wetland', 'inland wetland and coastal wetland', 'herbaceous wetland and woody wetland'],
        4: ['barren land', 'bare land', 'rock, sand, clay and soil'],
        5: ['water', 'lake, reservoir, river and ocean'],
        6: ['snow and ice'],
    },

    'openearthmap': {
        0: ['barren land', 'bare land', 'rock, sand, clay and soil'],
        1: ['rangeland and herbaceous wetland', 'grass, pasture, scrub and herbaceous wetland', 'herbaceous vegetation, shrub and herbaceous wetland', 'grass, pasture, shrub and herbaceous wetland', 'herbaceous vegetation, scrub and herbaceous wetland'],
        2: ['developed area except for building', 'built-up area except for building'],
        3: ['road', 'transportation'],
        4: ['tree and woody wetland'],
        5: ['water', 'lake, reservoir, river and ocean'],
        6: ['agricultural land', 'crop', 'crop land', 'arable land and permanent crop'],
        7: ['building'], 
        },

    'iran':{
        0: ['urban area', 'developed area', 'residential, commercial, industrial and transportation area', 'built-up area'],
        1: ['water', 'lake, reservoir, river and ocean'],
        2: ['wetland except for marshland'],
        3: ['barren land except for salty land and sand', 'bare land except for salty land and sand', 'rock, clay and soil except for salty land'],
        4: ['marshland'],
        5: ['salty land'],
        6: ['tree', 'forest', 'wood', 'broadleaf and coniferous forest', 'deciduous and evergreen forest', 'broadleaf and coniferous tree', 'deciduous and evergreen tree'],
        7: ['sand'],
        8: ['agricultural land', 'crop', 'cropland', 'arable land and permanent crop', 'herbaceous crop and woody crop', 'annual crop, orchard and vineyard'],
        9: ['rangeland', 'grass, pasture and scrub', 'herbaceous vegetation and shrub', 'grass, pasture and shrub', 'herbaceous vegetation and scrub'],
        },

    'worldcover':{
        0: ['tree including corresponding artificial vegetation and woody crop but except for mangrove', 'forest including corresponding artificial vegetation and woody crop but except for mangrove', 'wood including corresponding artificial vegetation and woody crop but except for mangrove'],
        1: ['shrub and scrub including corresponding artificial vegetation and woody crop', 'shrub including corresponding artificial vegetation and woody crop', 'scrub including corresponding artificial vegetation and woody crop'],
        2: ['grassland and pasture including corresponding artificial vegetation', 'herb and pasture including corresponding artificial vegetation', 'herbaceous vegetation and pasture including corresponding artificial vegetation', 'grassland and meadow including corresponding artificial vegetation', 'herb and meadow including corresponding artificial vegetation', 'herbaceous vegetation and meadow including corresponding artificial vegetation'],
        3: ['agricultural land except for woody crop', 'crop except for woody crop', 'cropland except for woody crop', 'arable land', 'herbaceous crop', 'annual crop'],
        4: ['built-up area without artificial vegetation', 'man-made structure without artificial vegetation', 'residential, commercial, industrial and transportation area except for artificial vegetation', 'developed area except for artificial vegetation'],
        5: ['barren land including mine site, dump site and construction site', 'bare land including mine site, dump site and construction site', 'rock, sand, clay and soil including mine site, dump site and construction site'],
        6: ['snow and ice'],
        7: ['water', 'lake, reservoir, river and ocean'],
        8: ['herbaceous wetland', 'non-forest wetland'],
        9: ['mangrove'],
        10:['moss and lichen'],
    },

    # 'usfs-lc':{
    #     0: ['tree', 'forest', 'wood', 'broadleaf forest and coniferous forest', 'deciduous forest and evergreen forest', 'broadleaf tree and coniferous tree', 'deciduous tree and evergreen tree'],
    #     1: ['mixed shrub and tree area', 'mixed scrub and tree area'],
    #     2: ['mixed grass and tree area', 'mixed herb and tree area', 'mixed herbaceous and woody area'],
    #     3: ['mixed barren and tree area', 'mixed barren and woody area'],
    #     4: ['shrub and scrub', 'shrub', 'scrub'],
    #     5: ['mixed grass and shrub area', 'mixed herb and shrub area', 'mixed grass and scrub area', 'mixed herb and scrub area'],
    #     6: ['mixed barren and shrub area', 'mixed barren and scrub area'],
    #     7: ['agricultural land and grassland', 'agricultural land and herbaceous vegetation', 'agricultural land and herb', 'grass or herb'],
    #     8: ['mixed grass and barren area', 'mixed herb and barren area'],
    #     9: ['barren and impervious area', 'barren land and impervious area'],
    #     10:['snow and ice'],
    #     11:['water', 'lake, reservoir, river and ocean'],
    # },

    # 'usfs-lu':{
    'usfs':{
        0: ['agricultural land', 'crop', 'cropland', 'arable land and permanent crop', 'herbaceous crop and woody crop', 'annual crop, orchard and vineyard'],
        1: ['developed area', 'urban area', 'residential, commercial, industrial and transportation area including artificial vegetation', 'built-up area'],
        2: ['tree and woody wetland', 'forest and woody wetland', 'wood and woody wetland', 'broadleaf forest, coniferous forest, and woody wetland', 'deciduous forest, evergreen forest, and woody wetland', 'broadleaf tree, coniferous tree, and woody wetland', 'deciduous tree, evergreen tree, and woody wetland'],
        3: ['herbaceous wetland', 'non-forest wetland'],
        4: ['rangeland', 'grass, shrub and pasture', 'herbaceous vegetation and shrub', 'grass, scrub and meadow', 'herbaceous vegetation and scrub'],
        },

    'nlcd-lc':{
        0: ['water', 'lake, reservoir, river and ocean'], #'Open water: areas of open water, generally with less than 0.25 cover of vegetation or soil'],
        1: ['ice and snow'], # Perennial ice or snow: 'areas characterized by a perennial cover of ice and/or snow, generally greater than 25% of total cover'],
        2: ['artificial vegetation', 'non-argricultural vegetation', 'artificial or non-argricultural vegetation'],  # Developed, open space: 'areas with a mixture of some constructed materials, but mostly vegetation in the form of lawn grasses. Impervious surfaces account for less than 20% of total cover. These areas most commonly include large-lot single-family housing units, parks, golf courses, and vegetation planted in developed settings for recreation, erosion control, or aesthetic purposes'], 
        3: ['developed low-intensity impervious area', 'built-up low-intensity impervious area'], # Developed, low intensity: 'areas with a mixture of constructed materials and vegetation. Impervious surfaces account for 20% to 49% percent of total cover. These areas most commonly include single-family housing units'],
        4: ['developed medium-intensity impervious area', 'built-up medium-intensity impervious area'], # Developed, medium intensity: 'areas with a mixture of constructed materials and vegetation. Impervious surfaces account for 50% to 79% of the total cover. These areas most commonly include single-family housing units'],
        5: ['developed high-intensity impervious area', 'built-up high-intensity impervious area'], # Developed high intensity: 'highly developed areas where people reside or work in high numbers. Examples include apartment complexes, row houses, and commercial/industrial. Impervious surfaces account for 80% to 100% of the total cover'],
        6: ['barren land inclduing mine site, dump site and construction site', 'bare land inclduing mine site, dump site and construction site', 'rock, sand, clay and soil inclduing mine site, dump site and construction site'], # Barren land (rock or sand or clay): 'areas of bedrock, desert pavement, scarps, talus, slides, volcanic material, glacial debris, sand dunes, strip mines, gravel pits, and other accumulations of earthen material. Generally, vegetation accounts for less than 15% of total cover'],
        7: ['deciduous forest and broadleaf forest', 'deciduous tree and broadleaf tree', 'deciduous wood and broadleaf wood'], # Deciduous forest: 'areas dominated by trees generally greater than 5 meters tall, and greater than 20% of total vegetation cover. More than 75% of the tree species shed foliage simultaneously in response to seasonal change'],
        8: ['evergreen forest and coniferous forest', 'evergreen tree and coniferous tree', 'coniferous wood and deciduous wood'], # Evergreen forest: 'areas dominated by trees generally greater than 5 meters tall, and greater than 20% of total vegetation cover. More than 75% of the tree species maintain their leaves all year. Canopy is never without green foliage'],
        9: ['mixed broadleaf and coniferous forest', 'mixed deciduous and evergreen forest', 'mixed broadleaf and coniferous tree', 'mixed deciduous and evergreen tree', 'mixed broadleaf and coniferous wood', 'mixed deciduous and evergreen wood'], # Mixed forest: 'areas dominated by trees generally greater than 5 meters tall, and greater than 20% of total vegetation cover. Neither deciduous nor evergreen species are greater than 75% of total tree cover'],
        10:['shrub and scrub', 'shrub', 'scrub'], # Shrub/scrub: 'areas dominated by shrubs less than 5 meters tall with shrub canopy typically greater than 20% of total vegetation. This class includes true shrubs, young trees in an early successional stage, or trees stunted from environmental conditions'],
        11:['grassland and herbaceous vegetation', 'grassland and herb', 'grass and herb'], # Grassland/herbaceous: 'areas dominated by gramanoid or herbaceous vegetation, generally greater than 80% of total vegetation. These areas are not subject to intensive management such as tilling, but can be utilized for grazing'],
        12:['pasture and hay', 'pasture', 'meadow'], # Pasture/hay: 'areas of grasses, legumes, or grass-legume mixtures planted for livestock grazing or the production of seed or hay crops, typically on a perennial cycle. Pasture/hay vegetation accounts for greater than 20% of total vegetation'],
        13:['agricultural land', 'crop', 'cropland', 'arable land and permanent crop', 'herbaceous crop and woody crop', 'annual crop, orchard and vineyard'], # Cultivated crops: 'areas used for the production of annual crops, such as corn, soybeans, vegetables, tobacco, and cotton, and also perennial woody crops such as orchards and vineyards. Crop vegetation accounts for greater than 20% of total vegetation. This class also includes all land being actively tilled'],
        14:['Woody wetland'], # Woody wetlands: 'areas where forest or shrubland vegetation accounts for greater than 20% of vegetative cover and the soil or substrate is periodically saturated with or covered with water'],
        15:['herbaceous wetland', 'non-forest wetland'], # Emergent herbaceous wetlands: 'areas where perennial herbaceous vegetation accounts for greater than 80% of vegetative cover and the soil or substrate is periodically saturated with or covered with water'],
    },

    'nlcd-imp':{
        0: ['road', 'transportation'],
        1: ['non-road and non-energy-related impervious area', 'non-transportation and non-energy-related impervious area', 'impervious area except for road and energy-related area'],
        2: ['energy-related impervious area'],
    },

    'sbtn':{
        0: ['tree', 'forest', 'wood', 'broadleaf forest and coniferous forest', 'deciduous forest and evergreen forest', 'broadleaf tree and coniferous tree', 'deciduous tree and evergreen tree'],
        1: ['rangeland', 'grass, pasture and scrub', 'herbaceous vegetation and shrub', 'grass, pasture and shrub', 'herbaceous vegetation and scrub'],
        2: ['water', 'lake, reservoir, river and ocean'],
        3: ['mangrove'],
        4: ['barren land', 'bare land', 'rock, sand, clay and soil'],
        5: ['ice and snow'],
        6: ['woody wetland'],
        7: ['herbaceous wetland', 'non-forest wetland'],
        8: ['agricultural land', 'crop', 'cropland', 'arable land and permanent crop', 'herbaceous crop and woody crop', 'annual crop, orchard and vineyard'],
        9: ['urban area including mine site, dump site and construction site', 'developed area including mine site, dump site and construction site', 'residential, commercial, industrial and transportation area including mine site, dump site and construction site', 'built-up area including mine site, dump site and construction site'],
        10: ['artificial vegetation'],   # ??? to double check
    },

    'ghsl':{
        0: ['non-developed area', 'non-built-up area', 'pervious area'],
        1: ['residential area'],
        2: ['non-residential built-up area', 'non-residential developed area', 'commercial, industrial and transportation area'],
    },

    # # # # # # # # test sets # # # # # # # #
    'multisenge': {
        0: ['built-up high-intensity impervious area'],  # Dense built-up areas
        1: ['built-up low-intensity imperious area'],   # sparely built-up areas
        2: ['industrial, commercial and transportation area'],              # Specialized built-Up Areas
        3: ['artificial, non-argricultural vegetation'],     # Specialized but vegetative areas     
        4: ['transportation'],       # Large scale networks
        5: ['arable lands'],
        6: ['vineyards'],
        7: ['orchards'],
        8: ['grasslands'],
        9: ['shrub and scrub'],
        10:['forest'],
        11:['bare land'],
        12:['wetland'],
        13:['water'],
    },

    # # # # # these two correspond to the visualization order not the order saved in lmdb # # # # #
    # 'osm':{
    #     0: ['Urban fabric'],
    #     1: ['Industrial, commercial and transport'],
    #     2: ['Arable land'],
    #     3: ['Artificial, non-argricultural vegetation'],
    #     4: ['Forest'],
    #     5: ['Mine, dump and construction sites'],
    #     6: ['Water bodies'],
    #     7: ['Open spaces'],
    #     8: ['Pastures'],
    #     9: ['Shrub and herbaceous associations'],
    #     10:['Permanent crops'],
    #     11:['Wetlands'],
    #     12:['Coastal wetlands'],
    # },

    # 'osm':{
    #     0: ['Urban fabric'],
    #     1: ['Industrial, commercial and transport'],
    #     2: ['Arable land'],
    #     3: ['Artificial, non-argricultural vegetation'],
    #     4: ['Forest'],
    #     5: ['Mine, dump and construction sites'],
    #     6: ['Waterbody'],
    #     7: ['Open spaces'],
    #     8: ['Pastureland'],
    #     9: ['Shrub and herbaceous associations'],
    #     10:['Permanent crops'],
    #     11:['Wetlands'],
    #     12:['Coastal wetlands'],
    # },
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    'dw': {
        0: ['water', 'Open water', 'Aquatic surface', 'Waterbody', 'Lake and river', 'Hydric zone', 'Aquatic zone', 'Marine and freshwater area'],
        1: ['tree', 'Forest', 'Woodland', 'Canopy', 'Timberland', 'Forested area', 'Arboreal zone'],
        2: ['grass', 'Grassland', 'Prairie', 'Savanna', 'Pastureland', 'Herbaceous cover'],
        3: ['wetland', 'Flooded vegetation', 'Wetland vegetation', 'Marshland', 'Swamp plant', 'Riparian area', 'Aquatic vegetation'], 
        4: ['crop', 'Farmland', 'Agricultural field', 'Cultivated land', 'Arable land', 'Cropland'],
        5: ['shrub and scrub', 'Shrubland', 'Heath', 'Brushland', 'Chaparral', 'Low vegetation'],
        6: ['built-up area', 'Urban area', 'Developed land', 'Residential and commercial zone', 'Infrastructure', 'Built-up land'],
        7: ['bare land', 'Barren terrain', 'Open soil', 'Unvegetated land', 'Exposed earth'],
        8: ['ice and snow', 'Glacial area', 'Snow cover', 'Ice field', 'Frozen surface'],
        9: ['Cloud', 'Cloud cover', 'Cloudy sky', 'Cloud layer', 'Cloud formation'],
    },

    'osm':{
        0: ['residential area'],
        1: ['arable land'],
        2: ['forest'],
        3: ['industrial, commercial and transportation area'],
        4: ['artificial, non-argricultural vegetation'],
        5: ['mine, dump and construction sites'],
        6: ['pasture'],
        7: ['permanent crops'],
        8: ['water'],
        9: ['bare land'],
        10:['shrub and herbaceous vegetation'],
        11:['inland wetland'],
        12:['coastal wetland'],
    },

}


class_weight_list = {
    'dw': [1.9, 1.0, 3.7, 4.8, 1.4, 1.0, 2.9, 1.7, 5.0, 10.0],
    'dynamicearthnet': [3.5, 2.8, 1.0, 10.0, 1.3, 3.2, 10.0],
    'openearthmap': [7.6, 1.0, 1.2, 2.1, 1.1, 3.8, 1.3, 1.2],
    'iran': [4.5, 10.0, 3.3, 1.0, 10.0, 10.0, 10.0, 6.3, 1.1, 2.1],
    'worldcover': [1.1, 2.2, 1.1, 1.0, 2.9, 1.5, 10.0, 1.6, 10.0, 10.0, 10.0],
    'usfs-lc': [1.0, 10.0, 5.1, 10.0, 4.5, 1.8, 7.1, 1.0, 10.0, 4.5, 10.0, 9.2],
    'usfs-lu': [1.6, 7.0, 1.0, 10.0, 1.0],
    'nlcd-lc': [6.8, 10.0, 6.4, 10.0, 10.0, 10.0, 10.0, 2.0, 1.5, 4.8, 1.0, 1.3, 3.0, 1.3, 3.7, 9.8],
    'nlcd-imp': [1.0, 1.7, 10.0],
    'sbtn': [1.2, 1.1, 6.7, 10.0, 1.4, 10.0, 5.6, 5.5, 1.0, 3.1, 2.3],
    'ghsl': [1.0, 1.1, 7.1],
}


class_weight_dict = {
    'dw': {
        0: 1.9,
        1: 1.0,
        2: 3.7,
        3: 4.8,
        4: 1.4,
        5: 1.0,
        6: 2.9,
        7: 1.7,
        8: 5.0,
        9: 10.0,
    },

    'dynamicearthnet': {
        0: 3.5,
        1: 2.8,
        2: 1.0,
        3: 10.0,
        4: 1.3,
        5: 3.2,
        6: 10.0,
    },

    'openearthmap': {
        0: 7.6,
        1: 1.0,
        2: 1.2,
        3: 2.1,
        4: 1.1,
        5: 3.8,
        6: 1.3,
        7: 1.2, 
        },

    'iran':{
        0: 4.5, 
        1: 10.0,
        2: 3.3,
        3: 1.0,
        4: 10.0,
        5: 10.0,
        6: 10.0,
        7: 6.3,
        8: 1.1,
        9: 2.1,
        },

    'worldcover':{
        0: 1.1,
        1: 2.2,
        2: 1.1,
        3: 1.0,
        4: 2.9,
        5: 1.5,
        6: 10.0,
        7: 1.6,
        8: 10.0,
        9: 10.0,
        10: 10.0,
    },

    'usfs-lc':{
        0: 1.0,
        1: 10.0,
        2: 5.1,
        3: 10.0,
        4: 4.5,
        5: 1.8,
        6: 7.1,
        7: 1.0,
        8: 10.0,
        9: 4.5,
        10: 10.0,
        11: 9.2,
    },

    'usfs-lu':{
        0: 1.6,
        1: 7.0,
        2: 1.0,
        3: 10.0,
        4: 1.0,
        },

    'nlcd-lc':{
        0: 6.8,
        1: 10.0,
        2: 6.4,
        3: 10.0,
        4: 10.0,
        5: 10.0,
        6: 10.0,
        7: 2.0,
        8: 1.5,
        9: 4.8,
        10: 1.0,
        11: 1.3,
        12: 3.0,
        13: 1.3,
        14: 3.7,
        15: 9.8, 
    },

    'nlcd-imp':{
        0: 1.0,
        1: 1.7,
        2: 10.0,
    },

    'sbtn':{
        0: 1.2,
        1: 1.1,
        2: 6.7,
        3: 10.0,
        4: 1.4,
        5: 10.0,
        6: 5.6,
        7: 5.5,
        8: 1.0,
        9: 3.1,
        10: 2.3,    
    },

    'ghsl':{
        0: 1.0,
        1: 1.1,
        2: 7.1,
    },
}