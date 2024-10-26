dataset_names = ['dw', 'dynamicearthnet', 'openearthmap']

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
        }
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
    'dw': {
        0: ['Water', 'Open water', 'Aquatic surface', 'Waterbody', 'Lake and river', 'Hydric zone', 'Aquatic zone', 'Marine and freshwater area'],
        1: ['Tree', 'Forest', 'Woodland', 'Canopy', 'Timberland', 'Forested area', 'Arboreal zone'],
        2: ['Grass', 'Grassland', 'Prairie', 'Savanna', 'Pastureland', 'Herbaceous cover'],
        3: ['Flooded vegetation', 'Wetland vegetation', 'Marshland', 'Swamp plant', 'Riparian area', 'Aquatic vegetation'], 
        4: ['Crop', 'Farmland', 'Agricultural field', 'Cultivated land', 'Arable land', 'Cropland'],
        5: ['Shrub and Scrub', 'Shrubland', 'Heath', 'Brushland', 'Chaparral', 'Low vegetation'],
        6: ['Built area', 'Urban area', 'Developed land', 'Residential and commercial zone', 'Infrastructure', 'Built-up land'],
        7: ['Bare land', 'Barren terrain', 'Open soil', 'Unvegetated land', 'Exposed earth'],
        8: ['Ice and Snow', 'Glacial area', 'Snow cover', 'Ice field', 'Frozen surface'],
        9: ['Cloud', 'Cloud cover', 'Cloudy sky', 'Cloud layer', 'Cloud formation'],
    },

    'dynamicearthnet': {
        0: ['impervious surface', 'Paved surface', 'Built-up area', 'Sealed land', 'Non-permeable area', 'Urban pavement'],
        1: ['agriculture', 'Farmland', 'Cropland', 'Arable land', 'Cultivated area', 'Agricultural field'],
        2: ['forest and other vegetation', 'Tree and grass', 'Woodland and grassland', 'Vegetated area', 'Green cover', 'Arboreal and herbaceous zone'],
        3: ['Wetland', 'Marshland', 'Swamp and Bog', 'Flooded vegetation', 'Peatland', 'Riparian area'],
        4: ['soil', 'Bare soil', 'Unvegetated land', 'Exposed earth', 'Barren terrain', 'Open ground'],
        5: ['water', 'Open water', 'Aquatic zone', 'Waterbody', 'Lake and river', 'Hydric zone', 'Marine and freshwater area'],
        6: ['snow and ice', 'Snow cover', 'Ice field', 'Frozen ground', 'Permafrost and Snowpack'],
    },

    'openearthmap': {
        0: ['Bareland', 'Bare soil', 'Barren land', 'Open ground', 'Unvegetated land', 'Exposed ground', 'Desolate area'],
        1: ['Rangeland', 'Pastureland', 'Grassland', 'Prairie', 'Grazing land', 'Herbaceous cover'],
        2: ['Developed space', 'Urban area', 'Built-up area', 'Residential and commercial area', 'Constructed zone'],
        3: ['Road', 'Highway amd Street and Path', 'Transportation area', 'Road network'],
        4: ['Tree', 'Forest', 'Woodland', 'Canopy', 'Timberland', 'Forested area', 'Arboreal zone'],
        5: ['Water', 'Open water', 'Aquatic area', 'Waterbody', 'Lakes and rivers', 'Hydrological features', 'Marine and freshwater areas'],
        6: ['Agriculture land', 'Farmland', 'Cropland', 'Arable land', 'Cultivated field', 'Agricultural plot', 'Crop zones'],
        7: ['Building', 'Built-up areas', 'Construction', 'Infrastructure', 'Dwelling and Commercial Facilities'],
    },

    'iran':{
        0: ['Urban', 'Urban area', 'Urban zones', 'City', 'Metropolis', 'Town', 'Residential and Industrial zones', 'Residential and Industrial areas'],
        1: ['Water', 'Open water', 'Aquatic area', 'Waterbody', 'Water body', 'Lakes, rivers, and sea or ocean', 'Hydrological features', 'Marine and freshwater areas'],
        2: ['Wetlands'],
        3: ['Kalut or yardang', 'Wind-eroded ridge', 'Desert ridge', 'Aeolian formation', 'Wind-carved structure', 'Rock outcrop', 'Desert landform', 'Wind-sculpted feature'],
        4: ['Marshland'],
        5: ['Salty land', 'Salt flat', 'Salt pan', 'Saline soil', 'Saline area', 'Salty flat'],
        6: ['Clay', 'Mud', 'Argillaceous earth', 'Clayey ground', 'Silt', 'Heavy soil'],
        7: ['Forest', 'Trees', 'Jungle', 'Grove', 'Wood', 'Timberland', 'Woodland', 'Wood land'],
        8: ['Outcrop', 'Rock exposure', 'Bedrock deposit', 'Surface rock', 'Geological formation', 'Mountain outcrop', 'Eroded bedrock', 'Rocky protrusion'],
        9: ['Uncovered plain', 'Open field', 'Open space'],
        10:['Sand', 'Dune', 'Sandy soil', 'Desert', 'Sandbank', 'Sand drift', 'Sandpit'],
        11:['Farmland', 'Agricultural land', 'Cropland', 'Arable land', 'Arable field', 'Farm field', 'Cultivated land', 'Agrarian area'],
        12:['Rangeland', 'Pastureland', 'Grazing land', 'Grassland', 'Meadow'],
    },

    'worldcover':{
        0: ['Tree cover'],
        1: ['Shrubland'],
        2: ['Grassland'],
        3: ['Cropland'],
        4: ['Built-up'],
        5: ['Bare / sparse vegetation'],
        6: ['Snow and ice'],
        7: ['Permanent water bodies'],
        8: ['Herbaceous wetland'],
        9: ['Mangroves'],
        10:['Moss and lichen'],
    },

    'usfs-lc':{
        0: ['Trees'],
        1: ['Shrubs and tree mix'],
        2: ['Grass or forb or herb and trees mix'],
        3: ['Barren and trees mix'],
        4: ['Shrubs'],
        5: ['Grass or forb or herb and shrubs mix'],
        6: ['Barren and shrubs mix'],
        7: ['Grass or forb or herb'],
        8: ['Barren and grass or forb or herb mix'],
        9: ['Barren or impervious'],
        10:['Snow or ice'],
        11:['Water'],
    },

    'usfs-lu':{
        0: ['Agriculture'],
        1: ['Developed'],
        2: ['Forest'],
        3: ['Non-Forest Wetland'],
        4: ['Rangeland or Pasture'],
        },

    'nlcd-lc':{
        0: ['Open water'], #'Open water: areas of open water, generally with less than 0.25 cover of vegetation or soil'],
        1: ['Perennial ice or snow'], # Perennial ice or snow: 'areas characterized by a perennial cover of ice and/or snow, generally greater than 25% of total cover'],
        2: ['Developed, open space'],  # Developed, open space: 'areas with a mixture of some constructed materials, but mostly vegetation in the form of lawn grasses. Impervious surfaces account for less than 20% of total cover. These areas most commonly include large-lot single-family housing units, parks, golf courses, and vegetation planted in developed settings for recreation, erosion control, or aesthetic purposes'], 
        3: ['Developed, low intensity'], # Developed, low intensity: 'areas with a mixture of constructed materials and vegetation. Impervious surfaces account for 20% to 49% percent of total cover. These areas most commonly include single-family housing units'],
        4: ['Developed, medium intensity'], # Developed, medium intensity: 'areas with a mixture of constructed materials and vegetation. Impervious surfaces account for 50% to 79% of the total cover. These areas most commonly include single-family housing units'],
        5: ['Developed high intensity'], # Developed high intensity: 'highly developed areas where people reside or work in high numbers. Examples include apartment complexes, row houses, and commercial/industrial. Impervious surfaces account for 80% to 100% of the total cover'],
        6: ['Barren land (rock or sand or clay)'], # Barren land (rock or sand or clay): 'areas of bedrock, desert pavement, scarps, talus, slides, volcanic material, glacial debris, sand dunes, strip mines, gravel pits, and other accumulations of earthen material. Generally, vegetation accounts for less than 15% of total cover'],
        7: ['Deciduous forest'], # Deciduous forest: 'areas dominated by trees generally greater than 5 meters tall, and greater than 20% of total vegetation cover. More than 75% of the tree species shed foliage simultaneously in response to seasonal change'],
        8: ['Evergreen forest'], # Evergreen forest: 'areas dominated by trees generally greater than 5 meters tall, and greater than 20% of total vegetation cover. More than 75% of the tree species maintain their leaves all year. Canopy is never without green foliage'],
        9: ['Mixed forest'], # Mixed forest: 'areas dominated by trees generally greater than 5 meters tall, and greater than 20% of total vegetation cover. Neither deciduous nor evergreen species are greater than 75% of total tree cover'],
        10:['Shrub or scrub'], # Shrub/scrub: 'areas dominated by shrubs less than 5 meters tall with shrub canopy typically greater than 20% of total vegetation. This class includes true shrubs, young trees in an early successional stage, or trees stunted from environmental conditions'],
        11:['Grassland or herbaceous'], # Grassland/herbaceous: 'areas dominated by gramanoid or herbaceous vegetation, generally greater than 80% of total vegetation. These areas are not subject to intensive management such as tilling, but can be utilized for grazing'],
        12:['Pasture or hay'], # Pasture/hay: 'areas of grasses, legumes, or grass-legume mixtures planted for livestock grazing or the production of seed or hay crops, typically on a perennial cycle. Pasture/hay vegetation accounts for greater than 20% of total vegetation'],
        13:['Cultivated crops'], # Cultivated crops: 'areas used for the production of annual crops, such as corn, soybeans, vegetables, tobacco, and cotton, and also perennial woody crops such as orchards and vineyards. Crop vegetation accounts for greater than 20% of total vegetation. This class also includes all land being actively tilled'],
        14:['Woody wetlands'], # Woody wetlands: 'areas where forest or shrubland vegetation accounts for greater than 20% of vegetative cover and the soil or substrate is periodically saturated with or covered with water'],
        15:['Emergent herbaceous wetlands'], # Emergent herbaceous wetlands: 'areas where perennial herbaceous vegetation accounts for greater than 80% of vegetative cover and the soil or substrate is periodically saturated with or covered with water'],
    },

    'nlcd-imp':{
        0: ['Roads', 'Transport', 'Transportation', 'Road network'],
        1: ['Non-road and non-energy impervious'],
        2: ['Energy impervious'],
    },


    # # # # # # # # test sets # # # # # # # #
    'multisenge': {
        0: ['Dense built-up areas'],
        1: ['Sparse built-up areas'],
        2: ['Specialized built-Up Areas'],
        3: ['Specialized but vegetative areas'],
        4: ['Large scale networks'],
        5: ['Arable lands'],
        6: ['Vineyards'],
        7: ['Orchards'],
        8: ['Grasslands'],
        9: ['Groces or hedges'],
        10:['Forests'],
        11:['Open spaces or mineral site'],
        12:['Wetlands'],
        13:['Water surfaces'],
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

    'osm':{
        0: ['Urban fabric'],
        1: ['Arable land'],
        2: ['Forest'],
        3: ['Industrial, commercial and transport'],
        4: ['Artificial, non-argricultural vegetation'],
        5: ['Mine, dump and construction sites'],
        6: ['Pastureland'],
        7: ['Permanent crops'],
        8: ['Waterbody'],
        9: ['Open spaces'],
        10:['Shrub and herbaceous vegetation associations'],
        11:['Wetlands'],
        12:['Coastal wetlands'],
    },

}