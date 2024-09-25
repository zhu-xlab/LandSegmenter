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

    'openearthmap': {
        'data_path': '/p/project1/hai_ws4eo/liu_ch/data/WeakFT/exact/openearthmap_val.lmdb',
    },
}


class_names = {
    'dw': {
        0: ['Water', 'Open water', 'Aquatic surface', 'Waterbody', 'Lakes and rivers', 'Hydric zone', 'Aquatic zone', 'Marine and freshwater areas'],
        1: ['Trees', 'Forest', 'Woodland', 'Canopy', 'Timberland', 'Forested areas', 'Arboreal zone'],
        2: ['Grass', 'Grassland', 'Prairie', 'Savanna', 'Pastureland', 'Herbaceous cover'],
        3: ['Flooded vegetation', 'Wetland vegetation', 'Marshlands', 'Swamp plants', 'Riparian areas', 'Aquatic vegetation'], 
        4: ['Crops', 'Farmland', 'Agricultural fields', 'Cultivated land', 'Arable land', 'Cropland'],
        5: ['Shrub and Scrub', 'Shrubland', 'Heath', 'Brushland', 'Chaparral', 'Low vegetation'],
        6: ['Built area', 'Urban area', 'Developed land', 'Residential and commercial zones', 'Infrastructure', 'Built-up land'],
        7: ['Bare land', 'Barren terrain', 'Open soil', 'Unvegetated land', 'Exposed earth'],
        8: ['Ice and Snow', 'Glacial area', 'Snow cover', 'Ice field', 'Frozen surface'],
    },

    'dynamicearthnet': {
        0: ['impervious surface', 'Paved surface', 'Built-up areas', 'Sealed land', 'Non-permeable area', 'Urban pavement'],
        1: ['agriculture', 'Farmland', 'Cropland', 'Arable land', 'Cultivated area', 'Agricultural fields'],
        2: ['forest and other vegetation', 'Trees and grass', 'Woodland and grassland', 'Vegetated areas', 'Green cover', 'Arboreal and herbaceous zones'],
        3: ['Wetlands', 'Marshland', 'Swamp and Bog', 'Flooded vegetation', 'Peatland', 'Riparian area'],
        4: ['soil', 'Bare soil', 'Unvegetated land', 'Exposed earth', 'Barren terrain', 'Open ground'],
        5: ['water', 'Open water', 'Aquatic zone', 'Waterbody', 'Lakes and rivers', 'Hydric zone', 'Marine and freshwater areas'],
        6: ['snow and ice', 'Snow cover', 'Ice field', 'Frozen ground', 'Permafrost and Snowpack'],
    },

    'openearthmap': {
        0: ['Bareland', 'Bare soil', 'Barren land', 'Open ground', 'Unvegetated land', 'Exposed ground', 'Desolate area'],
        1: ['Rangeland', 'Pastureland', 'Grassland', 'Prairie', 'Grazing land', 'Herbaceous cover'],
        2: ['Developed space', 'Urban area', 'Built-up areas', 'Residential and commercial areas', 'Constructed zones'],
        3: ['Road', 'Highway amd Street and Path', 'Transportation areas', 'Road network'],
        4: ['Tree', 'Forest', 'Woodland', 'Canopy', 'Timberland', 'Forested areas', 'Arboreal zone'],
        5: ['Water', 'Open water', 'Aquatic area', 'Waterbody', 'Lakes and rivers', 'Hydrological features', 'Marine and freshwater areas'],
        6: ['Agriculture land', 'Farmland', 'Cropland', 'Arable land', 'Cultivated field', 'Agricultural plot', 'Crop zones'],
        7: ['Building', 'Built-up areas', 'Construction', 'Infrastructure', 'Dwelling and Commercial Facilities'],
    }
}