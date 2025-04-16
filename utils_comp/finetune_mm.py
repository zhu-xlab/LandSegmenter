# block names for unets
BLOCK_NAMES = ['encoder.layer1', 'encoder.layer2', 'encoder.layer3', 'encoder.layer4',
               'decoder.blocks.0', 'decoder.blocks.1', 'decoder.blocks.2', 'decoder.blocks.3', 'decoder.blocks.4',
               'segmentation_head']

# # # # for model loading # # # # 
def get_and_modify_module_state_dict_keys(checkpoint, prt_tag, multi_modal_ft=False):
    # read module state dict from checkpoint file  
    if multi_modal_ft:
        replaces = ['encoder1.', 'encoder2.']
        if 'mf' in prt_tag:
            msd = checkpoint['state_dict']
            prefixs = ['net.encoder1.', 'net.encoder.']
        elif 'lf' in prt_tag:
            msd = checkpoint['state_dict']
            prefixs = ['net.model1.encoder.', 'net.model2.encoder.']
        elif 'ns' in prt_tag:
            prefixs, msd = [], []
            for ck in checkpoint:
                msd.append(ck['state_dict'])
                prefixs.append('net.')
        elif prt_tag=='mocoq':
            prefixs, msd = [], []
            for ck in checkpoint:
                msd.append(ck['state_dict'])
                prefixs.append('module.encoder_q.')
        elif prt_tag=='mocok':
            prefixs, msd = [], []
            for ck in checkpoint:
                msd.append(ck['state_dict'])
                prefixs.append('module.encoder_k.')
        # modify module state dict keys for loading - replace prefix
        if type(msd)==list:
            msd_new = {}
            for i in range(len(msd)):
                for k in list(msd[i].keys()):
                    if k.startswith(prefixs[i]):
                        newk = k.replace(prefixs[i], replaces[i])
                        msd_new[newk] = msd[i][k]
                    del msd[i][k]
            msd = msd_new
        else:
            for k in list(msd.keys()):
                for i in range(len(prefixs)):
                    if k.startswith(prefixs[i]):
                        newk = k.replace(prefixs[i], replaces[i])
                        msd[newk] = msd[k]
                    del msd[k]
    else: 
        if prt_tag=='mocoq':
            msd = checkpoint['state_dict']
            prefix = 'module.encoder_q.'
            replace = 'encoder.'
        elif prt_tag=='mocok':
            msd = checkpoint['state_dict']
            prefix = 'module.encoder_k.'
            replace = 'encoder.'
        elif prt_tag=='dinot':
            msd = checkpoint['teacher']
            prefix = 'module.backbone.'
            replace = 'encoder.'
        elif prt_tag=='dinos':
            msd = checkpoint['student']
            prefix = 'module.backbone.' 
            replace = 'encoder.'
        elif 'mfs1' in prt_tag:
            msd = checkpoint['state_dict']
            prefix = 'net.encoder1.'
            replace = 'encoder.'
        elif 'mfs2' in prt_tag:
            msd = checkpoint['state_dict']
            prefix = 'net.encoder.'
            replace = 'encoder.'
        elif 'lfs1' in prt_tag:
            msd = checkpoint['state_dict']
            prefix = 'net.model1.encoder.'
            replace = 'encoder.'
        elif 'lfs2' in prt_tag:
            msd = checkpoint['state_dict']
            prefix = 'net.model2.encoder.'
            replace = 'encoder.'
        elif 'ns' in prt_tag:
            msd = checkpoint['state_dict']
            prefix = 'net.' 
            replace = ''
        # modify module state dict keys for loading - replace prefix
        for k in list(msd.keys()):
            if k.startswith(prefix):
                newk = k.replace(prefix, replace)
                msd[newk] = msd[k]
            del msd[k]
            
    return msd


def fetch_required_state_dict(msd, wload_tag):
    if wload_tag=='encoder':
        msd = {k: msd[k] for k in msd.keys() if 'encoder' in k}
    elif wload_tag=='wocls':
        msd = {k: msd[k] for k in msd.keys() if 'encoder' in k or 'decoder' in k}
    return msd


def remove_encoder_prefix(msd):
    msd = {k.replace('encoder.', ''): msd[k] for k in msd.keys() if 'encoder' in k}
    return msd


def check_missing_keys(missing_keys, wload_tag):
    check_items = ['encoder']
    if wload_tag=='all':
        assert len(missing_keys)==0, f'Loading ({wload_tag}) has missing keys: {missing_keys.keys()}'
        return
    elif wload_tag=='wocls':
        check_items += ['decoder']
    elif wload_tag=='all':
        check_items += ['decoder', 'segmentation_head']
    # check
    mks = []
    for ci in check_items:
        for mk in missing_keys:
            if ci in mk:
                mks.append(mk)
    assert len(mks)==0, f'Loading ({wload_tag}) has missing keys: {mks}'  
    return

def fix_all_model_weights(net):
    for _, param in net.named_parameters():
        param.requires_grad = False
    print('All weights are fixed!')
    return net


def fix_model_weights(net, eft_tag):
    if eft_tag in ['efix', 'eft']:
        # fix encoder weights
        for name, param in net.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False
        print('Encoder weights are fixed!')
    return net


def unfix_model_weights(net):
    # fix encoder weights
    for name, param in net.named_parameters():
        param.requires_grad = True
    return net


def reset_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return 
# # # # for model loading # # # # 


# set adaptive lrs for different layers
def set_adaptive_lr_for_all_layers(net, lr_base, lr_adaptive_type='des', lr_adaptive_rate=0.8):
    assert lr_adaptive_type in ['des', 'asc'], "Please provide correct lr adaptive type ('asc' or 'des')!"
    params = []
    lr = lr_base
    
    block_names = BLOCK_NAMES
    
    # get lr list for different layers
    nbs = len(block_names)
    lrs = [lr*lr_adaptive_rate**i for i in range(nbs+1)]
    if lr_adaptive_type == 'asc':
        lrs.reverse()
    
    # set lr
    for name, param in net.named_parameters():
        sign = True
        for bi, block_name in enumerate(block_names):
            if block_name in name:
                params.append({'params': param, 'lr': lrs[bi+1]})
                sign = False
                break
        if sign:
            params.append({'params': param, 'lr': lrs[0]})
    
    return params
                
        
def set_adaptive_lr_for_decoder_layers(net, lr_base, lr_adaptive_type='asc', lr_adaptive_rate=0.8):
    assert lr_adaptive_type in ['des', 'asc'], "Please provide correct lr adaptive type ('asc' or 'des')!"
    params = []
    lr = lr_base
    
    block_names = BLOCK_NAMES[-6:]
    
    # get lr list for different layers
    nbs = len(block_names)
    lrs = [lr*lr_adaptive_rate**i for i in range(nbs)]
    if lr_adaptive_type == 'asc':
        lrs.reverse()
    
    # set lr
    for name, param in net.named_parameters():
        if 'encoder' in name:
            # assign min lr to other layers (encoder)
            params.append({'params': param, 'lr': min(lrs)})
        else:
            for bi, block_name in enumerate(block_names):
                if block_name in name:
                    params.append({'params': param, 'lr': lrs[bi]})
                    indicator = False
                    break
        
    return params