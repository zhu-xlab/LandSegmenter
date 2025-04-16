import os
import torch


# block names for unets
BLOCK_NAMES = ['encoder.layer1', 'encoder.layer2', 'encoder.layer3', 'encoder.layer4',
               'decoder.blocks.0', 'decoder.blocks.1', 'decoder.blocks.2', 'decoder.blocks.3', 'decoder.blocks.4',
               'segmentation_head']


# # # # for model loading # # # # 
def get_model_state_dict_from_saved_models(net, wpath, wload_tag, prt_tag):
    assert os.path.isfile(wpath), 'Given model checkpoint loading path is invalid!'
    checkpoint = torch.load(wpath, map_location=torch.device('cpu'))
    load_dec = True if wload_tag in ['wocls','all', 'de1', 'de2', 'de3', 'de4'] else False
    n_load_decb = int(wload_tag[-1]) if wload_tag.startswith('de') else 5
    msd = get_and_modify_module_state_dict_keys(checkpoint, prt_tag,
                                                load_decoder=load_dec,
                                                load_seghead=True if wload_tag=='all' else False,
                                                n_load_decoder_blocks=n_load_decb)
    if 'ns' in prt_tag: msd = fetch_required_state_dict(msd, wload_tag)   
    missing_keys, _ = net.load_state_dict(msd, strict=False)
    check_missing_keys(missing_keys, wload_tag)
    return net


def modify_msd_keys(msd, prefix, replace):
    for k in list(msd.keys()):
        if k.startswith(prefix):
            newk = k.replace(prefix, replace)
            if not k.startswith(f'{replace}fc'):
                msd[newk] = msd[k]
        del msd[k]
    return msd

def get_and_modify_module_state_dict_keys(checkpoint, prt_tag, load_decoder=False, load_seghead=False, n_load_decoder_blocks=5):
    # read module state dict from checkpoint file   
    if prt_tag=='mocoq':
        msd = checkpoint['state_dict'].copy()
        prefix = 'module.encoder_q.'
        replace = 'encoder.'
    elif prt_tag=='mocok':
        msd = checkpoint['state_dict'].copy()
        prefix = 'module.encoder_k.'
        replace = 'encoder.'
    elif prt_tag=='dinot':
        msd = checkpoint['teacher'].copy()
        prefix = 'module.backbone.'
        replace = 'encoder.'
    elif prt_tag=='dinos':
        msd = checkpoint['student'].copy()
        prefix = 'module.backbone.' 
        replace = 'encoder.'
    elif prt_tag=='satlas':
        msd = checkpoint.copy()
        prefix = 'backbone.resnet.'
        replace = 'encoder.'
    elif 'decur' in prt_tag:
        modality = int(prt_tag[-1])
        msd = checkpoint.copy()
        prefix = f'module.backbone_{modality}.'
        replace = 'encoder.'
    elif 'mfs1' in prt_tag:
        msd = checkpoint['state_dict'].copy()
        prefix = 'net.encoder1.'
        replace = 'encoder.'
    elif 'mfs2' in prt_tag:
        msd = checkpoint['state_dict'].copy()
        prefix = 'net.encoder.'
        replace = 'encoder.'
    elif 'lfs1' in prt_tag:
        msd = checkpoint['state_dict'].copy()
        prefix = 'net.model1.encoder.'
        replace = 'encoder.'
    elif 'lfs2' in prt_tag:
        msd = checkpoint['state_dict'].copy()
        prefix = 'net.model2.encoder.'
        replace = 'encoder.'
    elif 'ns' in prt_tag:
        msd = checkpoint['state_dict'].copy()
        prefix = 'net.encoder.' 
        replace = 'encoder.'
    elif prt_tag=='scalemae' or 'satmae' in prt_tag:
        s2_enc = checkpoint['model']
        msd = s2_enc.copy()
        for k in s2_enc.keys():
            if k.startswith(('blocks', 'cls', 'pos', 'patch', 'channel')):
                msd['encoder.'+k] = msd[k] 
            msd.pop(k)
    elif 'dofa' in prt_tag:
        msd = checkpoint.copy()
        for k in checkpoint.keys():
            if k.startswith(('blocks', 'cls', 'pos', 'patch', 'channel')):
                msd['encoder.'+k] = msd[k] 
            msd.pop(k)
    # modify module state dict keys for loading - replace prefix
    if ('scalemae' not in prt_tag) and ('satmae' not in prt_tag) and ('dofa' not in prt_tag):
        msd = modify_msd_keys(msd, prefix, replace)
    # # for decoder and seg head parts
    if load_decoder:
        msd2 = checkpoint['state_dict'].copy()
        replace = 'decoder.'
        if 'mfs1' in prt_tag:
            prefix = 'net.decoder.'
        elif 'mfs2' in prt_tag:
            prefix = 'net.decoder.'
        elif 'lfs1' in prt_tag:
            prefix = 'net.model1.decoder.'
        elif 'lfs2' in prt_tag:
            prefix = 'net.model2.decoder.'
        else:
            if 'ns' not in prt_tag:
                raise ValueError(f'Cannot load decoder for prt_tag={prt_tag}!')
            prefix = 'net.decoder.'
        # get specific decoder weights
        if n_load_decoder_blocks==5:
            msd2 = modify_msd_keys(msd2, prefix, replace)
            msd.update(msd2)
        else:
            for i in range(n_load_decoder_blocks):
                msd2_ = modify_msd_keys(msd2.copy(), f'{prefix}blocks.{i}', f'{replace}blocks.{i}')
                msd.update(msd2_)
        # # # # # # FOR DEBUGGING # # # # #
        # print(f'Loaded decoder with {n_load_decoder_blocks} blocks: {msd.keys()}')
    # # for seg head
    if load_seghead:
        msd3 = checkpoint['state_dict'].copy()
        if 'mfs1' in prt_tag:
            prefix = 'net.segmentation_head.'
            replace = 'segmentation_head.'
        elif 'mfs2' in prt_tag:
            prefix = 'net.segmentation_head.'
            replace = 'segmentation_head.'
        elif 'lfs1' in prt_tag:
            prefix = 'net.model1.segmentation_head.'
            replace = 'segmentation_head.'
        elif 'lfs2' in prt_tag:
            prefix = 'net.model2.segmentation_head.'
            replace = 'segmentation_head.'
        else:
            if 'ns' not in prt_tag:
                raise ValueError(f'Cannot load segmentation_head for prt_tag={prt_tag}!')
        msd3 = modify_msd_keys(msd3, prefix, replace)
        # merge msd
        msd.update(msd3)
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
        assert len(missing_keys)==0, f'Loading ({wload_tag}) has missing keys: {missing_keys}'
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
    if mks != ['encoder.norm.weight', 'encoder.norm.bias']:
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