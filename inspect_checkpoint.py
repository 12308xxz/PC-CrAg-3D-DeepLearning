"""Inspect a trusted author-generated study checkpoint without exposing local paths.

Only open checkpoints created by yourself or a trusted collaborator. Legacy
PyTorch checkpoint dictionaries may require weights_only=False (uses pickle).
"""
import argparse
import torch
from models import create_model_3d_baseline, create_model_3d_optimized, create_model_3d_clinical


def inspect(path, model_name, expected_threshold=None):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(obj, dict) or 'model_state' not in obj:
        raise ValueError('Expected training checkpoint dictionary with model_state')
    cfg = obj.get('config', {})
    print('Architecture:',model_name)
    print('Saved seed:',cfg.get('seed','not saved'))
    print('Saved external split:',cfg.get('external_split','not saved'))
    print('Saved epoch:',obj.get('epoch','not saved'))
    print('Saved threshold:',obj.get('threshold','not saved'))
    metrics = obj.get('metrics') or {}
    print('Saved development AUC:',metrics.get('auc','not saved'))
    if expected_threshold is not None:
        if obj.get('threshold') is None or abs(float(obj['threshold'])-expected_threshold)>1e-8:
            raise ValueError(f'Saved threshold != expected {expected_threshold}')
        print('PASS expected saved threshold')
    if model_name == 'baseline':
        model=create_model_3d_baseline(pretrained=False)
    elif model_name == 'optimized':
        model=create_model_3d_optimized(pretrained=False)
    else:
        model=create_model_3d_clinical(clinical_dim=cfg.get('clinical_dim',8),pretrained=False)
    model.load_state_dict(obj['model_state'],strict=True)
    print('PASS strict model_state loading (all parameter names and shapes match)')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Check a trusted final author checkpoint')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--model',choices=['baseline','optimized','fusion'],required=True)
    parser.add_argument('--expected-threshold',type=float)
    args=parser.parse_args()
    inspect(args.checkpoint,args.model,args.expected_threshold)
