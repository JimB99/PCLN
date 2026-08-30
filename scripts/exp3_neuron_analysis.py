"""Exp3 neuron specialization analysis

Loads a model checkpoint and inspects parameters related to dynamic neurons
by searching for parameter names containing 'neuron' or 'gate'. For each
candidate tensor, it computes per-neuron norms (if possible), plots a bar
chart, and writes a short markdown report with findings.

Usage:
    python scripts/exp3_neuron_analysis.py --checkpoint results/exp3_dynamic_neurons/best_model.pt

"""
import argparse
import os
import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_state_dict(path):
    try:
        # Prefer loading full checkpoint (weights_only=False) to support older-style checkpoints
        ck = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        # Older torch versions may not support the weights_only arg
        ck = torch.load(path, map_location='cpu')
    except Exception:
        # If torch refuses due to safe globals restrictions, try using the safe_globals context
        try:
            import argparse as _argparse
            with torch.serialization.safe_globals([_argparse.Namespace]):
                ck = torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            # Re-raise the original error if fallback fails
            raise
    if isinstance(ck, dict):
        for k in ('state_dict', 'model_state_dict', 'model', 'model_state'):
            if k in ck and isinstance(ck[k], dict):
                return ck[k]
        # Otherwise assume ck itself is a state dict
        return ck
    else:
        return ck


def infer_num_neurons(state_dict, default: int = 256) -> int:
    """Infer neuron count from gate output layers in the checkpoint."""
    for name, tensor in state_dict.items():
        if not hasattr(tensor, 'shape'):
            continue
        if name.endswith('.gate.2.weight') and len(tensor.shape) == 2:
            return int(tensor.shape[0])
    return default


def find_candidate_tensors(state_dict):
    candidates = {}
    for name, tensor in state_dict.items():
        if not hasattr(tensor, 'shape'):
            continue
        lname = name.lower()
        if 'neuron' in lname or 'gate' in lname or 'gen_neurons' in lname:
            candidates[name] = tensor.numpy()
    return candidates


def analyze_tensor(name, arr, out_dir, default_num_neurons=256):
    # Try to determine num_neurons
    shape = arr.shape
    num_neurons = None
    if default_num_neurons in shape:
        num_neurons = default_num_neurons
    else:
        # pick largest dimension under 2048
        dims = [d for d in shape if d <= 2048]
        if dims:
            num_neurons = max(dims)
    if num_neurons is None:
        return None

    # Try to compute per-neuron norms
    per_neuron = None
    if shape[0] == num_neurons:
        # rows correspond to neurons
        per_neuron = np.linalg.norm(arr.reshape(num_neurons, -1), axis=1)
    elif shape[-1] == num_neurons:
        per_neuron = np.linalg.norm(arr.reshape(-1, num_neurons), axis=0)
    elif arr.size % num_neurons == 0:
        per_neuron = np.linalg.norm(arr.reshape(num_neurons, -1), axis=1)
    else:
        # fallback: compute global stats only
        per_neuron = None

    report = {}
    report['name'] = name
    report['shape'] = shape
    if per_neuron is not None:
        # normalize
        norm = per_neuron / (per_neuron.sum() + 1e-12)
        report['per_neuron_norm'] = norm.tolist()
        # top neurons
        topk = int(min(20, len(norm)))
        top_idx = np.argsort(-norm)[:topk]
        report['top_neurons'] = [(int(i), float(norm[i])) for i in top_idx]

        # plot
        plt.figure(figsize=(10,3))
        plt.bar(np.arange(len(norm)), norm, width=0.7)
        plt.xlabel('Neuron index')
        plt.ylabel('Normalized norm')
        plt.title(f'Per-neuron normalized norm — {name}')
        plt.tight_layout()
        fname = os.path.join(out_dir, name.replace('/', '_') + '.png')
        plt.savefig(fname)
        plt.close()
        report['plot'] = fname
    else:
        # global stats
        report['mean'] = float(arr.mean())
        report['std'] = float(arr.std())

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='results/exp3_dynamic_neurons/best_model.pt')
    parser.add_argument('--outdir', type=str, default='results/exp3_analysis')
    parser.add_argument('--num_neurons', type=int, default=None,
                        help='Neuron count (auto-detected from checkpoint if omitted)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print('Loading checkpoint:', args.checkpoint)
    state = load_state_dict(args.checkpoint)
    print('Loaded, {} tensors'.format(len(state)))

    num_neurons = args.num_neurons or infer_num_neurons(state)
    print('Using num_neurons={}'.format(num_neurons))

    candidates = find_candidate_tensors(state)
    print('Found {} candidate tensors'.format(len(candidates)))

    reports = []
    for name, arr in candidates.items():
        try:
            report = analyze_tensor(name, arr, args.outdir, default_num_neurons=num_neurons)
            if report is not None:
                reports.append(report)
        except Exception as e:
            print('Skipping', name, 'error:', e)

    # Write markdown report
    md_lines = []
    md_lines.append('# exp3 neuron specialization analysis')
    md_lines.append('')
    md_lines.append(f'Checkpoint: `{args.checkpoint}`')
    md_lines.append('')
    if not reports:
        md_lines.append('No candidate tensors matching "neuron" or "gate" were found.')
    else:
        for r in reports:
            md_lines.append('## ' + r['name'])
            md_lines.append(f'- shape: `{r["shape"]}`')
            if 'per_neuron_norm' in r:
                md_lines.append(f'- plot: ![]({os.path.basename(r["plot"])})')
                md_lines.append('- top neurons (index, normalized_norm):')
                for idx, v in r['top_neurons']:
                    md_lines.append(f'  - {idx}: {v:.4f}')
            else:
                md_lines.append(f'- mean: {r.get("mean", 0):.6f}, std: {r.get("std",0):.6f}')
            md_lines.append('')

    report_path = os.path.join(args.outdir, 'REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    print('Wrote report to', report_path)

if __name__ == '__main__':
    main()
