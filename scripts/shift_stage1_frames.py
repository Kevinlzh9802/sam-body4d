#!/usr/bin/env python3
"""
Renumber the frames of a Stage 1 output made before stage1_masklet.py took frame numbers from the segment numbers:
a folder with only some segments (e.g. only cam13_seg011) was numbered from 0 instead of (11 - 1) x 600 = 6000.
Shifts every frame name and number by the offset of the first segment: images.zip / masks.zip entries,
id_mapping.json and masklets_meta.json (frame_start / frame_end), mask_bbox.json keys, mask_ids/*.jpg.
Only the standard library, so it runs on a login node. Refuses a folder whose first frame already matches.

python scripts/shift_stage1_frames.py <Stage 1 dir> [--seg-frames 600] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import zipfile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage1_dir')
    ap.add_argument('--seg-frames', type=int, default=600)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    d = args.stage1_dir
    with open(os.path.join(d, 'id_mapping.json')) as fh:
        idm = json.load(fh)
    first = idm['segments'][0]
    n = int(re.search(r'_seg(\d+)$', first['segment_key']).group(1))
    offset = (n - 1) * args.seg_frames - int(first['frame_start'])
    if offset == 0:
        print(f'{d}: already numbered from {first["frame_start"]} ({first["segment_key"]}); nothing to do')
        return
    print(f'{d}: {first["segment_key"]} starts at {first["frame_start"]}, shifting all frames by {offset:+d}')
    if args.dry_run:
        return
    rename = lambda name: re.sub(r'^(\d{8})', lambda m: f'{int(m.group(1)) + offset:08d}', name)  # noqa: E731

    for kind in ('images', 'masks'):
        src = os.path.join(d, f'{kind}.zip')
        if not os.path.exists(src):
            continue
        tmp = src + '.shifting'
        with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_STORED) as zout:
            for info in zin.infolist():
                base = os.path.basename(info.filename)
                new = os.path.join(os.path.dirname(info.filename), rename(base)) if base else info.filename
                zout.writestr(zipfile.ZipInfo(new, date_time=info.date_time), zin.read(info.filename))
        os.replace(tmp, src)
        print(f'  {kind}.zip renamed')

    for seg in idm['segments']:
        seg['frame_start'] = int(seg['frame_start']) + offset
        seg['frame_end'] = int(seg['frame_end']) + offset
    with open(os.path.join(d, 'id_mapping.json'), 'w') as fh:
        json.dump(idm, fh, indent=2)
    mp = os.path.join(d, 'masklets_meta.json')
    if os.path.exists(mp):
        with open(mp) as fh:
            meta = json.load(fh)
        meta['segment_id_mappings'] = idm['segments']
        meta['frame_offset_shift'] = offset
        with open(mp, 'w') as fh:
            json.dump(meta, fh, indent=2)
    bp = os.path.join(d, 'mask_bbox.json')
    if os.path.exists(bp):
        with open(bp) as fh:
            bb = json.load(fh)
        bb['annotations'] = {str(int(k) + offset): v for k, v in bb['annotations'].items()}
        with open(bp, 'w') as fh:
            json.dump(bb, fh)
    vd = os.path.join(d, 'mask_ids')
    if os.path.isdir(vd):
        for fn in sorted(os.listdir(vd), reverse=offset > 0):
            if re.match(r'^\d{8}', fn):
                shutil.move(os.path.join(vd, fn), os.path.join(vd, rename(fn)))
    print(f'  json files and mask_ids/ shifted by {offset:+d}')


if __name__ == '__main__':
    main()
