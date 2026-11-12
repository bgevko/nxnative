#!/usr/bin/env python3
"""Replace an existing shader image's inline .metal payloads with retained CAS references."""
import argparse
import contextlib
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dm/tools'))
import romfs, shader_cache, shader_lib


def optimize_image(image, cas):
    image = Path(image).expanduser().resolve()
    if image.suffix != '.img' or not image.is_file(): raise ValueError('an existing .img file is required')
    if Path(shader_lib.folder_of(image)).exists(): raise ValueError('a loose shader output folder exists beside the image; finish the build before optimizing')
    romfs.Image(image)
    paths = list(romfs.Path(image).iterdir())
    if any(not p.is_file() for p in paths): raise ValueError('a shader image must contain only files')
    files = {p.name: str(p) for p in paths}
    count = sum(n.endswith('.metal') for n in files)
    if shader_lib.SOURCES in files:
        with shader_lib.SourceReader(image) as reader: reader.validate()
        return 0, 0
    if not count: return 0, 0
    with shader_lib.SourceReader(image) as reader: reader.validate()
    cas = Path(cas).expanduser().resolve()
    if cas in (image, Path(str(image) + '.part').resolve()): raise ValueError('CAS must be separate from the image and its temporary write destination')
    if not cas.is_file(): raise ValueError('an existing CAS database is required')
    before = image.stat().st_size
    with contextlib.closing(shader_cache.Store(cas)) as store:
        shader_lib.write_library(files, image, store)
    return count, before - image.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', help='completed .img file to rewrite in place')
    parser.add_argument('--cas', required=True, help='existing shader/cache/cas.db to retain the sources in')
    args = parser.parse_args()
    try: count, saved = optimize_image(args.image, args.cas)
    except (OSError, ValueError, sqlite3.Error, shader_lib.SourceError) as error:
        parser.exit(1, '[RED] ' + str(error).removeprefix('[RED] ') + '\n')
    if count: print(f'retained {count:,} sources in CAS; image size change: {-saved:+,} bytes')
    else: print('no inline sources; image unchanged')


if __name__ == '__main__': main()
