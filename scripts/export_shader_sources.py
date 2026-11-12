#!/usr/bin/env python3
"""Export retained NVN Metal sources for inspection; exports never become build inputs."""
import argparse
import concurrent.futures as cf
import os
from pathlib import Path
import sys
import tempfile
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dm/tools'))
import shader_lib


def export_sources(lib, out):
    out = Path(out).resolve()
    authoritative = Path(shader_lib.folder_of(lib)).resolve()
    if out == authoritative or authoritative in out.parents:
        raise ValueError('inspection output must be outside the authoritative shader library directory')
    with shader_lib.SourceReader(lib) as sources:
        names = sorted(sources.names())
        out.mkdir(parents=True, exist_ok=True)
        if not names: return 0
        workers = min(len(names), os.cpu_count() or 1)
        local, readers = threading.local(), []
        def export(name):
            if not hasattr(local, 'reader'):
                local.reader = shader_lib.SourceReader(lib); readers.append(local.reader)
            data = local.reader.read_bytes(name)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=out, prefix='.' + name + '.', suffix='.tmp', delete=False) as stream:
                    temporary = Path(stream.name); stream.write(data)
                os.replace(temporary, out / name)
            finally:
                if temporary is not None: temporary.unlink(missing_ok=True)
        try:
            with cf.ThreadPoolExecutor(workers) as pool:
                for start in range(0, len(names), 4 * workers):
                    for _ in pool.map(export, names[start:start + 4 * workers]): pass
        finally:
            for reader in readers: reader.close()
        return len(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library', help='the build library b-<chain>, with or without .img')
    parser.add_argument('--out', required=True, help='inspection output directory; same-named generated sources are replaced')
    args = parser.parse_args()
    try: count = export_sources(args.library, args.out)
    except (OSError, ValueError, shader_lib.SourceError) as error: parser.exit(1, '[RED] ' + str(error).removeprefix('[RED] ') + '\n')
    print(f'exported {count:,} Metal sources to {Path(args.out).resolve()}')


if __name__ == '__main__': main()
