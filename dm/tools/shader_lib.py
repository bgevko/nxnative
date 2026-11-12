#!/usr/bin/env python3
# A game's NVN shader libraries as one file: b-<chain>.img beside the folder b-<chain>/ that dmsh_build.py writes. One file per shader variant is what made a build folder slow to copy or
# delete (Skyward Sword HD: 12,064 files for 2,470 shaders; 28,669 shaders in Tears of the Kingdom), and each regress.py chain build held another set. Nothing is unpacked to use it: the host
# (dm/gfx/nvn/msl_image.h) and the tools (romfs.py) read a shader in place through the index.
# Source payloads are retained in CAS; _b_sources.json maps their original names to exact image-owned entries. SourceReader is the source-only tool boundary; apps omit that metadata.
# The format is the RomFS image's (romfs_image.py), one flat folder, with each file's data aligned to ALIGN bytes: Metal is handed a pointer into the mapped file.
# A folder and an image may lie side by side, and each file comes from the folder first:
#   folder with a stamp (inputs.tsv)      a finished full build: the image, if any, is older and pack replaces it
#   folder without a stamp, and an image  a partial dmsh_build.py run (given keys) over a packed library, or a stopped build: the folder holds only what it rebuilt, and pack keeps the image's other files
#   folder alone                          a build folder from before this, a partial build, or a build in progress: used as it is
# pack writes the image, then removes the folder, so every file is in one or the other at each moment, and a host running from the folder goes on from the image. A pack that fails
# (a full drive) is one [RED] line, not an error: the build is done, and the folder works.
# An iPhone library may leave out the pulling vertex libraries when nothing will read them (pack's leave_out), which the marker PULLING_LEFT_OUT inside it says: the build that needs them compiles them,
# and pack_ipa.py refuses to pack such an image for an app that needs them, which would fail only on the device.
# Not part of the translation chain: it decides no output (dmsh_chain.py). dmsh_build.py writes the image itself (write, from the outputs it staged in the shader cache), and its folder holds
# only the shaders in flight; pack is for a folder an older chain's dmsh_build.py wrote (regress.py builds old revisions) or a build stopped before that.
# Usage: shader_lib.py pack <lib>                      the folder into the image
#        shader_lib.py expand <lib>                    the image back into a folder, only for a host from before this change
#        shader_lib.py path <lib>                      what to read: the folder or the image
#        shader_lib.py has <lib>                       exit status 0 when it holds a shader's library (<key>_b.metallib)
#        shader_lib.py cat <lib> <name>                one file's bytes on standard output
#        shader_lib.py subset <lib> <out.img> [--skip-suffix S]...  a new image without .metal sources and build reports, as an app carries it
# <lib> is the path b-<chain>, with or without .img.
import argparse, builtins, contextlib, fcntl, hashlib, json, os, re, shutil, sqlite3, stat, sys, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import romfs, shader_cache

ALIGN = 16   # of each file's data: the file is mapped by pages, so its first byte's alignment is its offset's in the image
STAMP = 'inputs.tsv'   # dmsh_chain.py STAMP
APP_SUFFIXES = ('.metallib', '.json', '.tsv')   # what the host reads (not the Mac build's .metal sources)
SOURCES = '_b_sources.json'
APP_LEAVE_OUT = ('_b_build.json', SOURCES)   # build-only reports and source references
PULLING, PULLING_LEFT_OUT = '_bp.metallib', '_bp_left_out'   # dmsh_pull.py's libraries (build_ipa.sh decides who wants them), and the marker of an image without them


def folder_of(lib):
    s = os.fspath(lib)
    return s[:-4] if s.endswith('.img') else s


def image_of(lib): return folder_of(lib) + '.img'
def name_of(lib): return os.path.basename(os.path.normpath(folder_of(lib)))   # b-<chain>, as the headers of c1_full.tsv and cb_extent.tsv name it


def view(lib):
    """lib as a romfs.Path to read from (the state table at the top of this file). A path that does not exist gives the folder's, whose glob is empty and whose reads fail as they did"""
    folder, image = folder_of(lib), image_of(lib)
    if os.path.isfile(image) and not os.path.isfile(os.path.join(folder, STAMP)): return romfs.Path(image)
    return romfs.Path(folder)


class SourceError(RuntimeError):
    """Required source storage failed; analysis must not treat this as an ordinary missing shader."""


@contextlib.contextmanager
def _source_lock(image, exclusive=False):
    image = os.path.abspath(os.fspath(image))
    path = os.path.join(os.path.dirname(image), '.' + os.path.basename(image) + '.sources.lock')
    with builtins.open(path, 'a+b' if exclusive else 'rb') as lock:
        try: fcntl.flock(lock, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as e: raise SourceError(f'[RED] shader sources are in use: {image}; retry after the other source operation finishes') from e
        yield


def _source_manifest(root):
    def unique(pairs):
        value = {}
        for key, data in pairs:
            if key in value: raise ValueError(f'duplicate source manifest key {key}')
            value[key] = data
        return value
    value = json.loads((root / SOURCES).read_bytes(), object_pairs_hook=unique)
    if not isinstance(value, dict) or type(value.get('version')) is not int or value['version'] != 1: raise ValueError('unsupported source manifest version')
    generation, cas, sources = value.get('generation'), value.get('cas'), value.get('sources')
    if not isinstance(generation, str) or not re.fullmatch('[0-9a-f]{32}', generation): raise ValueError('invalid source generation')
    if not isinstance(cas, str) or not cas or '\0' in cas or os.path.isabs(cas): raise ValueError('source CAS path must be relative')
    if not isinstance(sources, dict): raise ValueError('invalid source names')
    for name, digest in sources.items():
        if not name.endswith('.metal') or name.startswith('.') or any(c in name for c in ('/', '\\', '\0')) or '.tmp' in name: raise ValueError(f'invalid source name {name!r}')
        name.encode('utf-8')
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{32}', digest): raise ValueError(f'invalid source hash for {name}')
    return value


def _source_data(source):
    if callable(source): source = source()
    if isinstance(source, bytes): return source
    if source is None: raise SourceError('[RED] no shader source bytes to retain')
    if not isinstance(source, (str, os.PathLike)): raise TypeError('shader source must be bytes, a path, or a callable returning them')
    return romfs.Path(source).read_bytes()


def _drop_sources(store, generation):
    with store.batch():
        store.delete_prefix(f'image/{generation}/')
        store.delete_prefix(f'image-owner/{generation}')


def _sync_dir(path):
    path = os.path.abspath(path)
    while True:
        try: fd = os.open(path, os.O_RDONLY); break
        except FileNotFoundError:
            parent = os.path.dirname(path)
            if parent == path: raise
            path = parent
    try: os.fsync(fd)
    finally: os.close(fd)


def prune_sources(store):
    """Reclaim retired/interrupted generations; a busy or unreadable image never proves retirement."""
    removed = 0
    for owner in store.names('image-owner/'):
        generation = owner.removeprefix('image-owner/')
        try:
            if not re.fullmatch('[0-9a-f]{32}', generation): raise ValueError('invalid owner generation')
            record = json.loads(store.get(owner) or b'null')
            relative = record['image']
            if not isinstance(relative, str) or os.path.isabs(relative): raise ValueError('invalid owner path')
            image = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(store.path)), relative))
            guard = _source_lock(image, True) if os.path.isdir(os.path.dirname(image)) else contextlib.nullcontext()
            with guard:
                try: installed = os.stat(image)
                except FileNotFoundError: installed = None
                if installed is not None:
                    if not stat.S_ISREG(installed.st_mode): raise ValueError('shader image is not a regular file')
                    romfs.Image(image)   # a failed generic Path lookup is not proof that source metadata is absent
                    root = romfs.Path(image)
                    if (root / SOURCES).is_file() and _source_manifest(root)['generation'] == generation: continue
                    with builtins.open(image, 'rb') as stream: os.fsync(stream.fileno())
                _sync_dir(os.path.dirname(image))
                _drop_sources(store, generation); removed += 1
        except SourceError: continue   # a publisher/reader owns the image
        except (OSError, ValueError, TypeError, KeyError) as e:
            store.say(f'[RED] cannot verify retained shader owner {owner}: {e}; its sources stay')
    return removed


def write_library(files, dst, store, *, source_entries=None):
    """Pin sources before atomically publishing one image of their references and runtime payloads."""
    dst = os.path.abspath(os.fspath(dst))
    source_entries = source_entries or {}
    sources = {n: p for n, p in files.items() if n.endswith('.metal')}
    kept = {n: p for n, p in files.items() if not n.endswith('.metal') and n != SOURCES}
    with _source_lock(dst, True):
        old = romfs.Path(dst)
        before = _source_manifest(old) if (old / SOURCES).is_file() else None
        if sources:
            generation, hashes = uuid.uuid4().hex, {}
            with store.batch(durable=True):
                owner = json.dumps({'image': os.path.relpath(os.path.realpath(dst), os.path.dirname(store.path))}).encode()
                if store.put(f'image-owner/{generation}', owner) is None: raise SourceError('[RED] cannot retain shader source ownership; image not published')
                for name, source in sources.items():
                    if name.startswith('.') or any(c in name for c in ('/', '\\', '\0')) or '.tmp' in name: raise SourceError(f'[RED] invalid shader source name {name!r}')
                    entry, original, expected = f'image/{generation}/{name}', source_entries.get(name), None
                    if original is None and before and isinstance(source, (str, os.PathLike)) and os.path.abspath(os.fspath(source)) == os.path.join(dst, name):
                        cas = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(dst)), before['cas']))
                        original = f'image/{before["generation"]}/{name}'
                        expected = before['sources'][name]
                        if os.path.abspath(cas) != os.path.abspath(store.path):
                            with contextlib.closing(shader_cache.Store(cas, readonly=True)) as previous:
                                source = previous.get(original)
                            if source is None or hashlib.blake2b(source, digest_size=16).hexdigest() != before['sources'][name]: raise SourceError(f'[RED] cannot retain previous source {name}')
                            original = None
                    digest = store.link(original, entry) if original else store.put(entry, _source_data(source))
                    if digest is None: raise SourceError(f'[RED] cannot retain shader source {name}; image not published')
                    if expected is not None and digest != expected: raise SourceError(f'[RED] retained source identity changed for {name}; image not published')
                    hashes[name] = digest
            kept[SOURCES] = json.dumps({'version': 1, 'cas': os.path.relpath(store.path, os.path.dirname(os.path.realpath(dst))), 'generation': generation, 'sources': hashes}, sort_keys=True).encode()
        write(kept, dst, durable=before is not None)
    prune_sources(store)


def require_sources(lib):
    if not (view(lib) / SOURCES).is_file(): raise SourceError(f'[RED] shader sources in {lib} use an older output format; rebuild the whole game')
    with SourceReader(lib) as reader:
        reader.validate()


class SourceReader:
    def __init__(self, lib):
        self.root, self._store, self._guard, self._manifest = view(lib), None, None, None
        try:
            if not self.root.is_dir(): raise FileNotFoundError(f'no readable shader library at {lib}')
            if (self.root / SOURCES).is_file():
                self._guard = _source_lock(image_of(lib)); self._guard.__enter__()
                self.root = view(lib)
                self._manifest = _source_manifest(self.root)
                cas = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(image_of(lib))), self._manifest['cas']))
                self._store = shader_cache.Store(cas, readonly=True)
            self._names = set(self._manifest['sources']) if self._manifest is not None else {p.name for p in self.root.glob('*.metal') if not p.name.startswith('.') and '.tmp' not in p.name}
        except (OSError, ValueError, sqlite3.Error) as e:
            self.close()
            raise SourceError(f'[RED] cannot resolve shader sources for {lib}: {e}; rebuild the whole game') from e
        except RuntimeError as e:
            self.close()
            if isinstance(e, SourceError): raise
            raise SourceError(f'[RED] cannot resolve shader sources for {lib}: {str(e).removeprefix("[RED] ")}') from e
        except BaseException:
            self.close(); raise

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def close(self):
        if self._store is not None: self._store.close(); self._store = None
        if self._guard is not None: self._guard.__exit__(None, None, None); self._guard = None

    def names(self):
        return set(self._names)

    def validate(self):
        """Require complete analysis inputs without decoding every source on a reused build."""
        needed = {p.name[:-len('.metallib')] + '.metal' for p in self.root.glob('*_b.metallib')}
        if needed - self._names: raise SourceError(f'[RED] shader source inputs are missing for {self.root}; use the completed build source library or rebuild the whole game')
        if self._manifest is not None:
            assert self._store is not None
            prefix = f'image/{self._manifest["generation"]}/'
            expected = {prefix + n: digest for n, digest in self._manifest['sources'].items()}
            try: hashes = self._store.hashes(prefix)
            except (OSError, sqlite3.Error, RuntimeError) as e:
                raise SourceError(f'[RED] cannot validate shader sources for {self.root}: {str(e).removeprefix("[RED] ")}') from e
            if hashes != expected: raise SourceError(f'[RED] retained source blobs are missing or changed for {self.root}; rebuild the whole game')

    def read_bytes(self, name):
        if name not in self._names: raise FileNotFoundError(f'{self.root}/{name}')
        if self._manifest is None: return (self.root / name).read_bytes()
        try:
            assert self._store is not None
            data = self._store.get(f'image/{self._manifest["generation"]}/{name}')
            if data is None or hashlib.blake2b(data, digest_size=16).hexdigest() != self._manifest['sources'][name]:
                raise ValueError('missing source blob or content hash mismatch')
            return data
        except (OSError, ValueError, sqlite3.Error) as e:
            raise SourceError(f'[RED] cannot read shader source {self.root}/{name}: {e}; rebuild the whole game') from e
        except RuntimeError as e:
            raise SourceError(f'[RED] cannot read shader source {self.root}/{name}: {str(e).removeprefix("[RED] ")}') from e


def exists(lib): return os.path.isdir(folder_of(lib)) or os.path.isfile(image_of(lib))


def read_stamp(lib):   # dmsh_chain.py read_stamp, on a folder or an image
    try: return dict(l.split('\t', 1) for l in (view(lib) / STAMP).read_text().splitlines() if '\t' in l and not l.startswith('#'))
    except OSError: return {}


def table_head(lib):   # dmsh_chain.py table_head, on a folder or an image: the name is the folder's either way
    return f'# chain {name_of(lib)} inputs {read_stamp(lib).get("inputs", "none")}'


def _names(folder):
    """{name: (size, mtime)} of the files in a library folder. A file name starting with a dot is detritus (an AppleDouble ._x on an exFAT or FAT drive, .DS_Store), and so is what a stopped dmsh_build.py left (<name>.tmp-<pid>-<thread>)"""
    out = {}
    with os.scandir(folder) as it:
        for e in it:
            if e.name.startswith('.') or '.tmp' in e.name: continue
            if e.is_dir(follow_symlinks=False): raise ValueError(f'{e.path} is a folder: a shader library holds only files')
            st = e.stat(); out[e.name] = (st.st_size, st.st_mtime_ns)
    return out


def write(files, dst, *, durable=False):
    """files: {name: path to read (a romfs path may name a file inside an image), the bytes, or a function returning them (read only when its file is written: dmsh_build.py's outputs staged in the
    shader cache, so that no library is held in memory whole)} → the image dst, by way of dst.part. A stopped write leaves only the .part, and an image is whole or absent"""
    durable = durable or SOURCES in files
    part = dst + '.part'
    names = sorted((n.encode() for n in files))
    ents, off = [], romfs.HEAD.size
    with builtins.open(part, 'wb') as out:
        out.write(b'\0' * off)
        for name in names:
            off += -off % ALIGN
            out.seek(off)
            n, source = 0, files[name.decode()]
            if callable(source): source = source()
            if source is None: raise OSError(f'{name.decode()}: no bytes to write')
            if isinstance(source, bytes): out.write(source); n = len(source)
            elif not isinstance(source, (str, os.PathLike)): raise TypeError(f'{name.decode()}: {type(source).__name__} is not a path, bytes or a function')
            else:
                with romfs.open(source, 'rb') as f:
                    while b := f.read(8 << 20): out.write(b); n += len(b)
            ents.append((name, off, n)); off += n
        out.seek(off)
        pool, idx = bytearray(), bytearray(romfs.ENT.pack(1, len(names), 0, 0, 0, 0))   # entry 0, the root: its children are every other entry
        for name, a, b in ents:
            idx += romfs.ENT.pack(a, b, len(pool), len(name), 1, 0); pool += name
        out.write(idx); out.write(pool)
        out.seek(0); out.write(romfs.HEAD.pack(romfs.MAGIC, off, len(idx) + len(pool), len(names) + 1))
        if durable: out.flush(); os.fsync(out.fileno())
    romfs.Image(part)   # a read of it, as the host's check: nothing is published that a reader refuses
    os.replace(part, dst)
    if durable: _sync_dir(os.path.dirname(os.path.abspath(dst)))


def pack(lib, say=print, leave_out=False):
    """The folder into the image, and the folder removed. Returns whether an image now stands for the library. Without a folder there is nothing to do, unless leave_out finds pulling libraries in the
    image to take out: leave_out is the iPhone build's when nothing will read them"""
    folder, image = folder_of(lib), image_of(lib)
    try:
        before = _names(folder) if os.path.isdir(folder) else None
        if before is not None and not before: os.rmdir(folder); before = None
        if before is None and not os.path.isfile(image): return False
        files = {n: os.path.join(folder, n) for n in before or ()}
        if os.path.isfile(image) and (before is None or STAMP not in before):   # a partial build over a packed library, or the image alone
            with romfs.scandir(image) as it:
                for e in it: files.setdefault(e.name, e.path)
        pulling = [n for n in files if n.endswith(PULLING)]
        if before is None and not (leave_out and pulling): return True
        if leave_out:
            for n in pulling: del files[n]
            files[PULLING_LEFT_OUT] = b'the pulling vertex libraries are left out: nothing in this build reads them (dm/tools/shader_lib.py)\n'
        write(files, image)
        if before is None: return True
        if _names(folder) != before:   # a build wrote to it meanwhile: the folder is the newer one, and the next pack takes it
            say(f'{folder} changed while it was packed: it stays a folder'); return True
        shutil.rmtree(folder)
    except (OSError, ValueError) as e:
        say(f'[RED] could not pack {folder}, it stays a folder: {e}')
        try: os.unlink(image + '.part')
        except OSError: pass
        return False
    return True


def remove_image(lib):   # before a full build: its folder replaces the library, and a stopped build must not take old shaders for the ones it had not reached
    image = image_of(lib)
    root = romfs.Path(image)
    if (root / SOURCES).is_file():
        with _source_lock(image, True):
            manifest = _source_manifest(romfs.Path(image))
            os.unlink(image)
            _sync_dir(os.path.dirname(os.path.abspath(image)))
            _release_sources(image, manifest)
    else:
        try: os.unlink(image)
        except FileNotFoundError: pass
    try: os.unlink(image + '.part')
    except FileNotFoundError: pass


def _release_sources(image, manifest):
    cas = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(image)), manifest['cas']))
    if os.path.isfile(cas):
        with contextlib.closing(shader_cache.Store(cas)) as store: _drop_sources(store, manifest['generation'])


def remove_tree(path):
    """Remove shader output trees under source locks; release pins only after confirmed removal."""
    if not os.path.exists(path): return
    records = []
    with contextlib.ExitStack() as locks:
        for directory, _, names in os.walk(path):
            for name in sorted(names):
                if not name.endswith('.img'): continue
                image = os.path.join(directory, name)
                if not (romfs.Path(image) / SOURCES).is_file(): continue
                locks.enter_context(_source_lock(image, True))
                records.append((image, _source_manifest(romfs.Path(image))))
        shutil.rmtree(path)
        if records: _sync_dir(os.path.dirname(os.path.abspath(path)))
        for image, manifest in records: _release_sources(image, manifest)


def expand(lib):
    """The image back into a folder and the image removed, for a host from before this file (a kept host copy of the same chain). No build or run of the current sources needs it"""
    folder, image = folder_of(lib), image_of(lib)
    if not os.path.isfile(image): return
    if (romfs.Path(image) / SOURCES).is_file(): raise SourceError('[RED] source-referencing images cannot be expanded in place; use scripts/export_shader_sources.py for inspection')
    work = folder + '.expand'
    shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
    for p in romfs.Path(image).iterdir():
        with builtins.open(os.path.join(work, p.name), 'wb') as out: out.write(p.read_bytes())
    if os.path.isdir(folder):   # what a partial build made wins, as in pack
        for n in _names(folder): os.replace(os.path.join(folder, n), os.path.join(work, n))
        shutil.rmtree(folder)
    os.rename(work, folder); os.unlink(image)


def app_files(lib, skip_suffix=()):
    """{name: path} of what an app carries of the library: what the host reads, and of that what the build wants (build_ipa.sh leaves out the pulling vertex libraries when nothing uses them)"""
    return {p.name: str(p) for p in view(lib).iterdir() if p.name.endswith(APP_SUFFIXES) and p.name not in APP_LEAVE_OUT and not p.name.startswith('.') and '.tmp' not in p.name and not p.name.endswith(tuple(skip_suffix) or ('\0',))}


def subset(lib, dst, skip_suffix=()):   # → the number of files in the image dst
    files = app_files(lib, skip_suffix)
    if not any(n.endswith('.metallib') for n in files): raise ValueError(f'no shader libraries at {lib}')
    write(files, os.fspath(dst))
    return len(files)


def has_library(lib): return next(view(lib).glob('*_b.metallib'), None) is not None
def left_out_pulling(lib): return (view(lib) / PULLING_LEFT_OUT).is_file()   # whether a build that needs the pulling libraries must make them again


def main():
    ap = argparse.ArgumentParser(description='A game\'s shader libraries as one file, read in place')
    ap.add_argument('cmd', choices=('pack', 'expand', 'path', 'has', 'cat', 'subset'))
    ap.add_argument('lib'); ap.add_argument('arg', nargs='?')
    ap.add_argument('--skip-suffix', action='append', default=[])
    a = ap.parse_args()
    if a.cmd == 'pack': return 0 if pack(a.lib) else 1
    if a.cmd == 'expand': expand(a.lib); return 0
    if a.cmd == 'path': print(view(a.lib)); return 0
    if a.cmd == 'has': return 0 if has_library(a.lib) else 1
    if a.cmd == 'cat':
        try:
            if a.arg.endswith('.metal'):
                with SourceReader(a.lib) as reader: sys.stdout.buffer.write(reader.read_bytes(a.arg))
            else: sys.stdout.buffer.write((view(a.lib) / a.arg).read_bytes())
        except (OSError, SourceError) as e: sys.exit(f'[RED] {e}')
        return 0
    try: print(f'{a.arg}: {subset(a.lib, a.arg, a.skip_suffix)} files, {os.path.getsize(a.arg)} bytes')
    except (OSError, ValueError) as e: sys.exit(f'[RED] {e}')
    return 0


if __name__ == '__main__': sys.exit(main())
