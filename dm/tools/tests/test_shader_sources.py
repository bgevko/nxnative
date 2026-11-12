"""Sources stay in CAS while native shader payloads remain ordinary image entries."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'dm/tools'))
import romfs, shader_cache, shader_lib

NAME = '1111111111111111_b.metal'
GEN = 'a' * 32
MANIFEST = '_b_sources.json'


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.lib = self.root / 'cache/b-0123456789ab'
        self.image = Path(str(self.lib) + '.img')
        self.store = shader_cache.Store(self.root / 'cache/cas.db'); self.addCleanup(self.store.close)
        self.data = b'original source\n'
        self.digest = self.store.put(f'image/{GEN}/{NAME}', self.data)
        self.manifest = {'version': 1, 'cas': 'cas.db', 'generation': GEN, 'sources': {NAME: self.digest}}
        self.write()

    def write(self, manifest=None):
        value = self.manifest if manifest is None else manifest
        payload = json.dumps(value).encode() if not isinstance(value, bytes) else value
        shader_lib.write({MANIFEST: payload, '1111111111111111_b.metallib': b'compiled', '1111111111111111_b.json': b'{}'}, str(self.image))
        self.image.with_name('.' + self.image.name + '.sources.lock').touch()


class Sources(Base):
    def test_sources_are_enumerated_without_inline_payloads(self):
        self.assertFalse((romfs.Path(self.image) / NAME).is_file())
        with shader_lib.SourceReader(self.lib) as reader:
            self.assertEqual(reader.names(), {NAME})
            self.assertEqual(reader.read_bytes(NAME), self.data)
            with self.assertRaises(FileNotFoundError): reader.read_bytes('absent.metal')

    def test_reference_keeps_old_bytes_after_index_changes(self):
        self.store.put('index/key.json', b'new digest')
        self.store.put('shader/new/mac/' + NAME, b'new source')
        with shader_lib.SourceReader(self.lib) as reader: self.assertEqual(reader.read_bytes(NAME), self.data)

    def test_relative_cas_survives_build_tree_move(self):
        self.store.close()
        shutil.move(self.root / 'cache', self.root / 'moved')
        with shader_lib.SourceReader(self.root / 'moved' / self.lib.name) as reader:
            self.assertEqual(reader.read_bytes(NAME), self.data)

    def test_missing_database_or_blob_raises_source_error(self):
        self.store.delete_prefix('image/')
        with shader_lib.SourceReader(self.lib) as reader:
            with self.assertRaises(shader_lib.SourceError): reader.read_bytes(NAME)
        self.store.close(); Path(self.store.path).unlink()
        with self.assertRaises(shader_lib.SourceError): shader_lib.SourceReader(self.lib)
        self.assertFalse(Path(self.store.path).exists())

    def test_bad_hash_or_manifest_raises_source_error(self):
        self.store.put(f'image/{GEN}/{NAME}', b'wrong bytes')
        with shader_lib.SourceReader(self.lib) as reader:
            with self.assertRaises(shader_lib.SourceError): reader.read_bytes(NAME)
        self.write(b'{bad json')
        with self.assertRaises(shader_lib.SourceError): shader_lib.SourceReader(self.lib)

    def test_manifest_validation(self):
        cases = [('version', 2), ('generation', '../bad'), ('cas', '/absolute/cas.db'), ('cas', 1),
                 ('sources', {'../escape.metal': self.digest}), ('sources', {'/escape.metal': self.digest}),
                 ('sources', {'not-source.json': self.digest}), ('sources', {NAME: 'not-a-hash'})]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                manifest = copy.deepcopy(self.manifest); manifest[field] = value; self.write(manifest)
                with self.assertRaises(shader_lib.SourceError): shader_lib.SourceReader(self.lib)
        self.write(b'{"version":1,"version":1}')
        with self.assertRaises(shader_lib.SourceError): shader_lib.SourceReader(self.lib)

    def test_raw_historical_sources_use_existing_image_reads(self):
        shader_lib.write({NAME: self.data}, str(self.image))
        with shader_lib.SourceReader(self.lib) as reader:
            self.assertEqual(reader.names(), {NAME}); self.assertEqual(reader.read_bytes(NAME), self.data)

    def test_non_source_reads_and_app_subset_need_no_cas(self):
        self.store.close(); Path(self.store.path).unlink()
        self.assertEqual((shader_lib.view(self.lib) / '1111111111111111_b.metallib').read_bytes(), b'compiled')
        app = self.root / 'app.img'; shader_lib.subset(self.lib, app)
        self.assertNotIn(MANIFEST, {p.name for p in romfs.Path(app).iterdir()})
        self.assertFalse(Path(self.store.path).exists())


class Publication(Base):
    def publish(self, image=None, source: bytes | Path = b'new source'):
        shader_lib.write_library({NAME: source, '1111111111111111_b.metallib': b'compiled', '1111111111111111_b.json': b'{}'}, str(image or self.image), self.store)

    def test_publish_links_staged_sources_without_new_blobs(self):
        before = self.store._run('SELECT count(*) FROM blob')[0][0]
        shader_lib.write_library({NAME: lambda: self.fail('must not decode a staged source'), '1111111111111111_b.metallib': b'compiled'}, str(self.image), self.store,
                                 source_entries={NAME: f'image/{GEN}/{NAME}'})
        self.assertEqual(self.store._run('SELECT count(*) FROM blob')[0][0], before + 1)  # owner metadata only
        self.assertFalse((romfs.Path(self.image) / NAME).is_file())
        with shader_lib.SourceReader(self.lib) as reader: self.assertEqual(reader.read_bytes(NAME), self.data)

    def test_uncached_loose_source_is_retained(self):
        source = self.root / 'loose.metal'; source.write_bytes(b'loose')
        self.publish(source=source)
        self.store.prune_shaders(set(), time.time())
        with shader_lib.SourceReader(self.lib) as reader: self.assertEqual(reader.read_bytes(NAME), b'loose')

    def test_failed_publication_preserves_previous_image_and_pins(self):
        self.publish(); before = self.image.read_bytes()
        with mock.patch.object(shader_lib, 'write', side_effect=OSError('disk full')):
            with self.assertRaises(OSError): self.publish(source=b'changed')
        self.assertEqual(self.image.read_bytes(), before)
        shader_lib.prune_sources(self.store)
        with shader_lib.SourceReader(self.lib) as reader: self.assertEqual(reader.read_bytes(NAME), b'new source')

    def test_shared_blob_survives_one_image_retirement(self):
        other = self.root / 'cache/b-other.img'
        self.publish(); self.publish(other)
        shader_lib.remove_image(self.lib)
        shader_lib.prune_sources(self.store)
        with shader_lib.SourceReader(other) as reader: self.assertEqual(reader.read_bytes(NAME), b'new source')

    def test_busy_reader_blocks_replacement_and_pruning(self):
        self.publish()
        with shader_lib.SourceReader(self.lib) as reader:
            with self.assertRaises(shader_lib.SourceError): self.publish(source=b'changed')
            with self.assertRaises(shader_lib.SourceError): shader_lib.remove_image(self.lib)
            shader_lib.prune_sources(self.store)
            self.assertEqual(reader.read_bytes(NAME), b'new source')

    def test_interrupted_owner_is_reclaimed_after_unlock(self):
        with mock.patch.object(shader_lib, 'write', side_effect=OSError('disk full')):
            with self.assertRaises(OSError): self.publish(source=b'failed source')
        self.assertTrue(self.store.names('image-owner/'))
        shader_lib.prune_sources(self.store)
        self.assertEqual(self.store.names('image-owner/'), set())

    def test_failed_image_removal_keeps_sources(self):
        self.publish()
        unlink = os.unlink
        def fail_image(path, *args, **kwargs):
            if os.fspath(path) == str(self.image): raise PermissionError('busy image')
            return unlink(path, *args, **kwargs)
        with mock.patch.object(os, 'unlink', side_effect=fail_image):
            with self.assertRaises(PermissionError): shader_lib.remove_image(self.lib)
        shader_lib.prune_sources(self.store)
        with shader_lib.SourceReader(self.lib) as reader: self.assertEqual(reader.read_bytes(NAME), b'new source')

    def test_tree_retirement_keeps_references_if_removal_fails(self):
        self.publish()
        with mock.patch.object(shutil, 'rmtree', side_effect=OSError('cannot remove tree')):
            with self.assertRaises(OSError): shader_lib.remove_tree(self.image.parent)
        with shader_lib.SourceReader(self.lib) as reader: self.assertEqual(reader.read_bytes(NAME), b'new source')
        shader_lib.remove_tree(self.image.parent)
        self.assertFalse(self.image.exists())

    def test_production_reuse_rejects_raw_images_and_missing_references(self):
        shader_lib.write({NAME: self.data}, str(self.image))
        with self.assertRaises(shader_lib.SourceError): shader_lib.require_sources(self.lib)
        self.publish(); shader_lib.require_sources(self.lib)
        self.store.delete_prefix('image/')
        with self.assertRaises(shader_lib.SourceError): shader_lib.require_sources(self.lib)

    def test_corrupt_or_unreadable_image_keeps_retained_sources(self):
        self.publish()
        aliases = self.store.names('image/')
        valid = self.image.read_bytes()
        self.image.write_bytes(b'badmagic' + valid[8:])
        self.assertEqual(shader_lib.prune_sources(self.store), 0)
        self.assertEqual(self.store.names('image/'), aliases)
        self.image.write_bytes(valid)
        stat = os.stat
        def unreadable(path, *args, **kwargs):
            if os.fspath(path) == str(self.image): raise PermissionError('unreadable image')
            return stat(path, *args, **kwargs)
        with mock.patch.object(os, 'stat', side_effect=unreadable):
            self.assertEqual(shader_lib.prune_sources(self.store), 0)
        self.assertEqual(self.store.names('image/'), aliases)

    def test_source_free_replacement_is_synced_before_retirement(self):
        self.publish()
        events = []
        drop = shader_lib._drop_sources
        def record_drop(store, generation): events.append('drop'); return drop(store, generation)
        with mock.patch.object(os, 'fsync', side_effect=lambda _: events.append('sync')), mock.patch.object(shader_lib, '_drop_sources', side_effect=record_drop):
            shader_lib.write_library({'1111111111111111_b.metallib': b'compiled'}, self.image, self.store)
        self.assertIn('drop', events)
        self.assertGreaterEqual(events[:events.index('drop')].count('sync'), 2)

    def test_image_and_tree_retirement_keep_pins_when_sync_fails(self):
        for tree in (False, True):
            with self.subTest(tree=tree):
                directory = self.root / ('tree' if tree else 'single'); directory.mkdir()
                image = directory / 'b-test.img'; self.publish(image)
                before = self.store.names('image/')
                with mock.patch.object(os, 'fsync', side_effect=OSError('sync failed')):
                    with self.assertRaisesRegex(OSError, 'sync failed'):
                        shader_lib.remove_tree(directory) if tree else shader_lib.remove_image(image)
                self.assertEqual(self.store.names('image/'), before)

    def test_production_reuse_rejects_dangling_or_changed_blob_identity(self):
        for dangling in (False, True):
            self.publish()
            manifest = json.loads((romfs.Path(self.image) / MANIFEST).read_bytes())
            alias = f'image/{manifest["generation"]}/{NAME}'
            if dangling: self.store.db.execute('DELETE FROM blob WHERE id=(SELECT blob FROM entry WHERE name=?)', (alias,))
            else: self.store.put(alias, b'wrong source')
            with self.assertRaises(shader_lib.SourceError): shader_lib.require_sources(self.lib)


class Analysis(Base):
    def fixtures(self):
        key = NAME[:16]
        source = b'device float4* writable;\nfragment float4 main0(constant cb& fp_c1) { int _5 = in.v; return (fp_c1)._m0[_5]; }\n'
        bounded = '2222222222222222_b'
        files = {NAME: source, key + '_b.json': json.dumps({'stage': 'ps', 'bindings': [{'kind': 'cbuf', 'reg': 0, 'msl_name': 'fp_c1'}]}).encode(),
                 key + '_b.metallib': b'compiled', '_b_build.json': json.dumps([{'key': key, 'st': 'ok', 'err': ''}]).encode(),
                 bounded + '.metal': b'struct Args { constant float4* fp_c1 [[id(0)]]; }; kernel void main0() { float4 value = (*spvDescriptorSet0.fp_c1)._m0[3]; }',
                 bounded + '.json': json.dumps({'stage': 'cs', 'bindings': [{'kind': 'cbuf', 'reg': 0, 'msl_name': 'fp_c1'}]}).encode()}
        raw = self.root / 'baseline.img'; shader_lib.write(files, str(raw))
        shader_lib.write_library(files, self.image, self.store)
        return raw

    def test_guard_and_extent_tables_match_inline_sources(self):
        raw = self.fixtures()
        for tool in ('c1_guard.py', 'cb_extent.py'):
            bodies = []
            for i, lib in enumerate((raw, self.image)):
                target = self.root / (tool + str(i) + '.tsv')
                run = subprocess.run([sys.executable, REPO / 'dm/gfx/tools' / tool, lib, target], capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                bodies.append([line for line in target.read_text().splitlines() if not line.startswith('#')])
            self.assertTrue(bodies[0], tool)
            self.assertEqual(bodies[0], bodies[1])

    def test_missing_sources_do_not_produce_successful_tables(self):
        self.fixtures(); self.store.delete_prefix('image/')
        for tool in ('c1_guard.py', 'cb_extent.py'):
            target = self.root / (tool + '.tsv')
            run = subprocess.run([sys.executable, REPO / 'dm/gfx/tools' / tool, self.image, target], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0, tool)
            self.assertIn('[RED]', run.stderr)
            self.assertFalse(target.exists())

    def test_missing_manifest_does_not_produce_empty_safety_tables(self):
        self.fixtures()
        shader_lib.write({p.name: str(p) for p in romfs.Path(self.image).iterdir() if p.name != MANIFEST}, str(self.image))
        for tool in ('c1_guard.py', 'cb_extent.py'):
            target = self.root / (tool + '.tsv')
            run = subprocess.run([sys.executable, REPO / 'dm/gfx/tools' / tool, self.image, target], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0, tool)
            self.assertIn('[RED]', run.stderr)
            self.assertFalse(target.exists())

    def test_nvn_manifest_preserves_source_and_variant_hashes(self):
        from dm.tools.tests.test_regress import regress
        raw = self.fixtures()
        self.assertEqual(regress.nvn_manifest(romfs.Path(raw), 2), regress.nvn_manifest(romfs.Path(self.image), 2))
        self.store.delete_prefix('image/')
        with self.assertRaises(regress.shader_lib.SourceError): regress.nvn_manifest(romfs.Path(self.image), 2)

    def test_usage_keeps_source_feature_counts(self):
        import usage
        raw = self.fixtures()
        self.assertEqual(usage.shader_features(raw)['device_buffer_write'], 1)
        self.assertEqual(usage.shader_features(raw), usage.shader_features(self.image))
        self.store.delete_prefix('image/')
        with self.assertRaises(shader_lib.SourceError): usage.shader_features(self.image)

    def test_interface_check_fails_on_missing_referenced_source(self):
        from dm.tools.tests.test_regress import regress
        self.fixtures(); self.store.delete_prefix('image/')
        profile = self.root / 'profile'; profile.mkdir()
        (profile / 'shader_pairs.tsv').write_text(NAME[:16] + '\t2222222222222222\n')
        tree = SimpleNamespace(path=REPO, id='source-proof')
        game = SimpleNamespace(profile_dir=lambda _: profile, build=self.root)
        with mock.patch.object(regress, 'depth_zero', return_value=False):
            with self.assertRaises(regress.shader_lib.SourceError): regress.interface_check(tree, game, self.image.parent)

    def test_expand_rejects_referenced_image_without_modification(self):
        self.fixtures(); before = self.image.read_bytes()
        with self.assertRaises(shader_lib.SourceError): shader_lib.expand(self.lib)
        self.assertEqual(self.image.read_bytes(), before)
        with shader_lib.SourceReader(self.lib) as reader: self.assertIn(NAME, reader.names())


if __name__ == '__main__': unittest.main()
