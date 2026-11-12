"""Existing shader images lose only inline sources and retain exact CAS ownership."""
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest import mock
from dm.tools.tests.test_shader_sources import Base, CompressedCAS, NAME, REPO, romfs, shader_lib


def optimizer():
    spec = importlib.util.spec_from_file_location('optimize_shader_image_test', REPO / 'scripts/optimize_shader_image.py')
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class Optimize(Base):
    def setUp(self):
        super().setUp()
        self.data = b'original source\n' * 512
        self.store.put('shader/original/mac/' + NAME, self.data)
        self.files = {NAME: self.data, '1111111111111111_b.metallib': b'compiled',
                      '1111111111111111_b.json': b'{"chain":"old-chain"}',
                      'inputs.tsv': b'game\tfixture\ninputs\toriginal\n', '_b_build.json': b'[]'}
        shader_lib.write(self.files, str(self.image))
        self.tool = optimizer()

    def test_rewrite_preserves_non_sources_and_reuses_source_blob(self):
        before = self.image.stat().st_size
        count, saved = self.tool.optimize_image(self.image, self.store.path)
        self.assertEqual(count, 1)
        self.assertEqual(saved, before - self.image.stat().st_size)
        self.assertGreater(saved, 0)
        after = {p.name: p.read_bytes() for p in romfs.Path(self.image).iterdir()}
        self.assertNotIn(NAME, after)
        after.pop(shader_lib.SOURCES)
        self.assertEqual(after, {n: data for n, data in self.files.items() if n != NAME})
        with shader_lib.SourceReader(self.image) as reader:
            reader.validate()
            self.assertEqual(reader.read_bytes(NAME), self.data)
        digest = hashlib.blake2b(self.data, digest_size=16).digest()
        self.assertEqual(self.store._run('SELECT count(*) FROM blob WHERE hash=?', digest), [(1,)])

    def test_repeated_conversion_validates_without_rewriting(self):
        self.tool.optimize_image(self.image, self.store.path)
        before = self.image.read_bytes()
        entries = self.store.names('image/')
        self.assertEqual(self.tool.optimize_image(self.image, self.store.path), (0, 0))
        self.assertEqual(self.image.read_bytes(), before)
        self.assertEqual(self.store.names('image/'), entries)
        self.store.delete_prefix('image/')
        with self.assertRaises(shader_lib.SourceError): self.tool.optimize_image(self.image, self.store.path)

    def test_failed_write_preserves_original_image(self):
        before = self.image.read_bytes()
        with mock.patch.object(shader_lib, 'write', side_effect=OSError('disk full')):
            with self.assertRaises(OSError): self.tool.optimize_image(self.image, self.store.path)
        self.assertEqual(self.image.read_bytes(), before)
        self.assertEqual((romfs.Path(self.image) / NAME).read_bytes(), self.data)

    def test_missing_cas_is_not_created(self):
        before = self.image.read_bytes()
        missing = self.root / 'missing/cas.db'
        with self.assertRaises(ValueError): self.tool.optimize_image(self.image, missing)
        self.assertFalse(missing.parent.exists())
        self.assertEqual(self.image.read_bytes(), before)

    def test_loose_build_folder_is_rejected(self):
        self.lib.mkdir()
        before = self.image.read_bytes()
        with self.assertRaises(ValueError): self.tool.optimize_image(self.image, self.store.path)
        self.assertEqual(self.image.read_bytes(), before)

    def test_cas_cannot_be_the_image_write_destination(self):
        part = Path(str(self.image) + '.part'); part.write_bytes(b'preserve this file')
        before = self.image.read_bytes()
        for destination in (self.image, part):
            with self.assertRaises(ValueError): self.tool.optimize_image(self.image, destination)
        self.assertEqual(part.read_bytes(), b'preserve this file')
        self.assertEqual(self.image.read_bytes(), before)

    def test_corrupt_image_is_rejected_before_retaining_sources(self):
        before = b'badmagic' + self.image.read_bytes()[8:]
        self.image.write_bytes(before)
        entries = self.store.names('image/')
        with self.assertRaises(ValueError): self.tool.optimize_image(self.image, self.store.path)
        self.assertEqual(self.image.read_bytes(), before)
        self.assertEqual(self.store.names('image/'), entries)

    def test_incomplete_source_set_is_rejected(self):
        shader_lib.write({**self.files, '2222222222222222_b.metallib': b'other compiled'}, str(self.image))
        before = self.image.read_bytes()
        with self.assertRaises(shader_lib.SourceError): self.tool.optimize_image(self.image, self.store.path)
        self.assertEqual(self.image.read_bytes(), before)

    def test_source_free_image_is_unchanged_without_opening_cas(self):
        shader_lib.write({'1111111111111111_b.metallib': b'compiled'}, str(self.image))
        before = self.image.read_bytes(); missing = self.root / 'absent.db'
        self.assertEqual(self.tool.optimize_image(self.image, missing), (0, 0))
        self.assertFalse(missing.exists())
        self.assertEqual(self.image.read_bytes(), before)


class CompressedOptimize(CompressedCAS, Optimize): pass


if __name__ == '__main__': unittest.main()
