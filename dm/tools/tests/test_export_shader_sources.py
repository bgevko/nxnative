"""Optional inspection exports never become authoritative shader input."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock
from dm.tools.tests.test_shader_sources import Base, NAME, REPO, shader_lib


def exporter():
    spec = importlib.util.spec_from_file_location('export_shader_sources_test', REPO / 'scripts/export_shader_sources.py')
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class Export(Base):
    def test_exports_only_exact_metal_bytes(self):
        out = self.root / 'inspection'
        self.assertEqual(exporter().export_sources(self.lib, out), 1)
        self.assertEqual({p.name: p.read_bytes() for p in out.iterdir()}, {NAME: self.data})
        (out / NAME).write_bytes(b'inspection edit')
        with shader_lib.SourceReader(self.lib) as reader: self.assertEqual(reader.read_bytes(NAME), self.data)

    def test_failed_source_write_preserves_its_existing_target(self):
        out = self.root / 'inspection'; out.mkdir(); (out / NAME).write_bytes(b'previous export')
        module = exporter()
        with mock.patch.object(module.shader_lib.SourceReader, 'read_bytes', side_effect=module.shader_lib.SourceError('[RED] missing source')):
            with self.assertRaises(module.shader_lib.SourceError): module.export_sources(self.lib, out)
        self.assertEqual({p.name: p.read_bytes() for p in out.iterdir()}, {NAME: b'previous export'})

    def test_unrelated_destination_files_are_preserved(self):
        out = self.root / 'inspection'; out.mkdir(); (out / 'notes.txt').write_bytes(b'personal notes')
        exporter().export_sources(self.lib, out)
        self.assertEqual((out / 'notes.txt').read_bytes(), b'personal notes')

    def test_output_cannot_be_the_authoritative_library_directory(self):
        for target in (self.lib, self.lib / 'nested'):
            with self.assertRaises(ValueError): exporter().export_sources(self.lib, target)
        self.assertFalse(self.lib.exists())

    def test_missing_library_fails_without_creating_an_export(self):
        out = self.root / 'inspection'
        with self.assertRaises(shader_lib.SourceError): exporter().export_sources(self.root / 'missing', out)
        self.assertFalse(out.exists())

    def test_cli_reports_missing_cas_as_nonzero(self):
        self.store.close(); Path(self.store.path).unlink()
        result = subprocess.run([sys.executable, REPO / 'scripts/export_shader_sources.py', self.lib, '--out', self.root / 'export'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0); self.assertIn('[RED]', result.stderr)

    def test_build_cli_reports_source_error_without_traceback(self):
        from dm.tools.tests.test_nsp_chain import nsp
        with mock.patch.object(nsp, 'main', side_effect=nsp.shader_lib.SourceError('[RED] missing source; rebuild')), mock.patch.object(nsp, 'PROG') as progress:
            self.assertEqual(nsp.cli([]), 1)
            progress.say.assert_called_with('[RED] missing source; rebuild')

    def test_build_flag_exports_after_reused_shader_stage(self):
        self.build_export(True)

    def test_flag_off_has_no_export_work(self):
        self.build_export(False)

    def test_cli_overrides_spyder_export_setting(self):
        from dm.tools.tests.test_nsp_chain import nsp
        tree = ast.parse((REPO / 'scripts/build_ipa_from_nsp.py').read_text())
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'export_sources' for t in n.targets))
        for setting, option, expected in ((False, [], False), (True, [], True), (True, ['--no-export-shader-sources'], False), (False, ['--export-shader-sources'], True)):
            ns = dict(EXPORT_SHADER_SOURCES=setting, args=nsp.parse_args(['folder', *option]))
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<export precedence>', 'exec'), ns)
            self.assertIs(ns['export_sources'], expected)

    def build_export(self, enabled):
        from dm.tools.tests.test_nsp_chain import nsp
        args = nsp.parse_args(['folder', '--export-shader-sources' if enabled else '--no-export-shader-sources'])
        self.assertEqual(args.export_shader_sources, enabled)
        tree = ast.parse((REPO / 'scripts/build_ipa_from_nsp.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        branch = next(n for n in ast.walk(main) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'export_sources')
        stage = next(n for n in ast.walk(main) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'build_shaders')
        run = mock.Mock()
        reused = mock.Mock(return_value=self.root / 'shader/cache')
        ns = dict(export_sources=enabled, py=sys.executable, repo=REPO, sh=self.root / 'shader', xlate='translator', env={}, logs=self.root / 'logs', g=None, pkey='same key',
                  mac=True, seen=None, args=args, prepare_xcode=False, build_shaders=reused, run=run, PROG=mock.Mock(), chain_of=lambda _: '0123456789ab')
        exec(compile(ast.Module(body=[stage, branch], type_ignores=[]), '<build export>', 'exec'), ns)
        reused.assert_called_once()
        if enabled:
            self.assertEqual(run.call_count, 1)
            self.assertIn('export_shader_sources.py', str(run.call_args.args[0][1]))
            self.assertEqual(Path(run.call_args.args[0][2]), self.root / 'shader/cache/b-0123456789ab')
        else: run.assert_not_called()


if __name__ == '__main__': unittest.main()
