"""Real npm/proxy TLS regression tests; no package downloads or app imports.

Run with: python3 tests/install/test_sandbox_node_tls.py
Requires node, npm and openssl. All CAs, npm config/cache and processes are
throwaway; this never disables certificate verification.
"""
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parents[2]


@unittest.skipUnless(all(shutil.which(x) for x in ('node', 'npm', 'openssl')),
                     'node, npm and openssl required')
class SandboxNodeTLS(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='sandbox-node-tls-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.certs = self.root / 'certs'
        self.certs.mkdir()
        self.env = {
            'PATH': os.environ['PATH'], 'HOME': str(self.root),
            'OPENSSL_CONF': str(REPO / 'scripts/sandbox/openssl.cnf'),
            'npm_config_cache': str(self.root / 'npm-cache'),
            'npm_config_userconfig': str(self.root / 'npmrc'),
            'npm_config_globalconfig': str(self.root / 'global-npmrc'),
            'NO_PROXY': '', 'no_proxy': '',
        }
        for name in ('ca', 'real-ca'):
            subprocess.run([
                'openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                '-days', '1', '-subj', f'/CN={name}',
                '-keyout', str(self.certs / f'{name}.key'),
                '-out', str(self.certs / f'{name}.pem'),
                '-extensions', 'sandbox_ca_ext',
            ], env=self.env, check=True, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE)
        fixture = self.root / 'http/registry.sandbox.test/-/ping'
        fixture.parent.mkdir(parents=True)
        fixture.write_text('{}')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        # Execute the production proxy with its normal globals, on a test port.
        runner = (
            'import runpy,sys; '
            'p=sys.argv[1]; port=int(sys.argv.pop()); '
            'sys.argv=sys.argv[1:]; '
            'ns=runpy.run_path(p); '
            'ns["main"].__globals__["LISTEN_ADDRESS"]=("127.0.0.1",port); '
            'ns["main"]()'
        )
        self.log = open(self.root / 'proxy.log', 'w+')
        self.addCleanup(self.log.close)
        self.proxy = subprocess.Popen([
            sys.executable, '-c', runner, str(REPO / 'scripts/sandbox/proxy.py'),
            str(self.root / 'http'), str(self.certs),
            str(self.certs / 'real-ca.pem'), str(port),
        ], env=self.env, stdout=self.log, stderr=self.log)
        self.addCleanup(self.stop_proxy)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.proxy.poll() is not None:
                self.fail('proxy exited before readiness')
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=.2):
                    break
            except OSError:
                time.sleep(.05)
        else:
            self.fail('proxy did not become ready')
        self.proxy_url = f'http://127.0.0.1:{port}'

    def stop_proxy(self):
        self.proxy.terminate()
        try:
            self.proxy.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proxy.kill()
            self.proxy.wait(timeout=5)

    def ping(self, ca):
        env = dict(self.env, NODE_EXTRA_CA_CERTS=str(ca),
                   HTTPS_PROXY=self.proxy_url, HTTP_PROXY=self.proxy_url)
        return subprocess.run([
            'npm', 'ping', '--registry=https://registry.sandbox.test',
            '--strict-ssl=true', '--fetch-retries=0', '--fetch-timeout=5000',
            '--loglevel=notice', '--json',
        ], env=env, cwd=self.root, capture_output=True, text=True, timeout=15)

    def test_stage2_node_trusts_sandbox_proxy(self):
        stage = (REPO / 'scripts/sandbox/stage2-run.sh').read_text()
        match = re.search(r'--setenv NODE_EXTRA_CA_CERTS (/work/certs/\S+)', stage)
        assert match is not None, 'stage2 must explicitly configure Node CA trust'
        ca = self.certs / Path(match.group(1)).name
        result = self.ping(ca)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PONG', result.stderr)

    def test_unrelated_ca_is_rejected(self):
        result = self.ping(self.certs / 'real-ca.pem')
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stdout + result.stderr,
                         'CERT|certificate|SELF_SIGNED|UNABLE_TO_VERIFY')


class SandboxNodeHeaders(unittest.TestCase):
    def select_headers(self, override):
        script = (REPO / 'scripts/dev-sandbox.sh').read_text()
        start = script.index('NODE_DIR="${DEV_SANDBOX_NODE_DIR:-}"')
        end = script.index('WAYLAND_SOCKET=""', start)
        # Exercise the production selection block with a host Node present.
        command = ("command() { [ \"$*\" = '-v node' ] && "
                   "printf '/usr/local/bin/node\\n'; };\n" +
                   script[start:end] + '\nprintf "%s" "$NODE_DIR"')
        env = {'PATH': os.environ['PATH']}
        if override is not None:
            env['DEV_SANDBOX_NODE_DIR'] = override
        return subprocess.check_output(['bash', '-ceu', command], env=env, text=True)

    def test_host_node_does_not_override_installed_node_headers(self):
        self.assertEqual(self.select_headers(None), '')

    def test_explicit_header_directory_is_preserved(self):
        self.assertEqual(self.select_headers('/nix/store/test-node'),
                         '/nix/store/test-node')


if __name__ == '__main__':
    unittest.main(verbosity=2)
