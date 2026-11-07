import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server = load('server')
bench = load('benchmark')
model = load('maya_model')


class FakeClient:
    def __init__(self, slots=4, context=262144):
        self.slots, self.context = slots, context
        self.prompts = []
        self.lock = threading.Lock()

    def request(self, path):
        if path == '/v1/models':
            return {'data': [{'id': 'test'}]}
        return {'concurrency': {'serving': self.slots}, 'context': {'max_positions': self.context}}

    def complete(self, model, prompt, tokens):
        with self.lock:
            self.prompts.append(prompt)
        return {'text': '4', 'usage': {'completion_tokens': 16}, 'timings': {'predicted_per_second': 100}}


class Tests(unittest.TestCase):
    def test_qualify(self):
        self.assertEqual(bench.qualify(FakeClient())['concurrency']['serving'], 4)

    def test_slot_fallback_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'slots'):
            bench.qualify(FakeClient(slots=3))

    def test_small_context_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'context'):
            bench.qualify(FakeClient(context=65536))

    def test_bad_answer_rejected(self):
        client = FakeClient()
        client.complete = lambda *args: {'text': '5'}
        with self.assertRaisesRegex(RuntimeError, 'Arithmetic'):
            bench.qualify(client)

    def test_four_distinct_requests(self):
        client = FakeClient()
        result = bench.benchmark(client, 4, 'prompt', 16)
        self.assertEqual(len(result['requests']), 4)
        self.assertEqual(len(set(client.prompts)), 4)
        self.assertGreater(result['aggregate_end_to_end_tok_s'], 0)

    def test_key_private_and_stable(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(server, 'ROOT', Path(tmp)):
            path = server.ensure_key()
            original = path.read_text()
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(server.ensure_key().read_text(), original)
            self.assertNotIn(original.split('=', 1)[1].strip(), json.dumps(json.loads((ROOT / 'config/engines.json').read_text())))

    def test_pinned_start(self):
        cfg = json.loads((ROOT / 'config/engines.json').read_text())['strata']
        with tempfile.TemporaryDirectory() as tmp, patch.object(server, 'ROOT', Path(tmp)):
            command = server.start_command('strata', cfg, 8000, 'nvidia.com/gpu=all')
        self.assertIn('@sha256:', command[-1])
        self.assertIn('127.0.0.1:8000:8080', command)
        self.assertIn('PARALLEL=4', command)
        self.assertIn('CONTEXT=256k', command)
        self.assertEqual(command[command.index('--restart') + 1], 'no')
        self.assertNotIn('--ulimit', command)

    def test_hash_and_size_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model'
            path.write_bytes(b'abc')
            digest = model.sha256(path)
            self.assertTrue(model.verify(path, 3, digest))
            self.assertFalse(model.verify(path, 4, digest))
            self.assertFalse(model.verify(path, 3, 'bad'))

    def test_quant_profiles_keep_context_and_slots(self):
        for quant, model_name, family in [('q3', 'UD-Q3_K_XL', 'unsloth'),
                                          ('q3-xxs', 'UD-IQ3_XXS', 'unsloth'),
                                          ('q4', 'UD-IQ4_XS', 'unsloth')]:
            with patch('sys.argv', ['server.py', 'start', '--quant', quant]), \
                    patch.object(server, 'gpu_busy', return_value=''), \
                    patch.object(server, 'run', return_value=''), \
                    patch.object(server, 'start_command', return_value=['docker', 'fake']) as command:
                server.main()
                cfg = command.call_args.args[1]
                self.assertEqual(cfg['name'], 'qwenflash-strata-' + quant)
                self.assertEqual(cfg['environment']['MODEL'], model_name)
                self.assertEqual(cfg['environment']['FAMILY'], family)
                self.assertEqual(cfg['environment']['CONTEXT'], '256k')
                self.assertEqual(cfg['environment']['PARALLEL'], '4')

    def test_busy_gpu_refuses_start(self):
        with patch('sys.argv', ['server.py', 'start']), \
                patch.object(server, 'gpu_busy', return_value='123 other-server 22000 MiB'), \
                patch.object(server, 'run') as run:
            with self.assertRaisesRegex(SystemExit, 'stop them explicitly'):
                server.main()
            run.assert_not_called()

    def test_http_completion_accounting(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.server.authorization = self.headers.get('Authorization')
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.server.body = body
                response = json.dumps({'choices': [{'message': {'content': '4'}, 'finish_reason': 'stop'}],
                                       'usage': {'completion_tokens': 1, 'prompt_tokens': 12},
                                       'timings': {'predicted_per_second': 100}}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(response)))
                self.end_headers()
                self.wfile.write(response)
            def log_message(self, *args):
                pass
        httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                env = Path(tmp) / '.env'
                env.write_text('API_KEY=test-key\n')
                client = bench.Client(f'http://127.0.0.1:{httpd.server_port}', env)
                result = client.complete('test-model', '2+2', 16)
                self.assertEqual(result['usage']['prompt_tokens'], 12)
                self.assertEqual(result['timings']['predicted_per_second'], 100)
                self.assertEqual(httpd.authorization, 'Bearer test-key')
                self.assertEqual(httpd.body['reasoning_effort'], 'none')
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join()

    def test_client_requires_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '.env'
            path.write_text('')
            with self.assertRaisesRegex(ValueError, 'API_KEY'):
                bench.Client('http://localhost:8000', path)


if __name__ == '__main__':
    unittest.main()
