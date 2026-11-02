from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from time import sleep

import pytest
from fastapi.testclient import TestClient

from app.backends import FakeBackend
from app.batching import BatchScheduler, QueueFull, QueueTimeout
from app.main import create_app


class RecordingBackend(FakeBackend):
    def __init__(self):
        self.calls = []

    def generate(self, prompts, max_new_tokens):
        self.calls.append((list(prompts), max_new_tokens))
        return super().generate(prompts, max_new_tokens)


def test_two_clients_are_coalesced_and_results_remain_correct():
    backend = RecordingBackend()
    with TestClient(create_app(backend, batch_window_ms=100)) as client:
        gate = Barrier(3)

        def request(prompt):
            gate.wait()
            return client.post('/generate', json={'prompts': [prompt], 'max_new_tokens': 12})

        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(request, 'alpha')
            b = pool.submit(request, 'beta')
            gate.wait()
            responses = [a.result(), b.result()]
        assert all(r.status_code == 200 for r in responses)
        assert sorted(r.json()['results'][0]['text'] for r in responses) == [
            'Echo: alpha', 'Echo: beta']
        assert len(backend.calls) == 1
        assert len(backend.calls[0][0]) == 2
        assert client.get('/metrics').json()['batches_executed'] == 1


def test_incompatible_token_limits_do_not_share_a_model_call():
    backend = RecordingBackend()
    with TestClient(create_app(backend, batch_window_ms=50)) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(client.post, '/generate', json={
                'prompts': ['hi'], 'max_new_tokens': limit}) for limit in (8, 16)]
            assert all(f.result().status_code == 200 for f in futures)
        assert sorted(x[1] for x in backend.calls) == [8, 16]


def test_queue_limit_and_timeout():
    entered = Event()
    release = Event()

    class SlowBackend(FakeBackend):
        def generate(self, prompts, max_new_tokens):
            entered.set()
            release.wait(2)
            return super().generate(prompts, max_new_tokens)

    scheduler = BatchScheduler(SlowBackend(), max_queue_requests=1, batch_window_ms=0)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(scheduler.submit, ['busy'], 8, 2)
            assert entered.wait(1)
            second = pool.submit(scheduler.submit, ['waiting'], 8, 0.05)
            sleep(0.01)
            with pytest.raises(QueueFull):
                scheduler.submit(['overflow'], 8, 0.1)
            with pytest.raises(QueueTimeout):
                second.result()
            release.set()
            assert first.result()[0].text == 'Echo: busy'
    finally:
        release.set()
        scheduler.close()


def test_model_failure_propagates_to_each_client():
    class FailingBackend(FakeBackend):
        def generate(self, prompts, max_new_tokens):
            raise RuntimeError('GPU failure')

    with TestClient(create_app(FailingBackend(), batch_window_ms=10)) as client:
        response = client.post('/generate', json={'prompts': ['hi']})
        assert response.status_code == 500
        assert client.get('/metrics').json()['total_failures'] == 1
