"""Bounded, single-worker inference scheduler with opportunistic dynamic batching.

Requests are grouped only when max_new_tokens matches. One worker owns model
execution, avoiding simultaneous calls into a potentially non-threadsafe backend.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass
from threading import Condition, Thread
from time import monotonic
from typing import Callable

from app.backends import Generated, InferenceBackend


class QueueFull(Exception):
    pass


class QueueTimeout(Exception):
    pass


class SchedulerClosed(Exception):
    pass


@dataclass
class Job:
    prompts: list[str]
    max_new_tokens: int
    future: Future[list[Generated]]
    enqueued_at: float


class BatchScheduler:
    def __init__(self, backend: InferenceBackend, *, max_batch_prompts: int = 8,
                 max_queue_requests: int = 64, batch_window_ms: float = 20,
                 on_batch: Callable[[int, float], None] | None = None):
        if max_batch_prompts < 1 or max_queue_requests < 1 or batch_window_ms < 0:
            raise ValueError("invalid scheduler settings")
        self.backend = backend
        self.max_batch_prompts = max_batch_prompts
        self.max_queue_requests = max_queue_requests
        self.window_seconds = batch_window_ms / 1000
        self.on_batch = on_batch
        self.cv = Condition()
        self.queue: deque[Job] = deque()
        self.closed = False
        self.worker = Thread(target=self._run, daemon=True, name="inference-batcher")
        self.worker.start()

    def submit(self, prompts: list[str], max_new_tokens: int,
               timeout_seconds: float = 30) -> list[Generated]:
        if len(prompts) > self.max_batch_prompts:
            raise QueueFull("request exceeds configured batch size")
        future: Future[list[Generated]] = Future()
        with self.cv:
            if self.closed:
                raise SchedulerClosed()
            if len(self.queue) >= self.max_queue_requests:
                raise QueueFull("inference request queue full")
            self.queue.append(Job(prompts, max_new_tokens, future, monotonic()))
            self.cv.notify()
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()  # Skips queued work; running GPU computation cannot be preempted.
            raise QueueTimeout("inference request timed out") from exc

    def snapshot(self) -> dict[str, int]:
        with self.cv:
            return {"queue_depth": len(self.queue), "max_queue_requests": self.max_queue_requests}

    def close(self) -> None:
        with self.cv:
            self.closed = True
            while self.queue:
                job = self.queue.popleft()
                if not job.future.done():
                    job.future.set_exception(SchedulerClosed())
            self.cv.notify_all()
        self.worker.join(timeout=5)

    def _run(self) -> None:
        while True:
            with self.cv:
                while not self.queue and not self.closed:
                    self.cv.wait()
                if self.closed:
                    return
                first = self.queue.popleft()
                if first.future.cancelled():
                    continue
                batch = [first]
                size = len(first.prompts)
                deadline = first.enqueued_at + self.window_seconds
                while size < self.max_batch_prompts and not self.closed:
                    self.queue = deque(j for j in self.queue if not j.future.cancelled())
                    match = next((i for i, j in enumerate(self.queue)
                                  if j.max_new_tokens == first.max_new_tokens
                                  and size + len(j.prompts) <= self.max_batch_prompts), None)
                    if match is not None:
                        job = self.queue[match]
                        del self.queue[match]
                        batch.append(job)
                        size += len(job.prompts)
                        continue
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    self.cv.wait(timeout=remaining)
            prompts = [prompt for job in batch for prompt in job.prompts]
            started = monotonic()
            try:
                results = self.backend.generate(prompts, first.max_new_tokens)
                if len(results) != len(prompts):
                    raise RuntimeError("model returned incorrect result count")
                offset = 0
                for job in batch:
                    selected = results[offset: offset + len(job.prompts)]
                    offset += len(job.prompts)
                    if not job.future.done():
                        job.future.set_result(selected)
            except Exception as exc:
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(exc)
            finally:
                if self.on_batch:
                    self.on_batch(len(prompts), (monotonic() - started) * 1000)
