"""RunPod entrypoint. Keep the synchronous Nano engine on its own thread."""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from nano.worker import NanoWorker


def main():
    import runpod
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nano-worker")
    worker = executor.submit(NanoWorker).result()

    async def handler(job):
        # The SDK uses asyncio; running the engine's sync event loop on that
        # same thread would cause "event loop already running" errors.
        return await asyncio.wrap_future(executor.submit(worker.run, job.get("input") or {}))

    try:
        runpod.serverless.start({"handler": handler})
    finally:
        executor.submit(worker.close).result()
        executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
