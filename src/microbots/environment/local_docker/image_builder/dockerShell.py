import os
import logging
import asyncio
import signal
from datetime import datetime
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ShellCommunicator import ShellCommunicator

# Configure logging to see all logs including ShellCommunicator
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Ensure logs go to stdout
    ]
)

# Set specific logger levels
logging.getLogger('ShellCommunicator').setLevel(logging.DEBUG)
logging.getLogger('uvicorn').setLevel(logging.INFO)

logger = logging.getLogger(__name__)

shell = ShellCommunicator("bash")
shell.start_session()

# Inactivity timeout configuration
INACTIVITY_TIMEOUT_MINUTES = float(os.getenv("INACTIVITY_TIMEOUT_MINUTES", "60.0"))
last_activity_time = datetime.now()


class Message(BaseModel):
    message: str


async def check_inactivity():
    """Background task to check for inactivity and shutdown if timeout exceeded"""
    global last_activity_time

    # Check at 50% of timeout interval (with fudge factor) to ensure we catch timeouts reliably,
    # but cap at 60 seconds for long timeouts to avoid excessive polling
    check_interval = min(60, (INACTIVITY_TIMEOUT_MINUTES * 60.0) / 2)
    # Also ensure we never check more frequently than once every second
    check_interval = max(1, check_interval)

    while True:
        await asyncio.sleep(check_interval)

        idle_time = datetime.now() - last_activity_time
        idle_minutes = idle_time.total_seconds() / 60

        if idle_minutes >= INACTIVITY_TIMEOUT_MINUTES:
            logger.warning(
                f"🕐 No activity for {idle_minutes:.1f} minutes (timeout: {INACTIVITY_TIMEOUT_MINUTES}m). Shutting down..."
            )
            logger.info("Initiating graceful shutdown due to inactivity timeout")
            # Send SIGINT to trigger graceful shutdown. SIGINT is caught by uvicorn's signal handler
            # which performs proper cleanup (closing connections, running shutdown handlers, etc.)
            # before exiting. This is safer than SIGTERM or os.exit() which may skip cleanup.
            os.kill(os.getpid(), signal.SIGINT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events
    Will be invoked when the app starts and stops.
    """

    # Startup: start the inactivity checker which runs in background
    task = asyncio.create_task(check_inactivity())

    yield

    # Shutdown
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)


@app.post("/")
async def receive_message(message: Message):
    global last_activity_time
    last_activity_time = datetime.now()  # Update activity time on each request

    command_output = shell.send_command(message.message)
    return {"status": "success", "output": command_output}


if __name__ == "__main__":
    # Prefer BOT_PORT, else default 8080
    port = int(os.getenv("BOT_PORT") or 8080)
    uvicorn.run(app, host="0.0.0.0", port=port)
