import asyncio
import re
import time
from wifall import config


async def init():
    print('initializing: {}'.format(config['init_cmd']))
    proc = await asyncio.create_subprocess_shell(config['init_cmd'])
    await proc.wait()


async def collect(ping=True, duration=0.5):
    file = f'dat/{int(time.time()*1000)}.dat'
    print(f'collecting: {file}')
    log_proc = await asyncio.create_subprocess_shell(f'{config["log_cmd"]} {file}')
    if ping:
        ping_proc = await asyncio.create_subprocess_shell(config['ping_cmd'])
    await asyncio.sleep(duration)
    log_proc.terminate()
    await log_proc.wait()
    if ping:
        ping_proc.terminate()
        await ping_proc.wait()
    return file
