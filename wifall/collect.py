import asyncio
import time
from wifall import config


async def init():
    print('initializing: {}'.format(config['init_cmd']))
    proc = await asyncio.create_subprocess_shell(config['init_cmd'])
    await proc.wait()


async def collect(ping=True, duration=0.5):
    file = f'dat/{int(time.time()*1000)}.dat'
    print(f'collecting: {file}')
    log_proc = await asyncio.create_subprocess_exec(config["log_cmd"], file)
    if ping:
        ping_proc = await asyncio.create_subprocess_exec('ping', *config['ping_cmd'].split(' '))
    await asyncio.sleep(duration)
    if log_proc.returncode is None:
        log_proc.kill()
    if ping and ping_proc.returncode is None:
        ping_proc.kill()
        await ping_proc.wait()
    await log_proc.wait()
    return file
