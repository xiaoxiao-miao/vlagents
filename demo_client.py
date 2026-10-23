"""VLAgents 0.2.0 client smoke test. Run in conda env `rpd` after the openvla server is up."""

from time import sleep

import numpy as np
from vlagents.client import RemoteAgent
from vlagents.policies import Obs

# on_same_machine=True 会原地把 cameras 改成 SharedMemoryPayload，
# 所以每次 reset/act 都要传新的 Obs（不要复用同一份）。
agent = RemoteAgent("localhost", 8080, "openvla", on_same_machine=True)
with agent:
    while not agent.is_initialized():
        sleep(0.5)
    # OpenVLA 0.2.0 expects rgb_side at 256x256
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    agent.reset(Obs(cameras={"rgb_side": img.copy()}), "pick up the cube")
    print(agent.act(Obs(cameras={"rgb_side": img.copy()})))
