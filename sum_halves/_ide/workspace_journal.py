# 2026-04-17T09:21:43.164632152
import vitis

client = vitis.create_client()
client.set_workspace(path="sum_halves")

vitis.dispose()

