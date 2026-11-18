import os
import socket
import sys
from pathlib import Path
import pytest

os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args,**kwargs):
        raise AssertionError("Network forbidden during offline tests")
    monkeypatch.setattr(socket.socket,"connect",blocked)
    monkeypatch.setattr(socket.socket,"connect_ex",blocked)
    monkeypatch.setattr(socket,"create_connection",blocked)
