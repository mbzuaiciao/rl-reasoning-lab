import subprocess

import pytest

from experiments.exp14_model_selection import Sandbox, repair


@pytest.mark.parametrize("daemon_up", [True, False])
def test_docker_disconnect_is_not_a_failed_case(monkeypatch, daemon_up):
    sandbox = Sandbox.__new__(Sandbox)
    sandbox.docker = ["docker", "--context", "desktop-linux"]
    monkeypatch.setattr(repair.Sandbox, "run", lambda *args: ("exit_1", b"worker failed"))

    def server_version(command, *, check, stdout, timeout):
        assert command[:4] == [*sandbox.docker, "version"]
        assert check and timeout == 10
        if not daemon_up:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", server_version)
    if daemon_up:
        assert sandbox.run("pass") == ("exit_1", b"worker failed")
    else:
        with pytest.raises(subprocess.CalledProcessError):
            sandbox.run("pass")


def test_connection_error_survives_daemon_recovery(monkeypatch):
    sandbox = Sandbox.__new__(Sandbox)
    monkeypatch.setattr(
        repair.Sandbox, "run", lambda *args: ("exit_1", b"failed to connect to the docker API")
    )
    with pytest.raises(RuntimeError, match="Docker daemon connection failed"):
        sandbox.run("pass")
