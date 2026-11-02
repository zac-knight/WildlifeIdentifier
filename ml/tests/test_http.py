import httpx
import pytest

from wildlens.http import PoliteClient, Throttle


def client_for(handler, tmp_path) -> PoliteClient:
    return PoliteClient(tmp_path, interval=0, transport=httpx.MockTransport(handler))


def test_second_request_comes_from_cache(tmp_path) -> None:
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"ok": 1})

    client = client_for(handler, tmp_path)
    assert client.get_json("https://x.test/a", {"q": 1}) == {"ok": 1}
    assert client.get_json("https://x.test/a", {"q": 1}) == {"ok": 1}
    assert len(calls) == 1


def test_user_agent_identifies_the_project(tmp_path) -> None:
    seen = {}

    def handler(request):
        seen["agent"] = request.headers["user-agent"]
        return httpx.Response(200, json={})

    client_for(handler, tmp_path).get_json("https://x.test/a")
    assert seen["agent"].startswith("WildLens/")


def test_client_error_is_not_retried(tmp_path) -> None:
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        client_for(handler, tmp_path).get_json("https://x.test/a")
    assert len(calls) == 1


def test_throttle_spaces_calls() -> None:
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    throttle = Throttle(1.0, clock=lambda: now[0], sleep=sleep)
    throttle.wait()
    throttle.wait()
    assert sleeps == [1.0]
