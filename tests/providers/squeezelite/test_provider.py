"""Tests for the squeezelite player provider."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aioslimproto.cli import SlimCLICommand
from music_assistant_models.enums import ContentType
from music_assistant_models.media_items import AudioFormat

from music_assistant.providers.squeezelite.provider import SqueezelitePlayerProvider

FLOW_FORMAT = AudioFormat(content_type=ContentType.PCM_F32LE, sample_rate=192000, bit_depth=32)


@pytest.mark.parametrize(
    ("fmt", "output_format", "expected_content_type"),
    [
        (
            "wav",
            AudioFormat(content_type=ContentType.WAV, sample_rate=192000, bit_depth=24),
            "audio/wav;rate=192000;bitrate=24;channels=2",
        ),
        (
            "pcm",
            AudioFormat(content_type=ContentType.PCM_S24LE, sample_rate=96000, bit_depth=24),
            "audio/pcm;codec=pcm;rate=96000;bitrate=24;channels=2",
        ),
    ],
)
async def test_multi_client_stream_content_type_describes_output_format(
    fmt: str, output_format: AudioFormat, expected_content_type: str
) -> None:
    """Squeezelite derives the PCM params from the Content-Type, so it must match the stream."""

    async def _stream(**_kwargs: Any) -> AsyncGenerator[bytes]:
        yield b"\x00" * 16

    stream = MagicMock(done=False, audio_format=FLOW_FORMAT, queue_id="q", session_id="s")
    stream.get_stream = _stream
    sync_parent = MagicMock(multi_client_stream=stream)
    child = MagicMock(display_name="Child")
    provider = MagicMock()
    provider.mass.players.get_player.side_effect = lambda pid: (
        sync_parent if pid == "leader" else child
    )
    provider.mass.streams.audio.get_output_format = AsyncMock(return_value=output_format)
    provider.mass.streams.audio.get_player_output_plan.return_value = MagicMock(filter_params=[])
    request = MagicMock(method="GET")
    request.query = {"player_id": "leader", "fmt": fmt, "child_player_id": "child"}
    resp = MagicMock(prepare=AsyncMock(), write=AsyncMock())

    with patch("aiohttp.web.StreamResponse", return_value=resp) as stream_response:
        await SqueezelitePlayerProvider._serve_multi_client_stream(provider, request)

    assert stream_response.call_args.kwargs["headers"]["Content-Type"] == expected_content_type


@pytest.mark.parametrize(
    ("command", "args", "expected_cmd"),
    [
        ("play", [], "cmd_play"),
        ("pause", ["0"], "cmd_play"),
        ("pause", ["1"], "cmd_pause"),
        ("pause", [], "cmd_play_pause"),
    ],
)
async def test_cli_play_pause_is_routed_through_player_controller(
    command: str, args: list[str], expected_cmd: str
) -> None:
    """Play/pause sent by a player must go through the controller, so an idle queue resumes."""
    provider = MagicMock()
    provider.mass.players = MagicMock(
        cmd_play=AsyncMock(), cmd_pause=AsyncMock(), cmd_play_pause=AsyncMock()
    )
    cli_command = SlimCLICommand(player_id="aa:bb", command=command, args=args, kwargs={})

    await SqueezelitePlayerProvider._handle_cli_command(provider, cli_command)

    getattr(provider.mass.players, expected_cmd).assert_awaited_once_with("aa:bb")
    for other in {"cmd_play", "cmd_pause", "cmd_play_pause"} - {expected_cmd}:
        getattr(provider.mass.players, other).assert_not_awaited()


@pytest.mark.parametrize(
    ("command", "player_known"),
    [("play", False), ("stop", True), ("button", True)],
)
async def test_cli_other_commands_are_left_to_aioslimproto(
    command: str, player_known: bool
) -> None:
    """Commands we do not handle, or for unknown players, fall back to aioslimproto."""
    provider = MagicMock()
    provider.slimproto.get_player.return_value = MagicMock() if player_known else None
    provider.mass.players = MagicMock(cmd_play=AsyncMock())
    cli_command = SlimCLICommand(player_id="aa:bb", command=command, args=[], kwargs={})

    with pytest.raises(NotImplementedError):
        await SqueezelitePlayerProvider._handle_cli_command(provider, cli_command)
    provider.mass.players.cmd_play.assert_not_awaited()
