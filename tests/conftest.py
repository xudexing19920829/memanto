from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_auto_parse(monkeypatch):
    """Ensure tests are not affected by the local smart_parse config setting."""
    from memanto.app.config import settings

    monkeypatch.setattr(settings, "AUTO_PARSE_ENABLED", True)


@pytest.fixture(autouse=True)
def mock_moorcheh_for_tests():
    """Prevent tests from calling real Moorcheh APIs.

    Patches the backend-aware dispatcher and the cloud SDK class used inside
    ``moorcheh.py``; everything that goes through ``get_moorcheh_client`` or
    creates a cloud client lands on the same mock.
    """
    mock_instance = MagicMock()
    mock_instance.namespaces.create.return_value = {"status": "created"}
    mock_instance.namespaces.list.return_value = {"namespaces": []}

    with (
        patch(
            "memanto.app.services.agent_service.get_moorcheh_client",
            return_value=mock_instance,
        ),
        patch(
            "memanto.app.clients.moorcheh.MoorchehClient",
            return_value=mock_instance,
        ),
    ):
        yield mock_instance
