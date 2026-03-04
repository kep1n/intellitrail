"""Shared HTTP helpers for AEMET XML fetching.

This module provides a single :func:`fetch_xml` function used by both
``weather_alerts`` and ``weather_mountains`` to avoid code duplication.
"""

from __future__ import annotations

import logging
from xml.etree import ElementTree as ET

import httpx

logger = logging.getLogger(__name__)


def fetch_xml(url: str, client: httpx.Client, encoding: str = "utf-8") -> ET.Element:
    """Fetch a URL and parse the response body as XML.

    Args:
        url: The URL to fetch.
        client: An active ``httpx.Client`` to use for the request.
        encoding: Character encoding used when decoding the response body.
            Defaults to ``"utf-8"``.

    Returns:
        The parsed XML root element.

    Raises:
        httpx.HTTPStatusError: If the server returns a non-2xx status.
        xml.etree.ElementTree.ParseError: If the response body is not valid XML.
    """
    logger.debug("Fetching XML: %s", url)
    r = client.get(url, follow_redirects=True)
    r.raise_for_status()
    content = r.content.decode(encoding, errors="replace")
    return ET.fromstring(content)
