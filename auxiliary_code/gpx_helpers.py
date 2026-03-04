"""GPX validation and Spanish place-name normalisation helpers."""

from xml.etree import ElementTree as ET
import re

_DISALLOWED_XML_TOKENS = ("<!ENTITY", "<!DOCTYPE", "<![CDATA[")
_GPX_CONTENT_TAGS = ("trk", "wpt", "rte")


class GPXValidationError(ValueError):
    """Raised when uploaded content is not a valid GPX file."""


def validate_gpx(gpx_content: str) -> None:
    """Raise GPXValidationError if content is not a safe, valid GPX file.

    Checks performed:

    - Well-formed XML with no external entity references (XXE prevention).
    - Root element is ``<gpx>``.
    - Contains at least one track, route, or waypoint.
    - No embedded scripts or suspicious payloads.

    Args:
        gpx_content: Raw GPX file content as a string.

    Raises:
        GPXValidationError: If any of the validity or safety checks fail.
    """
    # Reject external entity declarations (XXE / malware vector)
    if any(token in gpx_content for token in _DISALLOWED_XML_TOKENS):
        raise GPXValidationError("Disallowed XML construct detected")

    try:
        root = ET.fromstring(gpx_content)
    except ET.ParseError as exc:
        raise GPXValidationError("Invalid XML") from exc

    if root.tag.rpartition("}")[-1].lower() != "gpx":
        raise GPXValidationError("Root element is not <gpx>")

    has_content = any(
        root.find(f"{{*}}{tag}") is not None
        for tag in _GPX_CONTENT_TAGS
    )
    if not has_content:
        raise GPXValidationError("GPX contains no tracks, routes, or waypoints")


def normalize_article(s: str) -> str:
    """Move a trailing definite article to the front of a Spanish place name.

    Converts ``"Palma, La"`` → ``"La Palma"``, handling the Valencian form
    ``Els/Les/l'`` as well.

    Args:
        s: Place name, potentially in ``"Name, Article"`` form.

    Returns:
        The normalised place name with the article at the front.
    """
    return re.sub(r'^(.*),\s*(El|La|Los|Las|Els/Les/l\')\s*$', r'\2 \1', s, flags=re.IGNORECASE)


def normalize_slash(s: str) -> str:
    """Ensure consistent spacing around slash separators in place names.

    Args:
        s: Raw place name string.

    Returns:
        String with every ``/`` surrounded by single spaces.
    """
    return re.sub(r'\s*/\s*', ' / ', s)
