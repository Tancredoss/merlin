"""Reusable gallery cards for MerLin Sphinx documentation."""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path
from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.errors import NoUri
from sphinx.util import logging
from sphinx.util.osutil import relative_uri

logger = logging.getLogger(__name__)

_ALLOWED_COLUMNS = {2, 3, 4}
_COLOR_PATTERN = re.compile(r"^[A-Za-z0-9#(),.%\s-]+$")


class MerlinGalleryNode(nodes.General, nodes.Element):
    """Docutils node containing gallery configuration and card data."""


def _sanitize_color(value: str) -> str | None:
    candidate = value.strip()
    if not candidate:
        return None
    if ";" in candidate or ":" in candidate:
        return None
    if not _COLOR_PATTERN.fullmatch(candidate):
        return None
    return candidate


def _normalize_docname(value: str) -> str:
    normalized = value.strip().replace("\\", "/").lstrip("/")
    for suffix in (".rst", ".md", ".ipynb"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _format_attrs(attrs: dict[str, str | None]) -> str:
    parts: list[str] = []
    for key, value in attrs.items():
        if value:
            parts.append(f'{key}="{escape(value, quote=True)}"')
    return " ".join(parts)


class MerlinGalleryDirective(Directive):
    has_content = False
    option_spec = {
        "data": directives.path,
        "columns": directives.nonnegative_int,
        "contour-color": directives.unchanged,
        "extra-class": directives.class_option,
    }

    def run(self) -> list[nodes.Node]:
        env = self.state.document.settings.env
        raw_data_path = self.options.get("data")
        if not raw_data_path:
            raise self.error(
                "The ':data:' option is required for '.. merlin-gallery::'."
            )

        data_path = self._resolve_data_path(raw_data_path)
        if not data_path.exists():
            raise self.error(
                f"Gallery data file '{raw_data_path}' was not found "
                f"(resolved to '{data_path}')."
            )
        env.note_dependency(str(data_path))

        try:
            raw_cards = json.loads(data_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise self.error(f"Invalid JSON in '{data_path}': {exc}") from exc

        if not isinstance(raw_cards, list):
            raise self.error(
                f"Gallery JSON '{data_path}' must be a list of card objects."
            )

        cards: list[dict[str, Any]] = []
        for idx, raw_card in enumerate(raw_cards, start=1):
            card = self._validate_card(raw_card, idx, data_path)
            if card:
                cards.append(card)

        columns = self.options.get("columns")
        if columns is not None and columns not in _ALLOWED_COLUMNS:
            logger.warning(
                "merlin-gallery: ':columns:' only accepts 2, 3, or 4. "
                "Falling back to responsive auto-fit.",
                location=(env.docname, self.lineno),
            )
            columns = None

        page_contour = None
        raw_page_contour = self.options.get("contour-color")
        if raw_page_contour:
            page_contour = _sanitize_color(raw_page_contour)
            if not page_contour:
                logger.warning(
                    "merlin-gallery: ignored invalid ':contour-color:' value '%s'.",
                    raw_page_contour,
                    location=(env.docname, self.lineno),
                )

        gallery_node = MerlinGalleryNode()
        gallery_node["cards"] = cards
        gallery_node["columns"] = columns
        gallery_node["page_contour"] = page_contour
        gallery_node["extra_classes"] = self.options.get("extra-class", [])
        return [gallery_node]

    def _resolve_data_path(self, raw_path: str) -> Path:
        env = self.state.document.settings.env
        srcdir = Path(env.srcdir)
        normalized = raw_path.replace("\\", "/")

        if normalized.startswith("/"):
            return srcdir / normalized.lstrip("/")

        doc_dir = srcdir / Path(env.docname).parent
        candidate = doc_dir / normalized
        if candidate.exists():
            return candidate
        return srcdir / normalized

    def _validate_card(
        self,
        raw_card: Any,
        index: int,
        data_path: Path,
    ) -> dict[str, Any] | None:
        env = self.state.document.settings.env
        prefix = f"merlin-gallery: skipped card #{index} in '{data_path.name}' because"

        if not isinstance(raw_card, dict):
            logger.warning(
                "%s it is not a JSON object.",
                prefix,
                location=(env.docname, self.lineno),
            )
            return None

        title = str(raw_card.get("title", "")).strip()
        summary = str(raw_card.get("summary", "")).strip()
        image = str(raw_card.get("image", "")).strip().replace("\\", "/").lstrip("/")

        if not title:
            logger.warning(
                "%s 'title' is missing.", prefix, location=(env.docname, self.lineno)
            )
            return None
        if not summary:
            logger.warning(
                "%s 'summary' is missing.", prefix, location=(env.docname, self.lineno)
            )
            return None
        if not image:
            logger.warning(
                "%s 'image' is missing.", prefix, location=(env.docname, self.lineno)
            )
            return None

        has_doc = bool(str(raw_card.get("doc", "")).strip())
        has_url = bool(str(raw_card.get("url", "")).strip())
        if has_doc == has_url:
            logger.warning(
                "%s it must define exactly one of 'doc' or 'url'.",
                prefix,
                location=(env.docname, self.lineno),
            )
            return None

        card: dict[str, Any] = {
            "title": title,
            "summary": summary,
            "image": image,
        }

        image_path = Path(env.srcdir) / image
        if not image_path.exists():
            logger.warning(
                "merlin-gallery: card #%d image '%s' does not exist in docs source.",
                index,
                image,
                location=(env.docname, self.lineno),
            )

        if has_doc:
            docname = _normalize_docname(str(raw_card["doc"]))
            card["kind"] = "doc"
            card["docname"] = docname
            if docname not in env.found_docs:
                logger.warning(
                    "merlin-gallery: card #%d references missing doc '%s'.",
                    index,
                    docname,
                    location=(env.docname, self.lineno),
                )
        else:
            card["kind"] = "url"
            card["url"] = str(raw_card["url"]).strip()

        tags = raw_card.get("tags", [])
        if tags is None:
            tags = []
        if not isinstance(tags, list):
            tags = [tags]
        card["tags"] = [str(tag).strip() for tag in tags if str(tag).strip()]

        contour_color = raw_card.get("contour_color")
        if contour_color:
            safe_contour = _sanitize_color(str(contour_color))
            if safe_contour:
                card["contour_color"] = safe_contour
            else:
                logger.warning(
                    "merlin-gallery: ignored invalid contour_color '%s' for card #%d.",
                    contour_color,
                    index,
                    location=(env.docname, self.lineno),
                )

        return card


def _resolve_link(
    translator: Any, from_docname: str, card: dict[str, Any]
) -> tuple[str, bool]:
    if card["kind"] == "doc":
        try:
            return translator.builder.get_relative_uri(
                from_docname, card["docname"]
            ), False
        except NoUri:
            return "#", False
    return card["url"], True


def visit_merlin_gallery_node_html(translator: Any, node: MerlinGalleryNode) -> None:
    from_docname = translator.builder.current_docname
    current_uri = translator.builder.get_target_uri(from_docname)

    classes = ["mq-gallery-grid", *node.get("extra_classes", [])]
    grid_styles: list[str] = []
    if node.get("page_contour"):
        grid_styles.append(f"--mq-page-contour: {node['page_contour']}")
    if node.get("columns"):
        grid_styles.append(
            f"grid-template-columns: repeat({int(node['columns'])}, minmax(0, 1fr))"
        )

    translator.body.append(
        f"<div {_format_attrs({'class': ' '.join(classes), 'style': '; '.join(grid_styles)})}>"
    )

    for card in node.get("cards", []):
        href, is_external = _resolve_link(translator, from_docname, card)
        image_uri = relative_uri(current_uri, card["image"])
        card_styles: list[str] = []
        if card.get("contour_color"):
            card_styles.append(f"--mq-card-contour: {card['contour_color']}")

        card_attrs: dict[str, str | None] = {
            "class": "mq-gallery-card",
            "href": href,
            "style": "; ".join(card_styles),
        }
        if is_external:
            card_attrs["target"] = "_blank"
            card_attrs["rel"] = "noopener noreferrer"

        translator.body.append(f"<a {_format_attrs(card_attrs)}>")
        translator.body.append(
            f'<div class="mq-gallery-card-media"><img src="{escape(image_uri, quote=True)}" '
            f'alt="{escape(card["title"], quote=True)}"></div>'
        )
        translator.body.append('<div class="mq-gallery-card-body">')
        translator.body.append(
            f'<p class="mq-gallery-card-title">{escape(card["title"])}</p>'
        )
        translator.body.append(
            f'<p class="mq-gallery-card-summary">{escape(card["summary"])}</p>'
        )

        tags = card.get("tags", [])
        if tags:
            translator.body.append('<div class="mq-gallery-card-tags">')
            for tag in tags:
                translator.body.append(
                    f'<span class="mq-gallery-card-tag">{escape(str(tag))}</span>'
                )
            translator.body.append("</div>")

        translator.body.append("</div>")
        translator.body.append("</a>")

    translator.body.append("</div>")
    raise nodes.SkipNode


def depart_merlin_gallery_node_html(translator: Any, node: MerlinGalleryNode) -> None:
    del translator, node


def visit_merlin_gallery_node_unsupported(
    translator: Any, node: MerlinGalleryNode
) -> None:
    del translator, node
    raise nodes.SkipNode


def depart_merlin_gallery_node_unsupported(
    translator: Any, node: MerlinGalleryNode
) -> None:
    del translator, node


def setup(app: Any) -> dict[str, Any]:
    app.add_directive("merlin-gallery", MerlinGalleryDirective)
    app.add_node(
        MerlinGalleryNode,
        html=(visit_merlin_gallery_node_html, depart_merlin_gallery_node_html),
        latex=(
            visit_merlin_gallery_node_unsupported,
            depart_merlin_gallery_node_unsupported,
        ),
        text=(
            visit_merlin_gallery_node_unsupported,
            depart_merlin_gallery_node_unsupported,
        ),
        man=(
            visit_merlin_gallery_node_unsupported,
            depart_merlin_gallery_node_unsupported,
        ),
        texinfo=(
            visit_merlin_gallery_node_unsupported,
            depart_merlin_gallery_node_unsupported,
        ),
    )
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
