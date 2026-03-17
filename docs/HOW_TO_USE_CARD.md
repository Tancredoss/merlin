# HOW_TO_USE_CARD

This guide explains how to use the reusable gallery card component on any documentation page.

## Where the component comes from

- Directive: `.. merlin-gallery::`
- Implemented in: `docs/source/_ext/merlin_gallery.py`
- Styled by: `docs/source/_static/css/style.css`

## Step 1: Create a JSON data file

Create a file like:

`docs/source/_data/galleries/my_page_cards.json`

Example:

```json
[
  {
    "title": "First Quantum Layers (Iris)",
    "summary": "Three practical ways to build and train QuantumLayer models on Iris.",
    "image": "_static/examples/iris_sepal_classes.png",
    "doc": "notebooks/FirstQuantumLayers",
    "tags": ["Classification", "Iris", "QuantumLayer"]
  },
  {
    "title": "MerLin GitHub",
    "summary": "Open the MerLin repository.",
    "image": "_static/img/merlin_black.png",
    "url": "https://github.com/merlinquantum/merlin",
    "tags": ["External"]
  }
]
```

## Step 2: Add the directive to any `.rst` page

```rst
.. merlin-gallery::
   :data: _data/galleries/my_page_cards.json
```

That is the minimum required usage.

## Optional directive options

```rst
.. merlin-gallery::
   :data: _data/galleries/my_page_cards.json
   :columns: 3
   :contour-color: #f2994a
   :extra-class: custom-gallery-hook
```

- `:columns:` can be `2`, `3`, or `4`
- `:contour-color:` sets a page-level contour color for all cards
- `:extra-class:` adds custom CSS class names on the gallery container

## Optional per-card contour override

Set `contour_color` on a specific card:

```json
{
  "title": "Card with custom contour",
  "summary": "Overrides the page contour for this card only.",
  "image": "_static/examples/iris_sepal_classes.png",
  "doc": "notebooks/FirstQuantumLayers",
  "contour_color": "#0ec8c3"
}
```

Priority:

1. Card `contour_color`
2. Page `:contour-color:`
3. Default (no colored contour)

## Rules for internal and external links

- Use `doc` for internal docs pages (without `.html`)
- Use `url` for external links
- Provide exactly one of `doc` or `url` per card

## Quick copy template

```json
[
  {
    "title": "My Card",
    "summary": "Short summary.",
    "image": "_static/examples/iris_sepal_classes.png",
    "doc": "notebooks/FirstQuantumLayers",
    "tags": ["Tag1", "Tag2"]
  }
]
```

```rst
.. merlin-gallery::
   :data: _data/galleries/my_page_cards.json
```
