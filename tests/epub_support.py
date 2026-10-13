"""Programmatic EPUB fixtures with original Russian text (no downloaded books)."""

from __future__ import annotations

import zipfile
from pathlib import Path

CONTAINER = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
</container>
"""

NS = 'xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"'


def xhtml(title: str, body: str) -> str:
    return (
        f'<?xml version="1.0" encoding="utf-8"?>\n<html {NS}><head><title>{title}</title></head>'
        f"<body>{body}</body></html>\n"
    )


COVER = xhtml("Обложка", '<div><img src="images/map.png" alt="Обложка книги"/></div>')
TITLEPAGE = xhtml(
    "Титул",
    '<section epub:type="titlepage"><h1>Тестовая книга</h1><p>Тестовый Автор</p></section>',
)
CONTENTS = xhtml(
    "Содержание",
    "<h1>Содержание</h1><ul>"
    '<li><a href="chapter1.xhtml">Глава 1. Утро</a></li>'
    '<li><a href="chapter2.xhtml">Глава 2. Дорога</a></li>'
    '<li><a href="split.xhtml#part1">Часть первая</a></li>'
    '<li><a href="split.xhtml#part2">Часть вторая</a></li>'
    '<li><a href="chapter3.xhtml">Глава 3. Город</a></li>'
    "</ul>",
)
CHAPTER1 = xhtml(
    "Глава 1",
    "<h1>Глава 1. Утро</h1>"
    '<p>Утро началось рано. Всё ещё спало. Её мать<a epub:type="noteref" href="#fn1">1</a> '
    "уже ходила по кухне.</p>"
    '<aside epub:type="footnote" id="fn1"><p>Имеется в виду старая мельница.</p></aside>'
    "<hr/>"
    "<p>— Пора, — сказала она сама себе.</p>"
    "<p>***</p>"
    "<p>Вечер пришёл неожиданно.</p>",
)
CHAPTER2 = xhtml(
    "Глава 2",
    "<h1>Глава 2. Дорога</h1>"
    "<p>Дорога шла через&nbsp;лес&shy;.</p>"
    '<p><img src="images/map.png" alt="Карта дороги"/></p>'
    "<p>Поезд опоздал.",
)
SPLIT = xhtml(
    "Части",
    '<h2 id="part1">Часть первая</h2><p>Первая часть книги начинается здесь.</p>'
    '<h2 id="part2">Часть вторая</h2><p>Вторая часть книги продолжается.</p>',
)
CHAPTER3 = xhtml(
    "Глава 3",
    "<h1>Глава 3. Город</h1>"
    '<p>Город спал<a href="notes.xhtml#n1"><sup>1</sup></a>, а река шумела.</p>'
    "<p>* * *</p>"
    "<p>Вечер наступил.</p>"
    "<hr/>"
    "<p>Конец главы.</p>",
)
NOTES = xhtml("Примечания", '<h1>Примечания</h1><p id="n1">Река называлась Смородина.</p>')
EXTRA = xhtml("Дополнение", "<h1>Дополнение</h1><p>Дополнительный текст вне порядка чтения.</p>")
BROKEN = xhtml("Сломано", "<h1>Сломанная глава</h1><p>Текст\xff\xfe</p>")

NAV = xhtml(
    "Навигация",
    '<nav epub:type="toc" id="toc"><ol>'
    '<li><a href="chapter1.xhtml">Глава 1. Утро</a></li>'
    '<li><a href="chapter2.xhtml">Глава 2. Дорога</a></li>'
    '<li><a href="split.xhtml#part1">Часть первая</a></li>'
    '<li><a href="split.xhtml#part2">Часть вторая</a></li>'
    "</ol></nav>",
)
NCX_POINTS = [
    ("Глава 1. Утро", "chapter1.xhtml"),
    ("Глава 2. Дорога", "chapter2.xhtml"),
    ("Часть первая", "split.xhtml#part1"),
    ("Часть вторая", "split.xhtml#part2"),
]
NCX = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><navMap>'
    + "".join(
        f'<navPoint id="np{index}" playOrder="{index}"><navLabel><text>{label}</text></navLabel>'
        f'<content src="{src}"/></navPoint>'
        for index, (label, src) in enumerate(NCX_POINTS, start=1)
    )
    + "</navMap></ncx>\n"
)

PNG_STUB = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _opf(*, nav: bool, ncx: bool, broken: bool) -> str:
    manifest_items = [
        ("cover", "cover.xhtml", "application/xhtml+xml", ' properties="cover-image"'),
        ("titlepage", "titlepage.xhtml", "application/xhtml+xml", ""),
        ("contents", "contents.xhtml", "application/xhtml+xml", ""),
        ("chapter1", "chapter1.xhtml", "application/xhtml+xml", ""),
        ("chapter2", "chapter2.xhtml", "application/xhtml+xml", ""),
        ("split", "split.xhtml", "application/xhtml+xml", ""),
        ("chapter3", "chapter3.xhtml", "application/xhtml+xml", ""),
        ("extra", "extra.xhtml", "application/xhtml+xml", ""),
        ("notes", "notes.xhtml", "application/xhtml+xml", ""),
        ("map", "images/map.png", "image/png", ""),
    ]
    if broken:
        manifest_items.insert(-1, ("broken", "broken.xhtml", "application/xhtml+xml", ""))
    if nav:
        manifest_items.insert(0, ("nav", "nav.xhtml", "application/xhtml+xml", ' properties="nav"'))
    if ncx:
        manifest_items.insert(0 if not nav else 1, ("ncx", "toc.ncx", "application/x-dtd-ncx", ""))
    manifest = "".join(
        f'<item id="{item_id}" href="{href}" media-type="{media}"{props}/>'
        for item_id, href, media, props in manifest_items
    )
    spine_ids: list[tuple[str, str]] = [
        ("cover", "yes"),
        ("titlepage", "yes"),
        ("contents", "yes"),
        ("chapter2", "yes"),
        ("chapter1", "yes"),
        ("split", "yes"),
        ("chapter3", "yes"),
        ("extra", "no"),
        ("notes", "yes"),
    ]
    if broken:
        spine_ids.insert(-1, ("broken", "yes"))
    spine = "".join(
        f'<itemref idref="{item_id}"' + ("" if linear == "yes" else ' linear="no"') + "/>"
        for item_id, linear in spine_ids
    )
    toc_attribute = ' toc="ncx"' if ncx else ""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        '<dc:identifier id="id">urn:test:book</dc:identifier>'
        "<dc:title>Тестовая книга</dc:title><dc:creator>Тестовый Автор</dc:creator>"
        "<dc:language>ru</dc:language></metadata>"
        f"<manifest>{manifest}</manifest><spine{toc_attribute}>{spine}</spine></package>\n"
    )


def write_epub(
    path: Path,
    *,
    nav: bool = True,
    ncx: bool = True,
    drm: bool = False,
    broken: bool = False,
    members: dict[str, bytes] | None = None,
) -> Path:
    """Write a fixture EPUB. ``broken`` adds a spine document that is not valid UTF-8."""
    documents: dict[str, str] = {
        "cover.xhtml": COVER,
        "titlepage.xhtml": TITLEPAGE,
        "contents.xhtml": CONTENTS,
        "chapter1.xhtml": CHAPTER1,
        "chapter2.xhtml": CHAPTER2,
        "split.xhtml": SPLIT,
        "chapter3.xhtml": CHAPTER3,
        "extra.xhtml": EXTRA,
        "notes.xhtml": NOTES,
    }
    if nav:
        documents["nav.xhtml"] = NAV
    entries: dict[str, bytes] = {
        "mimetype": b"application/epub+zip",
        "META-INF/container.xml": CONTAINER.encode("utf-8"),
        "OEBPS/content.opf": _opf(nav=nav, ncx=ncx, broken=broken).encode("utf-8"),
        "OEBPS/images/map.png": PNG_STUB,
    }
    if ncx:
        entries["OEBPS/toc.ncx"] = NCX.encode("utf-8")
    for name, markup in documents.items():
        entries[f"OEBPS/{name}"] = markup.encode("utf-8")
    if broken:
        entries["OEBPS/broken.xhtml"] = b"<html><body><p>\xff\xfe</p></body></html>"
    if drm:
        entries["META-INF/encryption.xml"] = (
            '<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
            'xmlns:enc="http://www.w3.org/2001/04/xmlenc#">'
            '<enc:EncryptedData><enc:CipherData><enc:CipherReference URI="OEBPS/chapter1.xhtml"/>'
            "</enc:CipherData></enc:EncryptedData></encryption>"
        ).encode("utf-8")
    entries.update(members or {})
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def write_minimal_epub(path: Path, body: str) -> Path:
    """A one-document EPUB whose only spine item holds ``body``."""
    entries = {
        "META-INF/container.xml": CONTAINER.encode("utf-8"),
        "OEBPS/content.opf": (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Minimal</dc:title></metadata>'
            '<manifest><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/></manifest>'
            '<spine><itemref idref="c1"/></spine></package>'
        ).encode("utf-8"),
        "OEBPS/c1.xhtml": xhtml("Текст", body).encode("utf-8"),
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path
