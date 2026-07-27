"""Export a Marp Markdown deck to an image-backed, Google Slides-ready PPTX.

The renderer uses a local headless Firefox installation and an existing
Marp-generated PPTX as the OOXML shell. It intentionally avoids network access
and package installation. Speaker-note comments are retained in the PPTX.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import textwrap
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from markdown_it import MarkdownIt
from PIL import Image


SLIDE_WIDTH = 1280
SLIDE_HEIGHT = 720

BASE_CSS = """
html, body {
  margin: 0;
  padding: 0;
  width: 1280px;
  background: #000;
}
body { overflow-x: hidden; }
section {
  position: relative;
  display: flex;
  flex-flow: column nowrap;
  justify-content: center;
  width: 1280px;
  height: 720px;
  overflow: hidden;
}
section::after {
  content: attr(data-page);
  position: absolute;
  right: 66px;
  bottom: 22px;
}
section > :first-child { margin-top: 0; }
section > :last-child { margin-bottom: 0; }
section.image-slide > img.marp-bg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  margin: 0;
  object-fit: contain;
}
"""


@dataclass(frozen=True)
class Slide:
    html: str
    notes: str
    css_classes: str
    paginate: bool


def parse_front_matter(source: str) -> tuple[str, str]:
    match = re.match(r"\A---\n(.*?)\n---\n", source, flags=re.DOTALL)
    if not match:
        raise ValueError("The source does not start with Marp front matter")

    front_matter = match.group(1)
    style_match = re.search(r"^style:\s*\|\n(.*)\Z", front_matter, re.MULTILINE | re.DOTALL)
    css = textwrap.dedent(style_match.group(1)) if style_match else ""
    return css, source[match.end() :]


def extract_notes(markdown: str) -> str:
    match = re.search(
        r"<!--\s*SPEAKER NOTES\s+[—-]\s*(.*?)-->",
        markdown,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return ""
    return "SPEAKER NOTES — " + textwrap.dedent(match.group(1)).strip()


def parse_slides(source: str) -> tuple[str, list[Slide]]:
    css, deck_markdown = parse_front_matter(source)
    renderer = MarkdownIt("commonmark", {"html": True})
    slides: list[Slide] = []

    for raw_slide in deck_markdown.split("\n---\n"):
        class_match = re.search(r"<!--\s*_class:\s*([^>]+?)\s*-->", raw_slide)
        css_classes = class_match.group(1).strip() if class_match else ""
        paginate = not bool(
            re.search(r"<!--\s*_paginate:\s*false\s*-->", raw_slide, re.IGNORECASE)
        )
        notes = extract_notes(raw_slide)
        visible_markdown = re.sub(
            r"<!--\s*SPEAKER NOTES\s+[—-]\s*.*?-->",
            "",
            raw_slide,
            flags=re.DOTALL | re.IGNORECASE,
        )
        visible_markdown = re.sub(
            r"!\[bg\s+contain\]\(([^)]+)\)",
            r'<img class="marp-bg" src="\1" alt="">',
            visible_markdown,
        )
        slides.append(
            Slide(
                html=renderer.render(visible_markdown),
                notes=notes,
                css_classes=css_classes,
                paginate=paginate,
            )
        )

    return css, slides


def build_html(
    css: str, slides: list[Slide], asset_root: Path, page_start: int = 1
) -> str:
    sections = []
    for number, slide in enumerate(slides, start=page_start):
        page = str(number) if slide.paginate else ""
        sections.append(
            f'<section class="{escape(slide.css_classes)}" data-page="{page}">'
            f"{slide.html}</section>"
        )

    return """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1280, initial-scale=1">
<base href="%s">
<style>%s\n%s</style>
</head>
<body>%s</body>
</html>
""" % (asset_root.as_uri() + "/", BASE_CSS, css, "\n".join(sections))


def firefox_screenshot(firefox: Path, html_file: Path, output: Path, profile: Path) -> None:
    command = [
        str(firefox),
        "--headless",
        "--no-remote",
        "--window-size",
        f"{SLIDE_WIDTH},{SLIDE_HEIGHT}",
        "--screenshot",
        str(output),
        html_file.as_uri(),
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=45,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Firefox did not render {html_file.name} within 45 seconds"
        ) from exc
    if result.returncode != 0 or not output.exists():
        raise RuntimeError(
            f"Firefox screenshot failed for {html_file.name} (exit {result.returncode})"
        )


def render_slide_images(
    firefox: Path, css: str, slides: list[Slide], asset_root: Path, workdir: Path
) -> list[Path]:
    rendered: list[Path] = []
    for index, slide in enumerate(slides, start=1):
        html_file = workdir / f"slide-{index}.html"
        html_file.write_text(
            build_html(css, [slide], asset_root, page_start=index), encoding="utf-8"
        )
        slide_image = workdir / f"slide-{index}.png"
        firefox_screenshot(
            firefox, html_file, slide_image, workdir / "firefox-profile"
        )
        with Image.open(slide_image) as image:
            if image.size != (SLIDE_WIDTH, SLIDE_HEIGHT):
                image.resize((SLIDE_WIDTH, SLIDE_HEIGHT), Image.Resampling.LANCZOS).save(
                    slide_image, optimize=True
                )
        rendered.append(slide_image)
    return rendered


def notes_xml(notes: str, slide_number: int) -> bytes:
    note_text = escape(notes)
    field_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"marp-slide-{slide_number}"))
    creation_id = 1_000_000_000 + slide_number
    xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr><p:sp><p:nvSpPr><p:cNvPr id="2" name="Slide Image Placeholder 1"/><p:cNvSpPr><a:spLocks noGrp="1" noRot="1" noChangeAspect="1"/></p:cNvSpPr><p:nvPr><p:ph type="sldImg"/></p:nvPr></p:nvSpPr><p:spPr/></p:sp><p:sp><p:nvSpPr><p:cNvPr id="3" name="Notes Placeholder 2"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" dirty="0"/><a:t xml:space="preserve">{note_text}</a:t></a:r><a:endParaRPr lang="en-US" dirty="0"/></a:p></p:txBody></p:sp><p:sp><p:nvSpPr><p:cNvPr id="4" name="Slide Number Placeholder 3"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="sldNum" sz="quarter" idx="10"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:fld id="{{{field_id}}}" type="slidenum"><a:rPr lang="en-US"/><a:t>{slide_number}</a:t></a:fld><a:endParaRPr lang="en-US"/></a:p></p:txBody></p:sp></p:spTree><p:extLst><p:ext uri="{{BB962C8B-B14F-4D97-AF65-F5344CB8AC3E}}"><p14:creationId xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" val="{creation_id}"/></p:ext></p:extLst></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>'''
    return xml.encode("utf-8")


def create_pptx(
    template: Path,
    output: Path,
    slide_images: list[Path],
    slides: list[Slide],
    title: str,
) -> None:
    with zipfile.ZipFile(template) as archive:
        entries = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}

    slide_template = entries["ppt/slides/slide1.xml"].decode("utf-8")
    count = len(slides)

    for name in list(entries):
        if re.match(r"ppt/(?:slides/slide|notesSlides/notesSlide)\d+\.xml$", name):
            del entries[name]
        elif re.match(
            r"ppt/(?:slides/_rels/slide|notesSlides/_rels/notesSlide)\d+\.xml\.rels$",
            name,
        ):
            del entries[name]
        elif re.match(r"ppt/media/Slide-\d+-image-1\.png$", name):
            del entries[name]

    presentation = entries["ppt/presentation.xml"].decode("utf-8")
    slide_ids = "".join(
        f'<p:sldId id="{255 + number}" r:id="rId{number + 1}"/>'
        for number in range(1, count + 1)
    )
    presentation = re.sub(
        r"<p:sldIdLst>.*?</p:sldIdLst>",
        f"<p:sldIdLst>{slide_ids}</p:sldIdLst>",
        presentation,
        flags=re.DOTALL,
    )
    presentation = re.sub(
        r'<p:notesMasterIdLst>.*?</p:notesMasterIdLst>',
        f'<p:notesMasterIdLst><p:notesMasterId r:id="rId{count + 2}"/></p:notesMasterIdLst>',
        presentation,
        flags=re.DOTALL,
    )
    entries["ppt/presentation.xml"] = presentation.encode("utf-8")

    relationships = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    ]
    relationships.extend(
        f'<Relationship Id="rId{number + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{number}.xml"/>'
        for number in range(1, count + 1)
    )
    relationships.extend(
        [
            f'<Relationship Id="rId{count + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" Target="notesMasters/notesMaster1.xml"/>',
            f'<Relationship Id="rId{count + 3}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="presProps.xml"/>',
            f'<Relationship Id="rId{count + 4}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps" Target="viewProps.xml"/>',
            f'<Relationship Id="rId{count + 5}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>',
            f'<Relationship Id="rId{count + 6}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles" Target="tableStyles.xml"/>',
        ]
    )
    entries["ppt/_rels/presentation.xml.rels"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(relationships)
        + "</Relationships>"
    ).encode("utf-8")

    content_types = entries["[Content_Types].xml"].decode("utf-8")
    content_types = re.sub(
        r'<Override PartName="/ppt/(?:slides/slide|notesSlides/notesSlide)\d+\.xml"[^>]*/>',
        "",
        content_types,
    )
    overrides = "".join(
        f'<Override PartName="/ppt/slides/slide{number}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        f'<Override PartName="/ppt/notesSlides/notesSlide{number}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>'
        for number in range(1, count + 1)
    )
    entries["[Content_Types].xml"] = content_types.replace(
        "</Types>", overrides + "</Types>"
    ).encode("utf-8")

    app_xml = entries["docProps/app.xml"].decode("utf-8")
    app_xml = re.sub(r"<Slides>\d+</Slides>", f"<Slides>{count}</Slides>", app_xml)
    app_xml = re.sub(r"<Notes>\d+</Notes>", f"<Notes>{count}</Notes>", app_xml)
    app_xml = re.sub(
        r"(<vt:lpstr>Slide Titles</vt:lpstr>\s*</vt:variant>\s*<vt:variant><vt:i4>)\d+(</vt:i4>)",
        rf"\g<1>{count}\2",
        app_xml,
    )
    titles = "".join(f"<vt:lpstr>Slide {number}</vt:lpstr>" for number in range(1, count + 1))
    app_xml = re.sub(
        r'<vt:vector size="\d+" baseType="lpstr">.*?</vt:vector>',
        f'<vt:vector size="{count + 3}" baseType="lpstr"><vt:lpstr>Arial</vt:lpstr><vt:lpstr>Calibri</vt:lpstr><vt:lpstr>Office Theme</vt:lpstr>{titles}</vt:vector>',
        app_xml,
        flags=re.DOTALL,
    )
    entries["docProps/app.xml"] = app_xml.encode("utf-8")

    core_xml = entries["docProps/core.xml"].decode("utf-8")
    core_xml = re.sub(r"<dc:title>.*?</dc:title>", f"<dc:title>{escape(title)}</dc:title>", core_xml)
    entries["docProps/core.xml"] = core_xml.encode("utf-8")

    for number, (image, slide) in enumerate(zip(slide_images, slides, strict=True), start=1):
        entries[f"ppt/slides/slide{number}.xml"] = re.sub(
            r'name="Slide \d+"', f'name="Slide {number}"', slide_template
        ).encode("utf-8")
        entries[f"ppt/slides/_rels/slide{number}.xml.rels"] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/Slide-{number}-image-1.png"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
            f'<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide{number}.xml"/>'
            "</Relationships>"
        ).encode("utf-8")
        entries[f"ppt/notesSlides/notesSlide{number}.xml"] = notes_xml(
            slide.notes, number
        )
        entries[f"ppt/notesSlides/_rels/notesSlide{number}.xml.rels"] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" Target="../notesMasters/notesMaster1.xml"/>'
            f'<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="../slides/slide{number}.xml"/>'
            "</Relationships>"
        ).encode("utf-8")
        entries[f"ppt/media/Slide-{number}-image-1.png"] = image.read_bytes()

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(entries):
            archive.writestr(name, entries[name])


def deck_title(slides: list[Slide], fallback: str) -> str:
    if not slides:
        return fallback
    match = re.search(r"<h1>(.*?)</h1>", slides[0].html, flags=re.DOTALL)
    return re.sub(r"<[^>]+>", "", match.group(1)).strip() if match else fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument(
        "--firefox",
        type=Path,
        default=Path("/Applications/Firefox.app/Contents/MacOS/firefox"),
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    css, slides = parse_slides(source.read_text(encoding="utf-8"))
    if not slides:
        raise ValueError("No slides found")

    with tempfile.TemporaryDirectory(prefix="marp-pptx-") as temporary:
        workdir = Path(temporary)
        images = render_slide_images(args.firefox, css, slides, source.parent, workdir)
        create_pptx(
            args.template.resolve(),
            output,
            images,
            slides,
            deck_title(slides, source.stem),
        )

    print(f"Exported {len(slides)} slides to {output}")


if __name__ == "__main__":
    main()
