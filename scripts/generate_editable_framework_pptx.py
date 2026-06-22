from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "260615最优方案算法框架图中文版_可编辑PPT.pptx"

SLIDE_W = 12192000
SLIDE_H = 6858000
SRC_W = 1800
SRC_H = 1120


def emu_x(x: float) -> int:
    return round(x / SRC_W * SLIDE_W)


def emu_y(y: float) -> int:
    return round(y / SRC_H * SLIDE_H)


def emu_w(w: float) -> int:
    return round(w / SRC_W * SLIDE_W)


def emu_h(h: float) -> int:
    return round(h / SRC_H * SLIDE_H)


def color(hex_color: str) -> str:
    return hex_color.replace("#", "").upper()


def esc(text: str) -> str:
    return escape(text)


class SlideBuilder:
    def __init__(self):
        self._id = 2
        self.items: list[str] = []

    def next_id(self) -> int:
        self._id += 1
        return self._id

    def text_box(
        self,
        x,
        y,
        w,
        h,
        lines,
        font_size=16,
        bold=False,
        fill="FFFFFF",
        stroke="FFFFFF",
        text_color="0F172A",
        radius=True,
        align="l",
        name="TextBox",
    ):
        self.shape(
            x,
            y,
            w,
            h,
            lines=lines,
            font_size=font_size,
            bold=bold,
            fill=fill,
            stroke=stroke,
            text_color=text_color,
            radius=radius,
            align=align,
            name=name,
        )

    def shape(
        self,
        x,
        y,
        w,
        h,
        fill="FFFFFF",
        stroke="CBD5E1",
        stroke_width=1.5,
        prst="roundRect",
        lines=None,
        font_size=16,
        bold=False,
        text_color="0F172A",
        radius=True,
        align="l",
        name="Shape",
        no_fill=False,
    ):
        sid = self.next_id()
        if isinstance(lines, str):
            lines = [lines]
        lines = lines or []
        prst = "roundRect" if radius else prst
        fill_xml = "<a:noFill/>" if no_fill else f'<a:solidFill><a:srgbClr val="{color(fill)}"/></a:solidFill>'
        line_xml = (
            f'<a:ln w="{round(stroke_width * 12700)}">'
            f'<a:solidFill><a:srgbClr val="{color(stroke)}"/></a:solidFill>'
            "</a:ln>"
        )
        para_xml = []
        for i, line in enumerate(lines):
            para_xml.append(
                f"""
                <a:p>
                  <a:pPr algn="{align}"/>
                  <a:r>
                    <a:rPr lang="zh-CN" sz="{font_size * 100}" b="{1 if bold else 0}">
                      <a:solidFill><a:srgbClr val="{color(text_color)}"/></a:solidFill>
                      <a:latin typeface="Microsoft YaHei"/>
                      <a:ea typeface="Microsoft YaHei"/>
                    </a:rPr>
                    <a:t>{esc(line)}</a:t>
                  </a:r>
                  <a:endParaRPr lang="zh-CN" sz="{font_size * 100}"/>
                </a:p>
                """
            )
        if not para_xml:
            para_xml.append("<a:p/>")
        body = f"""
        <p:sp>
          <p:nvSpPr>
            <p:cNvPr id="{sid}" name="{esc(name)}"/>
            <p:cNvSpPr/>
            <p:nvPr/>
          </p:nvSpPr>
          <p:spPr>
            <a:xfrm>
              <a:off x="{emu_x(x)}" y="{emu_y(y)}"/>
              <a:ext cx="{emu_w(w)}" cy="{emu_h(h)}"/>
            </a:xfrm>
            <a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>
            {fill_xml}
            {line_xml}
          </p:spPr>
          <p:txBody>
            <a:bodyPr wrap="square" lIns="91440" tIns="54864" rIns="91440" bIns="54864"/>
            <a:lstStyle/>
            {''.join(para_xml)}
          </p:txBody>
        </p:sp>
        """
        self.items.append(body)

    def line(self, x1, y1, x2, y2, stroke="64748B", width=2.0, arrow=True, name="Arrow"):
        sid = self.next_id()
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1) or 1
        h = abs(y2 - y1) or 1
        flip_h = ' flipH="1"' if x2 < x1 else ""
        flip_v = ' flipV="1"' if y2 < y1 else ""
        arrow_xml = '<a:tailEnd type="triangle"/>' if arrow else ""
        body = f"""
        <p:sp>
          <p:nvSpPr>
            <p:cNvPr id="{sid}" name="{esc(name)}"/>
            <p:cNvSpPr/>
            <p:nvPr/>
          </p:nvSpPr>
          <p:spPr>
            <a:xfrm{flip_h}{flip_v}>
              <a:off x="{emu_x(x)}" y="{emu_y(y)}"/>
              <a:ext cx="{emu_w(w)}" cy="{emu_h(h)}"/>
            </a:xfrm>
            <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
            <a:ln w="{round(width * 12700)}">
              <a:solidFill><a:srgbClr val="{color(stroke)}"/></a:solidFill>
              {arrow_xml}
            </a:ln>
          </p:spPr>
          <p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>
        </p:sp>
        """
        self.items.append(body)


def slide_xml() -> str:
    s = SlideBuilder()

    # Title and subtitle.
    s.text_box(72, 42, 900, 46, "新版：漏斗收敛 + 创新点侧注", 24, True, fill="F8FAFC", stroke="F8FAFC")
    s.text_box(74, 86, 1040, 30, "四层漏斗从宽泛输入逐步收敛到精确动作；两侧方框说明对应创新点。", 12, True, fill="F8FAFC", stroke="F8FAFC")

    # Main canvas.
    s.shape(70, 145, 1660, 875, fill="FFFFFF", stroke="CBD5E1", stroke_width=1.3, radius=True, name="主画布")
    s.text_box(650, 178, 500, 32, "由宽到窄的安全约束收敛", 20, True, fill="FFFFFF", stroke="FFFFFF", align="ctr")
    s.text_box(625, 214, 550, 28, "候选输入 → 可行路点 → 可靠触发 → 方向性动作", 12, True, fill="FFFFFF", stroke="FFFFFF", text_color="475569", align="ctr")

    # Editable funnel outline as a trapezoid.
    s.shape(330, 265, 1140, 535, fill="F8FAFC", stroke="CBD5E1", stroke_width=1.5, prst="trapezoid", radius=False, name="漏斗外轮廓")

    levels = [
        (410, 290, 980, 78, "EFF6FF", "2563EB", "1. 候选输入空间", "协同感知对象、V2X-only 目标、上游规划路点"),
        (470, 405, 860, 78, "FFF7ED", "EA580C", "2. 几何可行路点空间", "可见性边界 + 接近速度放大 + 多目标排斥，先把路点推出危险区"),
        (540, 520, 720, 78, "F5F3FF", "7C3AED", "3. 低误报触发空间", "检测-修正解耦；AND 降噪；front_ttc<3s 保留紧急响应"),
        (620, 635, 560, 78, "ECFDF5", "059669", "4. 方向性安全动作空间", "front/rear 风险分解；后车压力触发保持速度 + 横向逃逸"),
    ]
    for x, y, w, h, fill, stroke, title, desc in levels:
        s.shape(x, y, w, h, fill=fill, stroke=stroke, stroke_width=1.7, name=title)
        s.text_box(x + 28, y + 17, w - 60, 24, title, 15, True, fill=fill, stroke=fill, text_color=stroke)
        s.text_box(x + 28, y + 45, w - 60, 22, desc, 10, True, fill=fill, stroke=fill, text_color="334155")

    s.line(900, 368, 900, 405, "94A3B8", 1.8)
    s.line(900, 483, 900, 520, "94A3B8", 1.8)
    s.line(900, 598, 900, 635, "94A3B8", 1.8)

    s.shape(690, 745, 420, 64, fill="FEF2F2", stroke="DC2626", stroke_width=1.7, name="最终安全约束输出")
    s.text_box(710, 763, 380, 26, "最终安全约束输出", 16, True, fill="FEF2F2", stroke="FEF2F2", text_color="991B1B", align="ctr")
    s.text_box(525, 822, 750, 28, "修正后路点 / target_speed_factor / lane_escape / 可解释风险信号", 11, True, fill="FFFFFF", stroke="FFFFFF", text_color="334155", align="ctr")

    callouts = [
        (105, 270, 235, 118, "EFF6FF", "2563EB", "协同输入不只看自车", ["V2X-only 与可见性标记", "进入同一约束计算"], 340, 329, 410, 329),
        (105, 440, 235, 132, "FFF7ED", "EA580C", "创新点一：动态安全区间", ["2.5m / 4.0m 自适应边界", "接近越快，约束越强"], 340, 506, 470, 444),
        (1460, 385, 250, 132, "F5F3FF", "7C3AED", "创新点二：低误报触发", ["几何修正与检测解耦", "AND + front_ttc 覆盖"], 1460, 452, 1330, 444),
        (1460, 560, 250, 145, "ECFDF5", "059669", "创新点三：RearEscape", ["后方压力不等于急刹", "前方非立即碰撞时横向逃逸"], 1460, 632, 1180, 674),
    ]
    for x, y, w, h, fill, stroke, title, lines, ax1, ay1, ax2, ay2 in callouts:
        s.shape(x, y, w, h, fill=fill, stroke=stroke, stroke_width=1.5, name=title)
        s.text_box(x + 26, y + 24, w - 46, 24, title, 12, True, fill=fill, stroke=fill)
        s.text_box(x + 26, y + 57, w - 46, 50, lines, 9, True, fill=fill, stroke=fill, text_color="334155")
        s.line(ax1, ay1, ax2, ay2, stroke, 1.8)

    s.shape(620, 890, 560, 58, fill="FFFFFF", stroke="94A3B8", stroke_width=1.3, name="核心主张")
    s.text_box(640, 908, 520, 25, "核心主张：把协同感知收敛为可执行安全约束，而不是只给风险分数", 11, True, fill="FFFFFF", stroke="FFFFFF", align="ctr")

    s.text_box(74, 1054, 1400, 26, "当前综合最优主线：Hybrid+AND+TTC+RearEscape-thr0.30；图中完整三层风险接口只作为扩展能力提示，不作为最优主路径。", 8, True, fill="F8FAFC", stroke="F8FAFC", text_color="475569")

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      {''.join(s.items)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""


def write_pptx():
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""
    presentation = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""
    pres_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
</Relationships>"""
    slide_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>"""
    master = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""
    master_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""
    layout = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""
    layout_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""
    theme = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Editable Framework Theme">
  <a:themeElements>
    <a:clrScheme name="Office"><a:dk1><a:srgbClr val="000000"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="1F2937"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2><a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="EA580C"/></a:accent2><a:accent3><a:srgbClr val="7C3AED"/></a:accent3><a:accent4><a:srgbClr val="059669"/></a:accent4><a:accent5><a:srgbClr val="DC2626"/></a:accent5><a:accent6><a:srgbClr val="94A3B8"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="Office"><a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:majorFont><a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>"""
    core = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>SafeCoDriver Editable Framework</dc:title>
  <dc:creator>Codex</dc:creator>
</cp:coreProperties>"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft PowerPoint</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>1</Slides>
</Properties>"""

    with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)
        z.writestr("ppt/presentation.xml", presentation)
        z.writestr("ppt/_rels/presentation.xml.rels", pres_rels)
        z.writestr("ppt/slides/slide1.xml", slide_xml())
        z.writestr("ppt/slides/_rels/slide1.xml.rels", slide_rels)
        z.writestr("ppt/slideMasters/slideMaster1.xml", master)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", master_rels)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", layout)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", layout_rels)
        z.writestr("ppt/theme/theme1.xml", theme)
    print(OUT)


if __name__ == "__main__":
    write_pptx()
