from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
W, H = 1800, 1120
FONT = "WenQuanYi Zen Hei"


def esc(s: str) -> str:
    return escape(s, {'"': '&quot;'})


def line_text(x, y, lines, size=24, weight=500, color="#1f2937", gap=None, anchor="start", cls=""):
    if isinstance(lines, str):
        lines = [lines]
    if gap is None:
        gap = int(size * 1.42)
    extra = f' class="{cls}"' if cls else ""
    out = [f'<text x="{x}" y="{y}" font-family=\'{FONT}\' font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{extra}>']
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else gap
        out.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
    out.append("</text>")
    return "\n".join(out)


def rect(x, y, w, h, fill="#fff", stroke="#d1d5db", sw=2, r=26, extra=""):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'


def pill(x, y, w, h, text, fill, color="#fff", size=18):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" fill="{fill}"/>'
        + line_text(x + w / 2, y + h / 2 + size * 0.35, text, size=size, weight=700, color=color, anchor="middle")
    )


def arrow(x1, y1, x2, y2, color="#64748b", sw=3, curve=0):
    if curve:
        mx = (x1 + x2) / 2
        return f'<path d="M{x1} {y1} C {mx} {y1 + curve}, {mx} {y2 - curve}, {x2} {y2}" fill="none" stroke="{color}" stroke-width="{sw}" marker-end="url(#arrow)" stroke-linecap="round"/>'
    return f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="{sw}" marker-end="url(#arrow)" stroke-linecap="round"/>'


def icon_car(x, y, scale=1, color="#0f766e"):
    s = scale
    return f"""
<g transform="translate({x} {y}) scale({s})" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 34 L20 18 Q23 12 31 12 L61 12 Q69 12 72 18 L80 34"/>
  <path d="M8 34 H84 V52 H8 Z" fill="#ffffff"/>
  <circle cx="25" cy="55" r="7" fill="#ffffff"/>
  <circle cx="67" cy="55" r="7" fill="#ffffff"/>
  <path d="M31 18 H61 L67 34 H25 Z"/>
</g>"""


def icon_shield(x, y, scale=1, color="#2563eb"):
    s = scale
    return f"""
<g transform="translate({x} {y}) scale({s})" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round">
  <path d="M42 8 L74 20 V43 C74 65 60 80 42 88 C24 80 10 65 10 43 V20 Z" fill="#ffffff"/>
  <path d="M27 46 L38 57 L59 34"/>
</g>"""


def icon_clock(x, y, scale=1, color="#7c3aed"):
    s = scale
    return f"""
<g transform="translate({x} {y}) scale({s})" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round">
  <circle cx="42" cy="42" r="34" fill="#ffffff"/>
  <path d="M42 22 V43 L58 53"/>
</g>"""


def icon_lane(x, y, scale=1, color="#b45309"):
    s = scale
    return f"""
<g transform="translate({x} {y}) scale({s})" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
  <path d="M18 84 C24 58 28 36 30 10"/>
  <path d="M70 84 C64 58 60 36 58 10"/>
  <path d="M44 80 V54"/>
  <path d="M44 44 V24"/>
  <path d="M32 37 L44 24 L56 37"/>
</g>"""


def icon_signal(x, y, scale=1, color="#059669"):
    s = scale
    return f"""
<g transform="translate({x} {y}) scale({s})" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round">
  <circle cx="42" cy="58" r="7" fill="#ffffff"/>
  <path d="M24 42 Q42 27 60 42"/>
  <path d="M14 29 Q42 5 70 29"/>
  <path d="M32 55 L16 78"/>
  <path d="M52 55 L68 78"/>
</g>"""


def icon_grid(x, y, scale=1, color="#ea580c"):
    s = scale
    return f"""
<g transform="translate({x} {y}) scale({s})" fill="none" stroke="{color}" stroke-width="2.5">
  <rect x="10" y="10" width="72" height="72" rx="12" fill="#ffffff"/>
  <path d="M34 10 V82 M58 10 V82 M10 34 H82 M10 58 H82"/>
  <circle cx="58" cy="34" r="8" fill="#fed7aa"/>
</g>"""


def icon_eye(x, y, scale=1, color="#0e7490"):
    s = scale
    return f"""
<g transform="translate({x} {y}) scale({s})" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
  <path d="M8 46 Q42 12 76 46 Q42 80 8 46 Z" fill="#ffffff"/>
  <circle cx="42" cy="46" r="13" fill="#cffafe"/>
</g>"""


def base_defs():
    return f"""
<defs>
  <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
    <stop offset="0%" stop-color="#f8fafc"/>
    <stop offset="55%" stop-color="#eef6ff"/>
    <stop offset="100%" stop-color="#fff7ed"/>
  </linearGradient>
  <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="#0f172a" flood-opacity="0.12"/>
  </filter>
  <filter id="thinShadow" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#0f172a" flood-opacity="0.10"/>
  </filter>
  <marker id="arrow" markerWidth="12" markerHeight="12" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#64748b"/>
  </marker>
  <style>
    .small {{ font-family: {FONT}; font-size: 18px; fill: #475569; }}
    .tiny {{ font-family: {FONT}; font-size: 15px; fill: #64748b; }}
    .code {{ font-family: "JetBrains Mono", "Consolas", monospace; font-size: 17px; fill: #334155; }}
    .cardTitle {{ font-family: {FONT}; font-size: 26px; font-weight: 800; fill: #0f172a; }}
  </style>
</defs>
"""


def svg_wrap(body, title, subtitle, note=""):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
{base_defs()}
<rect width="{W}" height="{H}" fill="url(#bg)"/>
{line_text(72, 72, title, size=38, weight=800, color="#0f172a")}
{line_text(74, 108, subtitle, size=19, weight=500, color="#475569")}
{body}
{line_text(74, 1072, note or "当前综合最优主线：Hybrid+AND+TTC+RearEscape-thr0.30；图中完整三层风险接口只作为扩展能力提示，不作为最优主路径。", size=15, color="#64748b")}
</svg>
"""


def variant_1():
    body = []
    body.append(rect(70, 145, 1660, 865, fill="#ffffff", stroke="#dbeafe", sw=2, r=34, extra='filter="url(#softShadow)"'))
    body.append(pill(112, 185, 178, 44, "输入层", "#2563eb"))
    body.append(pill(478, 185, 240, 44, "创新点一：路点安全集", "#ea580c"))
    body.append(pill(852, 185, 250, 44, "创新点二：低误报触发", "#7c3aed"))
    body.append(pill(1228, 185, 285, 44, "创新点三：方向性处置", "#0f766e"))
    body.append(pill(1525, 185, 145, 44, "结果", "#dc2626"))

    cards = [
        (110, 270, 270, 535, "#eff6ff", "#2563eb", "协同感知 + 路点", ["PerceptionResult", "自车/目标状态", "可见性与 V2X-only", "10 个未来路点"], icon_signal(188, 535, .95, "#2563eb")),
        (455, 270, 300, 535, "#fff7ed", "#ea580c", "时序几何安全集", ["可见目标：2.5m", "不可见目标：4.0m", "接近速度放大", "多目标排斥合力"], icon_grid(560, 535, 1.05, "#ea580c")),
        (825, 270, 300, 535, "#f5f3ff", "#7c3aed", "检测-修正解耦", ["路点修正始终执行", "V1 只负责风险触发", "AND：p>0.30 且 geom", "front_ttc<3s 覆盖"], icon_clock(932, 535, 1.02, "#7c3aed")),
        (1195, 270, 330, 535, "#ecfdf5", "#0f766e", "方向性风险 + RearEscape", ["front/side TTC", "rear TTC 与 rear gap", "后车近且前方非立即碰撞", "保持速度 + 横向逃逸"], icon_lane(1322, 535, 1.03, "#0f766e")),
        (1570, 270, 120, 535, "#fef2f2", "#dc2626", "效果", ["WPC", "0.29%", "base", "0% Coll.", "stress", "25% Coll."], icon_shield(1585, 610, .78, "#dc2626")),
    ]
    for x, y, w, h, fill, stroke, title, lines, icon in cards:
        body.append(rect(x, y, w, h, fill=fill, stroke=stroke, sw=2.2, r=28, extra='filter="url(#thinShadow)"'))
        body.append(line_text(x + 28, y + 58, title, size=26, weight=800, color="#0f172a"))
        body.append(icon)
        yy = y + 145
        for i, item in enumerate(lines):
            body.append(f'<circle cx="{x+42}" cy="{yy+i*42-7}" r="5.5" fill="{stroke}"/>')
            body.append(line_text(x + 60, yy + i * 42, item, size=20 if w > 130 else 17, weight=600 if item in ["0.29%", "0% Coll.", "25% Coll."] else 500, color="#334155"))
    for x1, x2, c in [(380, 455, "#2563eb"), (755, 825, "#ea580c"), (1125, 1195, "#7c3aed"), (1525, 1570, "#0f766e")]:
        body.append(arrow(x1, 535, x2, 535, c, 4))
    body.append(rect(455, 845, 1070, 95, fill="#f8fafc", stroke="#cbd5e1", sw=1.8, r=24))
    body.append(line_text(490, 883, "论文表达重点", size=24, weight=800, color="#0f172a"))
    body.append(line_text(490, 918, "不是把碰撞概率直接变成刹车，而是先约束未来路点，再用方向性风险决定安全动作。", size=21, weight=500, color="#334155"))
    return svg_wrap("\n".join(body), "方案 A：创新层级式算法框架图", "以三项创新为主轴，适合论文 Method Overview 第一张图。")


def variant_2():
    body = []
    body.append(rect(80, 150, 1640, 855, fill="#ffffff", stroke="#cbd5e1", sw=2, r=34, extra='filter="url(#softShadow)"'))
    body.append(rect(130, 230, 300, 575, fill="#f0f9ff", stroke="#0284c7", sw=2.2, r=30))
    body.append(icon_car(185, 295, 1.15, "#0284c7"))
    body.append(line_text(170, 455, ["统一输入", "协同感知目标", "上游规划路点"], size=28, weight=800, color="#0f172a"))
    body.append(line_text(170, 585, ["状态 / 速度 / 尺寸", "可见性 / V2X-only", "未来 5s 路点序列"], size=20, color="#475569"))

    body.append(rect(545, 210, 485, 275, fill="#fff7ed", stroke="#ea580c", sw=2.4, r=32, extra='filter="url(#thinShadow)"'))
    body.append(icon_grid(585, 250, .85, "#ea580c"))
    body.append(line_text(690, 260, "几何约束分支", size=30, weight=850, color="#9a3412"))
    body.append(line_text(690, 310, ["时序安全区间：每个路点都有动态危险区", "可见性边界 + 接近速度放大 + 多车排斥", "输出：修正后路点，WPC 直接下降"], size=21, color="#334155", gap=34))

    body.append(rect(545, 555, 485, 275, fill="#f5f3ff", stroke="#7c3aed", sw=2.4, r=32, extra='filter="url(#thinShadow)"'))
    body.append(icon_clock(590, 595, .82, "#7c3aed"))
    body.append(line_text(690, 605, "触发判断分支", size=30, weight=850, color="#5b21b6"))
    body.append(line_text(690, 655, ["V1 碰撞概率只用于是否触发控制", "AND 降低误报：p>0.30 且 geom", "前/侧向 TTC<3s 保留紧急响应"], size=21, color="#334155", gap=34))

    body.append(rect(1140, 330, 300, 380, fill="#ecfdf5", stroke="#059669", sw=2.4, r=34, extra='filter="url(#thinShadow)"'))
    body.append(icon_lane(1242, 370, .92, "#059669"))
    body.append(line_text(1185, 520, "风险方向解码", size=30, weight=850, color="#065f46"))
    body.append(line_text(1185, 575, ["front/side TTC", "rear TTC", "rear gap < 18m", "rear_ttc < 2.5s"], size=21, color="#334155", gap=33))

    body.append(rect(1530, 300, 135, 440, fill="#fefce8", stroke="#ca8a04", sw=2.4, r=34, extra='filter="url(#thinShadow)"'))
    body.append(line_text(1598, 365, ["Rear", "Escape"], size=27, weight=900, color="#854d0e", anchor="middle", gap=34))
    body.append(line_text(1598, 465, ["保持", "速度", "+", "横向", "逃逸"], size=25, weight=800, color="#334155", anchor="middle", gap=40))
    body.append(line_text(1598, 690, "lane_escape=1", size=16, weight=700, color="#854d0e", anchor="middle"))

    body.append(arrow(430, 520, 545, 350, "#0284c7", 4, curve=-50))
    body.append(arrow(430, 520, 545, 695, "#0284c7", 4, curve=50))
    body.append(arrow(1030, 348, 1140, 480, "#ea580c", 4, curve=25))
    body.append(arrow(1030, 690, 1140, 595, "#7c3aed", 4, curve=-25))
    body.append(arrow(1440, 520, 1530, 520, "#059669", 4))
    body.append(arrow(1600, 740, 1600, 880, "#ca8a04", 4))

    body.append(rect(1220, 820, 445, 92, fill="#f8fafc", stroke="#94a3b8", sw=1.8, r=24))
    body.append(line_text(1248, 858, "闭环输出：修正路点 + target_speed_factor + lane_escape", size=21, weight=700, color="#0f172a"))
    body.append(line_text(1248, 888, "BS3_blind_stress：collision=0，overlap=0，WPC=0/2200", size=18, color="#475569"))
    return svg_wrap("\n".join(body), "方案 B：双分支解耦框架图", "突出“路点修正”和“风险触发”两条分支解耦，适合说明为什么不是单一阈值策略。")


def variant_3():
    body = []
    body.append(rect(80, 150, 1640, 850, fill="#ffffff", stroke="#bae6fd", sw=2, r=34, extra='filter="url(#softShadow)"'))
    body.append(line_text(900, 210, "从协同感知到安全约束的闭环场景图", size=31, weight=850, color="#0f172a", anchor="middle"))
    body.append(rect(560, 275, 680, 470, fill="#f8fafc", stroke="#cbd5e1", sw=2.2, r=28))
    body.append('<path d="M615 700 C740 540 1040 490 1185 310" fill="none" stroke="#94a3b8" stroke-width="46" stroke-linecap="round"/>')
    body.append('<path d="M615 700 C740 540 1040 490 1185 310" fill="none" stroke="#ffffff" stroke-width="6" stroke-linecap="round" stroke-dasharray="24 22"/>')
    body.append(icon_car(760, 560, .9, "#2563eb"))
    body.append(icon_car(1000, 385, .75, "#ef4444"))
    body.append(icon_car(635, 665, .72, "#b45309"))
    body.append('<path d="M735 595 C815 540 910 492 1045 430" fill="none" stroke="#ea580c" stroke-width="5" stroke-linecap="round" marker-end="url(#arrow)"/>')
    body.append('<path d="M755 640 C850 595 1005 560 1125 520" fill="none" stroke="#0f766e" stroke-width="5" stroke-dasharray="14 10" stroke-linecap="round" marker-end="url(#arrow)"/>')
    body.append(line_text(620, 320, "场景核心", size=25, weight=850, color="#0f172a"))
    body.append(line_text(620, 355, ["前/侧向冲突", "后方追车压力", "遮挡或 V2X-only 不确定性"], size=19, color="#475569", gap=29))

    callouts = [
        (135, 270, 350, 155, "#eff6ff", "#2563eb", "输入不再只是自车视野", ["协同目标 + 可见性标记", "让盲区目标进入约束计算"], icon_eye(165, 312, .65, "#2563eb")),
        (130, 515, 360, 180, "#fff7ed", "#ea580c", "创新点一：动态安全区间", ["2.5m/4.0m 自适应边界", "接近越快，危险区越大", "多车排斥避免次生路点碰撞"], icon_grid(166, 575, .62, "#ea580c")),
        (1315, 270, 355, 160, "#f5f3ff", "#7c3aed", "创新点二：低误报触发", ["修正与检测解耦", "p>0.30 且 geom", "front_ttc<3s 紧急覆盖"], icon_clock(1348, 315, .62, "#7c3aed")),
        (1315, 515, 355, 190, "#ecfdf5", "#059669", "创新点三：RearEscape", ["识别后方压力", "前方非 1s 内立即碰撞时", "保持速度 + 横向逃逸"], icon_lane(1350, 570, .65, "#059669")),
    ]
    for x, y, w, h, fill, stroke, title, lines, ic in callouts:
        body.append(rect(x, y, w, h, fill=fill, stroke=stroke, sw=2, r=24, extra='filter="url(#thinShadow)"'))
        body.append(ic)
        body.append(line_text(x + 105, y + 45, title, size=23, weight=850, color="#0f172a"))
        body.append(line_text(x + 105, y + 82, lines, size=18, color="#334155", gap=27))
    body.append(arrow(485, 360, 610, 420, "#2563eb", 3))
    body.append(arrow(490, 610, 690, 595, "#ea580c", 3))
    body.append(arrow(1315, 360, 1170, 425, "#7c3aed", 3))
    body.append(arrow(1315, 620, 1110, 595, "#059669", 3))

    body.append(rect(560, 800, 680, 120, fill="#fef2f2", stroke="#ef4444", sw=2.2, r=28))
    body.append(line_text(600, 845, "案例证据：BS3_blind_stress", size=26, weight=850, color="#991b1b"))
    body.append(line_text(600, 885, "RearEscape 同时避开前向碰撞和后向追尾：collision=0，overlap=0，WPC=0/2200。", size=21, color="#334155"))
    return svg_wrap("\n".join(body), "方案 C：场景驱动式算法框架图", "用路口压力场景承载算法结构，适合论文中解释为什么需要 RearEscape。")


def variant_4():
    body = []
    body.append(rect(80, 150, 1640, 850, fill="#ffffff", stroke="#d1d5db", sw=2, r=34, extra='filter="url(#softShadow)"'))
    cx, cy = 900, 560
    rings = [
        (360, "#eff6ff", "#2563eb", "外环：协同感知与上游路点"),
        (280, "#fff7ed", "#ea580c", "中环：时序几何安全集"),
        (205, "#f5f3ff", "#7c3aed", "内环：低误报风险触发"),
        (130, "#ecfdf5", "#059669", "核心：方向性处置"),
    ]
    for r, fill, stroke, label in rings:
        body.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="3" opacity="0.92"/>')
        body.append(line_text(cx, cy - r + 42, label, size=22, weight=850, color=stroke, anchor="middle"))
    body.append(rect(cx - 170, cy - 72, 340, 144, fill="#ffffff", stroke="#0f766e", sw=2.8, r=28, extra='filter="url(#thinShadow)"'))
    body.append(line_text(cx, cy - 18, "RearEscape", size=34, weight=900, color="#064e3b", anchor="middle"))
    body.append(line_text(cx, cy + 25, "后车压力 -> 保持速度 + 横向逃逸", size=20, weight=700, color="#334155", anchor="middle"))
    body.append(icon_lane(cx - 40, cy + 50, .55, "#0f766e"))

    nodes = [
        (260, 260, "#2563eb", "输入", ["PerceptionResult", "V2X-only 标记", "未来路点"]),
        (1310, 260, "#ea580c", "几何约束", ["2.5m / 4.0m", "接近速度放大", "多车排斥"]),
        (260, 760, "#7c3aed", "风险触发", ["V1 概率", "geom 威胁数", "front_ttc<3s"]),
        (1310, 760, "#dc2626", "实验结果", ["DeepAccident WPC=0.29%", "SUMO base Coll.=0%", "SUMO stress Coll.=25%"]),
    ]
    for x, y, color, title, lines in nodes:
        body.append(rect(x, y, 320, 165, fill="#ffffff", stroke=color, sw=2.2, r=24, extra='filter="url(#thinShadow)"'))
        body.append(line_text(x + 28, y + 45, title, size=26, weight=850, color=color))
        body.append(line_text(x + 28, y + 82, lines, size=19, color="#334155", gap=28))
    body.append(arrow(580, 330, cx - 280, cy - 180, "#2563eb", 3, curve=-20))
    body.append(arrow(1310, 340, cx + 260, cy - 160, "#ea580c", 3, curve=20))
    body.append(arrow(580, 825, cx - 210, cy + 170, "#7c3aed", 3, curve=20))
    body.append(arrow(cx + 260, cy + 170, 1310, 825, "#dc2626", 3, curve=-20))
    body.append(line_text(900, 960, "图意：越靠近中心，约束越从“感知信息”转化为“可执行安全动作”。", size=22, weight=700, color="#334155", anchor="middle"))
    return svg_wrap("\n".join(body), "方案 D：同心约束层框架图", "用层级包围关系表达方法，不是流程图，适合突出安全约束逐层收紧。")


def variant_5():
    body = []
    body.append(rect(80, 150, 1640, 850, fill="#ffffff", stroke="#cbd5e1", sw=2, r=34, extra='filter="url(#softShadow)"'))
    body.append('<path d="M260 250 H1540 L1320 850 H480 Z" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2.2"/>')
    levels = [
        (310, 295, 1180, 105, "#eff6ff", "#2563eb", "候选输入空间", ["协同感知对象、V2X-only 目标、上游规划路点"]),
        (390, 430, 1020, 105, "#fff7ed", "#ea580c", "几何可行路点空间", ["可见性边界 + 接近速度放大 + 多目标排斥，先把路点推出危险区"]),
        (470, 565, 860, 105, "#f5f3ff", "#7c3aed", "低误报触发空间", ["检测-修正解耦；AND 降噪；front_ttc<3s 保留紧急响应"]),
        (550, 700, 700, 105, "#ecfdf5", "#059669", "方向性安全动作空间", ["后方压力不等于急刹；RearEscape 输出保持速度和横向逃逸"]),
    ]
    for x, y, w, h, fill, stroke, title, lines in levels:
        body.append(rect(x, y, w, h, fill=fill, stroke=stroke, sw=2.2, r=24, extra='filter="url(#thinShadow)"'))
        body.append(line_text(x + 36, y + 42, title, size=27, weight=850, color=stroke))
        body.append(line_text(x + 36, y + 78, lines, size=20, color="#334155"))
    body.append(icon_signal(185, 300, .75, "#2563eb"))
    body.append(icon_grid(230, 435, .75, "#ea580c"))
    body.append(icon_clock(270, 570, .75, "#7c3aed"))
    body.append(icon_lane(310, 705, .75, "#059669"))
    body.append(arrow(900, 400, 900, 430, "#64748b", 3))
    body.append(arrow(900, 535, 900, 565, "#64748b", 3))
    body.append(arrow(900, 670, 900, 700, "#64748b", 3))
    body.append(rect(700, 875, 400, 80, fill="#fef2f2", stroke="#dc2626", sw=2.2, r=26))
    body.append(line_text(900, 925, "最终安全约束输出", size=29, weight=900, color="#991b1b", anchor="middle"))
    body.append(line_text(900, 982, "修正后路点 / 速度因子 / lane_escape / 可解释风险信号", size=22, weight=700, color="#334155", anchor="middle"))
    body.append(line_text(1490, 920, ["核心论文主张", "把协同感知转为", "可执行约束", "而不是只给风险分数"], size=23, weight=800, color="#0f172a", anchor="middle", gap=34))
    return svg_wrap("\n".join(body), "方案 E：约束漏斗式算法框架图", "用漏斗表达“候选空间逐层收紧到安全动作”，适合放在方法章节开头。")


def variant_6():
    body = []
    body.append(rect(70, 145, 1660, 875, fill="#ffffff", stroke="#cbd5e1", sw=2, r=34, extra='filter="url(#softShadow)"'))
    body.append(line_text(900, 205, "由宽到窄的安全约束收敛", size=30, weight=900, color="#0f172a", anchor="middle"))
    body.append(line_text(900, 235, "候选输入 → 可行路点 → 可靠触发 → 方向性动作", size=19, weight=700, color="#475569", anchor="middle"))

    # Funnel body. The outer trapezoid gives the figure a non-flowchart structure.
    body.append('<path d="M330 265 H1470 L1270 800 H530 Z" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2.4"/>')
    levels = [
        (410, 290, 980, 78, "#eff6ff", "#2563eb", "1. 候选输入空间", "协同感知对象、V2X-only 目标、上游规划路点"),
        (470, 405, 860, 78, "#fff7ed", "#ea580c", "2. 几何可行路点空间", "可见性边界 + 接近速度放大 + 多目标排斥，先把路点推出危险区"),
        (540, 520, 720, 78, "#f5f3ff", "#7c3aed", "3. 低误报触发空间", "检测-修正解耦；AND 降噪；front_ttc<3s 保留紧急响应"),
        (620, 635, 560, 78, "#ecfdf5", "#059669", "4. 方向性安全动作空间", "front/rear 风险分解；后车压力触发保持速度 + 横向逃逸"),
    ]
    for x, y, w, h, fill, stroke, title, desc in levels:
        body.append(rect(x, y, w, h, fill=fill, stroke=stroke, sw=2.2, r=22, extra='filter="url(#thinShadow)"'))
        body.append(line_text(x + 28, y + 34, title, size=23, weight=900, color=stroke))
        body.append(line_text(x + 28, y + 61, desc, size=16, weight=700, color="#334155"))
    body.append(arrow(900, 368, 900, 405, "#94a3b8", 3))
    body.append(arrow(900, 483, 900, 520, "#94a3b8", 3))
    body.append(arrow(900, 598, 900, 635, "#94a3b8", 3))
    body.append(rect(690, 745, 420, 64, fill="#fef2f2", stroke="#dc2626", sw=2.2, r=22, extra='filter="url(#thinShadow)"'))
    body.append(line_text(900, 786, "最终安全约束输出", size=25, weight=900, color="#991b1b", anchor="middle"))
    body.append(line_text(900, 836, "修正后路点 / target_speed_factor / lane_escape / 可解释风险信号", size=18, weight=800, color="#334155", anchor="middle"))

    # Innovation callouts distributed on both sides, attached to the relevant funnel layer.
    callouts = [
        (105, 270, 235, 118, "#eff6ff", "#2563eb", "协同输入不只看自车", ["V2X-only 与可见性标记", "进入同一约束计算"], icon_eye(124, 309, .48, "#2563eb"), 340, 329, 410, 329),
        (105, 440, 235, 132, "#fff7ed", "#ea580c", "创新点一：动态安全区间", ["2.5m / 4.0m 自适应边界", "接近越快，约束越强"], icon_grid(124, 485, .48, "#ea580c"), 340, 506, 470, 444),
        (1460, 385, 250, 132, "#f5f3ff", "#7c3aed", "创新点二：低误报触发", ["几何修正与检测解耦", "AND + front_ttc 覆盖"], icon_clock(1478, 430, .48, "#7c3aed"), 1460, 452, 1330, 444),
        (1460, 560, 250, 145, "#ecfdf5", "#059669", "创新点三：RearEscape", ["后方压力不等于急刹", "前方非立即碰撞时横向逃逸"], icon_lane(1478, 606, .48, "#059669"), 1460, 632, 1180, 674),
    ]
    for x, y, w, h, fill, stroke, title, lines, ic, ax1, ay1, ax2, ay2 in callouts:
        body.append(rect(x, y, w, h, fill=fill, stroke=stroke, sw=2, r=20, extra='filter="url(#thinShadow)"'))
        body.append(ic)
        body.append(line_text(x + 78, y + 38, title, size=18, weight=900, color="#0f172a"))
        body.append(line_text(x + 78, y + 67, lines, size=14, weight=700, color="#334155", gap=22))
        body.append(arrow(ax1, ay1, ax2, ay2, stroke, 2.7))

    # Keep only the method claim after removing the scenario and case-effect boxes.
    body.append(rect(620, 890, 560, 58, fill="#ffffff", stroke="#94a3b8", sw=1.8, r=18))
    body.append(line_text(900, 927, "核心主张：把协同感知收敛为可执行安全约束，而不是只给风险分数", size=18, weight=900, color="#0f172a", anchor="middle"))
    return svg_wrap(
        "\n".join(body),
        "新版：漏斗收敛 + 创新点侧注",
        "四层漏斗从宽泛输入逐步收敛到精确动作；两侧方框说明对应创新点。",
    )


def main():
    outputs = {
        "260615最优方案算法框架图中文版.svg": variant_6(),
        "260616最优方法算法框架图_中文.svg": variant_6(),
        "260615最优方案算法框架图中文版_新版_漏斗场景创新点.svg": variant_6(),
        "260615最优方案算法框架图中文版_方案A_创新层级.svg": variant_1(),
        "260615最优方案算法框架图中文版_方案B_双分支解耦.svg": variant_2(),
        "260615最优方案算法框架图中文版_方案C_场景驱动.svg": variant_3(),
        "260615最优方案算法框架图中文版_方案D_同心约束层.svg": variant_4(),
        "260615最优方案算法框架图中文版_方案E_约束漏斗.svg": variant_5(),
    }
    for name, content in outputs.items():
        (ROOT / name).write_text(content, encoding="utf-8")
        print(name)


if __name__ == "__main__":
    main()
