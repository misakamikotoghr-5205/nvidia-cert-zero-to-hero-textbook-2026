#!/usr/bin/env python3
r"""
《英伟达认证工程师从零精通标准课本（2026 全栈详尽版）》排版与 PDF 编译引擎
Build & Print Engine: Markdown + LaTeX (KaTeX) + Pygments + SVG -> Print-ready A4 PDF via Headless Chrome.

关键防翻车设计：
1. 严格保护管线：Fenced Code -> Inline Code -> Raw SVG -> Display Math ($$ / \[) -> Inline Math ($ / \() -> markdown-it-py -> 还原。
   彻底杜绝 `$d_k$` 中的下划线被 markdown-it-py 误解析为 `<em>k</em>`！
2. 零网络依赖离线 KaTeX 渲染：优先通过本地 Node.js + KaTeX 批量预渲染为静态 HTML+CSS，并内嵌纯 Python LaTeX 兜底渲染器。
3. 零 CSS 渐变：严格使用纯色块与实线边框，彻底杜绝 Skia/PDF 生成海量 shading 对象导致体积暴涨（确保 PDF < 24 MB）。
4. 完整印刷级样式：A4 版心、运行页眉页脚、自动多级目录、长代码块自动换行不裁切、短表格/标题防孤立分页、Ghostscript 自动压缩保底。
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.util import ClassNotFound

# ---------------------------------------------------------------------------
# 路径常量定义
# ---------------------------------------------------------------------------
PROJECT_DIR = Path("/home/magnus51235608/GoogleDrive/05_制作/NVIDIA_ZeroToHero_Textbook_2026")
CHAPTERS_DIR = PROJECT_DIR / "chapters"
OUTPUT_DIR_DRAFT = Path("/home/magnus51235608/GoogleDrive/05_制作")
OUTPUT_DIR_REVIEW = Path("/home/magnus51235608/GoogleDrive/03_待审")

BOOK_FILENAME_BASE = "英伟达认证工程师从零精通标准课本_2026全栈详尽版"
BOOK_TITLE = "英伟达认证工程师从零精通标准课本"
BOOK_SUBTITLE = "2026 全栈详尽版 · 从零基础 AI 数学与 CUDA 底层到万卡集群、Agentic AI 与高并发生产推理"
BOOK_EDITION = "2026 全栈官方认证考纲对齐版 · Associate (NCA) & Professional (NCP) 双轨实战教材"

VOLUME_FILES = [
    "vol1_foundations.md",
    "vol2_llm_and_training.md",
    "vol3_rag_and_agents.md",
    "vol4_inference_and_serving.md",
    "vol5_infra_and_exams.md",
]

MAX_PDF_BYTES = 24_000_000  # 严格低于 25 MB (25,165,824 B) 限制，留足安全余量
CHROME_BIN = "/usr/local/bin/google-chrome"
KATEX_BUNDLE_DIR = Path("/tmp/katex_bundle/node_modules/katex")
NODE_FALLBACK_BIN = "/home/magnus51235608/.hermes/tools/node-26.7.0-linux-x64/bin/node"
NPM_FALLBACK_BIN = "/home/magnus51235608/.hermes/tools/node-26.7.0-linux-x64/bin/npm"

# ---------------------------------------------------------------------------
# 安全的 IPv4 校验白名单（防止真实服务器公网/内网 IP 泄漏）
# ---------------------------------------------------------------------------
ALLOWED_EXAMPLE_IPS = {
    "127.0.0.1",
    "0.0.0.0",
    "255.255.255.0",
    "255.255.255.255",
    "255.255.0.0",
    "255.0.0.0",
    "8.8.8.8",
    "1.1.1.1",
}
ALLOWED_IP_PREFIXES = ("192.0.2.", "198.51.100.", "203.0.113.")

# 合法 HTML 标签白名单，防止文档中类似 `<UNK>` / `<BOS>` / `<d>` 被当成 HTML 标签吞掉
SAFE_HTML_TAGS = {
    "a", "abbr", "b", "blockquote", "br", "caption", "circle", "code", "col",
    "colgroup", "dd", "defs", "del", "details", "div", "dl", "dt", "em",
    "figcaption", "figure", "g", "h1", "h2", "h3", "h4", "h5", "h6", "hr",
    "i", "img", "ins", "kbd", "li", "line", "mark", "marker", "ol", "p",
    "path", "polygon", "polyline", "pre", "rect", "s", "samp", "section",
    "small", "span", "strong", "sub", "summary", "sup", "svg", "table",
    "tbody", "td", "text", "tfoot", "th", "thead", "tr", "tspan", "u", "ul",
    "var",
}


def _resolve_tool_bin(name: str, fallback_path: str) -> Optional[str]:
    """定位外部可执行文件；若 PATH 中缺失则显式回退至兜底路径。"""
    found = shutil.which(name)
    if found:
        return found
    fb = Path(fallback_path)
    if fb.is_file() and os.access(fb, os.X_OK):
        return str(fb)
    return None


def find_katex_dir() -> Optional[Path]:
    """定位本地 KaTeX 安装目录；若缺失且有 npm 则自动安装至 /tmp/katex_bundle。"""
    candidates = [
        KATEX_BUNDLE_DIR,
        PROJECT_DIR / "node_modules" / "katex",
        Path("/usr/local/lib/node_modules/katex"),
    ]
    for cand in candidates:
        if (cand / "dist" / "katex.min.css").is_file() and (cand / "dist" / "katex.js").is_file():
            return cand

    npm_bin = _resolve_tool_bin("npm", NPM_FALLBACK_BIN)
    if npm_bin:
        bundle_root = Path("/tmp/katex_bundle")
        try:
            bundle_root.mkdir(parents=True, exist_ok=True)
            npm_env = {**os.environ, "PATH": f"{Path(npm_bin).parent}:{os.environ.get('PATH', '')}"}
            if not (bundle_root / "package.json").exists():
                subprocess.run([npm_bin, "init", "-y"], cwd=str(bundle_root), check=True,
                               env=npm_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(
                [npm_bin, "install", "katex@0.16.11", "--no-audit", "--no-fund"],
                cwd=str(bundle_root), check=True, timeout=60,
                env=npm_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if (KATEX_BUNDLE_DIR / "dist" / "katex.min.css").is_file():
                return KATEX_BUNDLE_DIR
        except Exception as exc:
            print(f"[WARN] 自动安装本地 KaTeX 失败，将启用 Python 兜底渲染器: {exc}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# 纯 Python LaTeX 兜底渲染器（当 Node/KaTeX 不可用或个别语法解析失败时使用）
# ---------------------------------------------------------------------------
LATEX_SYMBOLS = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\vartheta": "ϑ", r"\iota": "ι", r"\kappa": "κ",
    r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ",
    r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ",
    r"\upsilon": "υ", r"\phi": "φ", r"\varphi": "φ", r"\chi": "χ",
    r"\psi": "ψ", r"\omega": "ω",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Xi": "Ξ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Upsilon": "Υ",
    r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
    r"\times": "×", r"\cdot": "·", r"\approx": "≈", r"\le": "≤", r"\leq": "≤",
    r"\ge": "≥", r"\geq": "≥", r"\neq": "≠", r"\ne": "≠", r"\equiv": "≡",
    r"\to": "→", r"\rightarrow": "→", r"\leftarrow": "←", r"\Rightarrow": "⇒",
    r"\Leftarrow": "⇐", r"\leftrightarrow": "↔", r"\infty": "∞",
    r"\sum": "∑", r"\prod": "∏", r"\int": "∫", r"\partial": "∂",
    r"\nabla": "∇", r"\pm": "±", r"\mp": "∓", r"\in": "∈", r"\notin": "∉",
    r"\subset": "⊂", r"\subseteq": "⊆", r"\cup": "∪", r"\cap": "∩",
    r"\forall": "∀", r"\exists": "∃", r"\propto": "∝", r"\sim": "∼",
    r"\simeq": "≃", r"\cong": "≅", r"\ll": "≪", r"\gg": "≫",
    r"\ell": "ℓ", r"\cdots": "⋯", r"\dots": "…", r"\ldots": "…",
    r"\quad": " ", r"\qquad": "  ", r"\,": " ", r"\;": " ", r"\!": "",
    r"\{": "{", r"\}": "}", r"\%": "%", r"\_": "_", r"\$": "$",
    r"\left": "", r"\right": "",
}


def _extract_braced(s: str, pos: int) -> Tuple[str, int]:
    """提取从 pos 处 `{...}` 内的完整内容并返回 (内容, 结束后下标)。"""
    if pos >= len(s) or s[pos] != "{":
        if pos < len(s):
            return s[pos], pos + 1
        return "", pos
    depth = 0
    start = pos + 1
    i = pos
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            i += 2
            continue
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i], i + 1
        i += 1
    return s[start:], len(s)


def fallback_latex_to_html(latex: str, display: bool = False) -> str:
    """将常见 LaTeX 公式转为清晰可排版的 styled HTML（兜底保护）。"""
    s = latex.strip()
    matrices: List[str] = []

    # 处理矩阵环境 \begin{pmatrix}...\end{pmatrix} / bmatrix 并用占位符保护其 HTML
    def _matrix_repl(m: re.Match) -> str:
        env = m.group(1)
        body = m.group(2).strip()
        rows = [r.strip() for r in re.split(r"\\\\", body) if r.strip()]
        tr_html = "".join(
            "<tr>" + "".join(
                f"<td>{fallback_latex_to_html(cell.strip(), False)}</td>"
                for cell in row.split("&")
            ) + "</tr>"
            for row in rows
        )
        cls = "pymath-matrix pymath-bmatrix" if "b" in env else "pymath-matrix"
        idx = len(matrices)
        matrices.append(f'<span class="{cls}"><table>{tr_html}</table></span>')
        return f"@@PYMATRIX_{idx}@@"

    s = re.sub(r"\\begin\{([pb]?matrix)\}(.*?)\\end\{\1\}", _matrix_repl, s, flags=re.S)

    # 递归替换 \frac{A}{B} 与 \sqrt{A}
    def _replace_cmd(text: str) -> str:
        out: List[str] = []
        i = 0
        while i < len(text):
            if text.startswith(r"\frac", i) and (i + 5 == len(text) or not text[i + 5].isalpha()):
                num, j = _extract_braced(text, i + 5)
                den, k = _extract_braced(text, j)
                out.append(
                    f'<span class="pymath-frac"><span class="pymath-num">{_replace_cmd(num)}</span>'
                    f'<span class="pymath-den">{_replace_cmd(den)}</span></span>'
                )
                i = k
                continue
            if text.startswith(r"\sqrt", i) and (i + 5 == len(text) or not text[i + 5].isalpha()):
                idx = i + 5
                if idx < len(text) and text[idx] == "[":
                    end_b = text.find("]", idx)
                    if end_b != -1:
                        idx = end_b + 1
                arg, k = _extract_braced(text, idx)
                out.append(f'<span class="pymath-sqrt">√<span class="pymath-sqrt-body">{_replace_cmd(arg)}</span></span>')
                i = k
                continue
            if any(text.startswith(cmd, i) for cmd in (r"\text", r"\mathrm", r"\mathbf", r"\operatorname", r"\mathbb")):
                m_cmd = re.match(r"\\(text|mathrm|mathbf|operatorname|mathbb)\b", text[i:])
                if m_cmd:
                    cmd_name = m_cmd.group(1)
                    arg, k = _extract_braced(text, i + len(m_cmd.group(0)))
                    style = ' style="font-weight:700;"' if cmd_name == "mathbf" else ' style="font-style:normal;"'
                    out.append(f"<span{style}>{html.escape(arg)}</span>")
                    i = k
                    continue
            if text[i] in ("^", "_") and i + 1 < len(text):
                tag = "sup" if text[i] == "^" else "sub"
                if text[i + 1] == "{":
                    arg, k = _extract_braced(text, i + 1)
                    out.append(f"<{tag}>{_replace_cmd(arg)}</{tag}>")
                    i = k
                else:
                    out.append(f"<{tag}>{html.escape(text[i + 1])}</{tag}>")
                    i += 2
                continue
            # 符号替换
            matched_sym = False
            for sym, uni in sorted(LATEX_SYMBOLS.items(), key=lambda x: -len(x[0])):
                if text.startswith(sym, i):
                    after = i + len(sym)
                    if sym[-1].isalpha() and after < len(text) and text[after].isalpha():
                        continue
                    out.append(html.escape(uni))
                    i = after
                    matched_sym = True
                    break
            if matched_sym:
                continue
            if text[i] in ("{", "}"):
                i += 1
                continue
            out.append(html.escape(text[i]))
            i += 1
        return "".join(out)

    rendered = _replace_cmd(s)
    for idx, m_html in enumerate(matrices):
        rendered = rendered.replace(f"@@PYMATRIX_{idx}@@", m_html)
    if display:
        return f'<div class="math-block pymath-display">{rendered}</div>'
    return f'<span class="pymath-inline">{rendered}</span>'


def batch_render_math(
    math_items: List[Tuple[str, bool]],
    katex_dir: Optional[Path],
) -> List[str]:
    """批量将 [(latex, is_display), ...] 渲染为 HTML。优先离线 KaTeX，单条失败自动降级兜底。"""
    if not math_items:
        return []
    if katex_dir is None:
        return [fallback_latex_to_html(tex, disp) for tex, disp in math_items]

    node_bin = _resolve_tool_bin("node", NODE_FALLBACK_BIN)
    if not node_bin:
        return [fallback_latex_to_html(tex, disp) for tex, disp in math_items]

    js_script = r"""
const fs = require('fs');
const katex = require(process.argv[1]);
const items = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const macros = {
  "\\R": "\\mathbb{R}",
  "\\N": "\\mathbb{N}",
  "\\E": "\\mathbb{E}",
  "\\argmax": "\\operatorname*{arg\\,max}",
  "\\argmin": "\\operatorname*{arg\\,min}",
  "\\softmax": "\\operatorname{softmax}",
  "\\diag": "\\operatorname{diag}",
  "\\LayerNorm": "\\operatorname{LayerNorm}",
  "\\RMSNorm": "\\operatorname{RMSNorm}"
};
const out = items.map(([tex, disp]) => {
  try {
    const html = katex.renderToString(tex, {
      displayMode: !!disp,
      throwOnError: true,
      strict: false,
      output: 'html',
      macros
    });
    return { ok: true, html };
  } catch (e) {
    try {
      const html = katex.renderToString(tex, {
        displayMode: !!disp,
        throwOnError: false,
        strict: false,
        output: 'html',
        macros
      });
      return { ok: false, err: String(e.message || e), html };
    } catch (e2) {
      return { ok: false, fatal: true, err: String(e2.message || e2), html: '' };
    }
  }
});
process.stdout.write(JSON.stringify(out));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as tf:
        json.dump(math_items, tf, ensure_ascii=False)
        json_path = tf.name

    try:
        res = subprocess.run(
            [node_bin, "-e", js_script, str(katex_dir), json_path],
            capture_output=True, text=True, check=True, timeout=60,
        )
        parsed = json.loads(res.stdout)
        results: List[str] = []
        warn_count = 0
        for idx, entry in enumerate(parsed):
            tex, disp = math_items[idx]
            if entry.get("ok"):
                results.append(entry["html"])
            elif not entry.get("fatal") and entry.get("html"):
                warn_count += 1
                if warn_count <= 10:
                    print(f"[WARN] KaTeX 容错渲染 #{idx}: {entry.get('err')} | tex={tex[:80]}", file=sys.stderr)
                results.append(entry["html"])
            else:
                warn_count += 1
                print(f"[WARN] KaTeX 回退 Python 渲染 #{idx}: {entry.get('err')} | tex={tex[:80]}", file=sys.stderr)
                results.append(fallback_latex_to_html(tex, disp))
        if warn_count > 10:
            print(f"[WARN] 共有 {warn_count} 条公式触发 KaTeX 容错模式。", file=sys.stderr)
        return results
    except Exception as exc:
        print(f"[WARN] Node.js KaTeX 批量执行异常，降级为 Python 渲染器: {exc}", file=sys.stderr)
        return [fallback_latex_to_html(tex, disp) for tex, disp in math_items]
    finally:
        try:
            os.unlink(json_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Markdown 预处理（严格防破坏公式/代码块）与后处理
# ---------------------------------------------------------------------------
def check_plaintext_ips(text: str, source_name: str = "input") -> List[str]:
    """扫描文本中的非示例 IPv4 地址并返回告警列表。"""
    pattern = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")
    flagged: List[str] = []
    for m in pattern.finditer(text):
        ip = m.group(0)
        if ip in ALLOWED_EXAMPLE_IPS or ip.startswith(ALLOWED_IP_PREFIXES):
            continue
        flagged.append(ip)
    if flagged:
        unique_ips = sorted(set(flagged))
        print(f"[WARN] 在 {source_name} 检测到疑似非示例 IPv4: {unique_ips}", file=sys.stderr)
    return flagged


def _extract_fenced_code_blocks(md_text: str) -> Tuple[str, List[Tuple[str, str, str]]]:
    """
    使用 CommonMark 围栏状态机提取所有代码块。
    返回 (替换占位符后的 Markdown, [(lang, code, indent), ...])。
    """
    lines = md_text.splitlines(keepends=True)
    out_lines: List[str] = []
    blocks: List[Tuple[str, str, str]] = []

    in_fence = False
    fence_char = ""
    fence_len = 0
    fence_indent = ""
    fence_lang = ""
    buf: List[str] = []

    open_re = re.compile(r"^([ \t]*)(`{3,}|~{3,})([^\n]*)\r?\n?$")
    for line in lines:
        if not in_fence:
            m = open_re.match(line)
            if m:
                indent, fence, info = m.groups()
                # 反引号围栏的 info string 不能包含反引号
                if fence[0] == "`" and "`" in info:
                    out_lines.append(line)
                    continue
                in_fence = True
                fence_char = fence[0]
                fence_len = len(fence)
                fence_indent = indent
                fence_lang = info.strip().split()[0] if info.strip() else ""
                buf = []
                continue
            out_lines.append(line)
        else:
            stripped = line.lstrip(" \t")
            end_re = re.compile(rf"^{re.escape(fence_char)}{{{fence_len},}}[ \t]*\r?\n?$")
            if end_re.match(stripped):
                idx = len(blocks)
                blocks.append((fence_lang, "".join(buf), fence_indent))
                # 若在缩进列表内，保持缩进占位符，否则作为独立块占位符
                out_lines.append(f"{fence_indent}@@CODE_BLOCK_{idx}@@\n")
                in_fence = False
            else:
                buf.append(line)

    if in_fence:
        idx = len(blocks)
        blocks.append((fence_lang, "".join(buf), fence_indent))
        out_lines.append(f"{fence_indent}@@CODE_BLOCK_{idx}@@\n")

    return "".join(out_lines), blocks


def _extract_inline_codes(text: str) -> Tuple[str, List[str]]:
    """按 CommonMark 规则提取行内代码 `...`（支持多反引号定界符）。"""
    codes: List[str] = []
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "`":
            j = i
            while j < n and text[j] == "`":
                j += 1
            tick_len = j - i
            # 向后查找完全相同长度的反引号闭合，不跨空行
            k = j
            matched_end = -1
            while k < n:
                if text[k] == "\n" and k + 1 < n and text[k + 1] == "\n":
                    break
                if text[k] == "`":
                    m = k
                    while m < n and text[m] == "`":
                        m += 1
                    if m - k == tick_len:
                        matched_end = m
                        inner = text[j:k]
                        break
                    k = m
                else:
                    k += 1
            if matched_end != -1:
                if len(inner) >= 2 and inner.startswith(" ") and inner.endswith(" ") and inner.strip():
                    inner = inner[1:-1]
                idx = len(codes)
                codes.append(inner)
                out.append(f"@@INLINE_CODE_{idx}@@")
                i = matched_end
                continue
            out.append(text[i:j])
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out), codes


def _extract_raw_svgs(text: str) -> Tuple[str, List[str]]:
    """保护内联 <svg>...</svg> 图解不被 markdown-it-py 或数学正则破坏。"""
    svgs: List[str] = []

    def _repl(m: re.Match) -> str:
        idx = len(svgs)
        svgs.append(m.group(0))
        return f"\n\n@@RAW_SVG_{idx}@@\n\n"

    new_text = re.sub(r"<svg\b[^>]*>.*?</svg>", _repl, text, flags=re.S | re.I)
    return new_text, svgs


def _is_valid_inline_math(content: str, char_after: str) -> bool:
    r"""
    精准区分 `$x_1$` / `$ 16\Psi $` 与货币金额 `$2.5 ... $10`：
    - 闭定界符紧接数字 -> 拒判（货币成对出现）
    - 剥离 \text{} 等文字容器后若仍含中文且缺少 LaTeX 数学结构 -> 拒判
    """
    if not content or not content.strip():
        return False
    if "\n\n" in content:
        return False
    if char_after and char_after.isdigit():
        return False

    stripped = content.strip()
    # 纯数字（如 $100$ 几乎不会是公式，除非前后无货币特征；其实 $100 单独一个不会匹配）
    if re.fullmatch(r"\d+(?:[.,]\d+)?", stripped):
        return False

    # 剥离 \text{...}, \mathrm{...}, \mathbf{...}, \operatorname{...} 检查裸文本
    core = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\{[^{}]*\}", "", stripped)
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", core))
    has_latex_ops = bool(re.search(r"(\\[a-zA-Z]+|[_^{}=<>+\-*/×·])", stripped))

    if has_cjk:
        # 若内部含裸中文，且像货币描述（如 "2.5/小时，相比 A100 的 "），拒判
        if re.match(r"^\s*\d+(?:\.\d+)?\s*(?:/|美元|元|万|亿|块|美金|小时|月|年)", stripped):
            return False
        if not re.search(r"(\\[a-zA-Z]+|[_^=])", core):
            return False

    # 若以数字开头且含空格中文/货币词且无 LaTeX 语法，拒判
    if re.match(r"^\d+(?:\.\d+)?\s+", stripped) and not has_latex_ops:
        return False

    return True


def _extract_math_expressions(text: str) -> Tuple[str, List[Tuple[str, bool]]]:
    r"""
    依次提取块级公式 ($$...$$, \[...\]) 与行内公式 ($...$, \(...\))。
    提前将转义 \$ 保护为 @@ESCAPED_DOLLAR@@，确保绝对安全。
    """
    math_items: List[Tuple[str, bool]] = []
    # 保护 \$ 转义
    protected = text.replace(r"\$", "@@ESCAPED_DOLLAR@@")

    # 1. 提取 $$...$$ 块级公式（兼容引用块内每行前置 `> `）
    def _block_repl(m: re.Match) -> str:
        raw = m.group(1)
        # 若处于 blockquote 内，去掉每行开头的 "> "
        cleaned_lines = [re.sub(r"^[ \t]*>[ \t]?", "", ln) for ln in raw.splitlines()]
        tex = "\n".join(cleaned_lines).strip()
        if not tex:
            return m.group(0)
        idx = len(math_items)
        math_items.append((tex, True))
        return f"@@MATH_BLOCK_{idx}@@"

    protected = re.sub(r"\$\$(.*?)\$\$", _block_repl, protected, flags=re.S)

    # 2. 提取 \[...\] 块级公式
    def _bracket_block_repl(m: re.Match) -> str:
        tex = m.group(1).strip()
        if not tex:
            return m.group(0)
        idx = len(math_items)
        math_items.append((tex, True))
        return f"@@MATH_BLOCK_{idx}@@"

    protected = re.sub(r"\\\[(.*?)\\\]", _bracket_block_repl, protected, flags=re.S)

    # 3. 提取 \(...\) 行内公式
    def _paren_inline_repl(m: re.Match) -> str:
        tex = m.group(1).strip()
        if not tex:
            return m.group(0)
        idx = len(math_items)
        math_items.append((tex, False))
        return f"@@MATH_INLINE_{idx}@@"

    protected = re.sub(r"\\\((.*?)\\\)", _paren_inline_repl, protected, flags=re.S)

    # 4. 逐字符扫描提取 $...$ 行内公式（不跨换行或仅限同一段落，稳妥起见允许单行及同段落）
    out: List[str] = []
    i = 0
    n = len(protected)
    while i < n:
        if protected[i] == "$":
            # 向后在同一段落内寻找闭合 $
            j = i + 1
            found = -1
            while j < n:
                if protected[j] == "\n":
                    # 行内公式一般不跨行；允许单个软换行但绝不跨空行
                    if j + 1 < n and protected[j + 1] == "\n":
                        break
                if protected[j] == "$":
                    cand = protected[i + 1:j]
                    char_after = protected[j + 1] if j + 1 < n else ""
                    if _is_valid_inline_math(cand, char_after):
                        found = j
                        break
                    else:
                        # 这个 $ 不是合法数学闭合，放弃本次开头 $
                        break
                j += 1
            if found != -1:
                tex = protected[i + 1:found].strip()
                idx = len(math_items)
                math_items.append((tex, False))
                out.append(f"@@MATH_INLINE_{idx}@@")
                i = found + 1
                continue
        out.append(protected[i])
        i += 1

    return "".join(out), math_items


def _sanitize_unknown_angle_tags(text: str) -> str:
    """
    在数学和代码块已提取保护的前提下，把像 `<UNK>`、`<BOS>`、`<EOS>`、`<d>` 等
    非标准 HTML 标签转为转义字符 `&lt;UNK&gt;`，避免被 Chrome 当成未知 HTML 标签隐藏！
    """
    tag_re = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9_-]*)(\s[^<>]*?)?(/?)>")

    def _repl(m: re.Match) -> str:
        tag_name = m.group(2).lower()
        if tag_name in SAFE_HTML_TAGS:
            return m.group(0)
        return html.escape(m.group(0))

    return tag_re.sub(_repl, text)


def _render_code_block_html(lang: str, code: str) -> str:
    """使用 Pygments 为围栏代码块生成语法高亮 HTML；ASCII 框图使用等宽架构框样式。"""
    stripped = code.rstrip("\n")
    line_count = stripped.count("\n") + 1
    lang_norm = lang.lower().strip()

    # 检查是否为 ASCII 架构/框图（无语言标注或 text/ascii 且含框线字符）
    box_chars = set("┌┐└┘├┤┬┴┼─│═║╔╗╚╝╠╣╦╩╬▲▼◀▶")
    is_ascii_art = (lang_norm in ("", "text", "ascii", "diagram", "plaintext")) and any(c in box_chars for c in stripped)

    if is_ascii_art:
        escaped = html.escape(stripped)
        short_cls = " short-block" if line_count <= 20 else ""
        return (
            f'<div class="diagram-box{short_cls}">'
            f'<div class="diagram-header">ARCHITECTURE / FLOW DIAGRAM</div>'
            f'<pre class="ascii-diagram"><code>{escaped}</code></pre>'
            f'</div>'
        )

    alias_map = {
        "py": "python", "pbtxt": "protobuf", "proto": "protobuf",
        "sh": "bash", "shell": "bash", "zsh": "bash",
        "yml": "yaml", "c++": "cpp", "cuda": "cuda", "cu": "cuda",
        "colang": "yaml", "jsonl": "json", "dockerfile": "docker",
    }
    lexer_name = alias_map.get(lang_norm, lang_norm)
    lexer = None
    if lexer_name and lexer_name not in ("text", "plaintext", "txt", "output", "log"):
        try:
            lexer = get_lexer_by_name(lexer_name, stripall=False)
        except ClassNotFound:
            lexer = None

    if lexer is not None:
        formatter = HtmlFormatter(noclasses=False, cssclass="highlight", nowrap=False)
        highlighted = highlight(stripped, lexer, formatter)
    else:
        highlighted = f'<div class="highlight"><pre><code>{html.escape(stripped)}</code></pre></div>'

    label = html.escape(lang.upper()) if lang else "CODE / TERMINAL"
    short_cls = " short-block" if line_count <= 16 else ""
    return (
        f'<div class="code-card{short_cls}">'
        f'<div class="code-header"><span>{label}</span><span>{line_count} lines</span></div>'
        f'{highlighted}'
        f'</div>'
    )


def _add_heading_ids_and_extract_toc(body_html: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    为正文中的所有 <h1>/<h2>/<h3> 注入唯一 id、分类 class（区分卷首/章首/小节），
    并提取生成自动多级目录列表。
    """
    toc: List[Dict[str, str]] = []
    heading_re = re.compile(r"<(h[1-3])([^>]*)>(.*?)</\1>", re.S)
    counter = 0

    def _repl(m: re.Match) -> str:
        nonlocal counter
        tag, attrs, inner = m.groups()
        # 剥离内联 HTML 标签以获取纯文本标题
        plain = re.sub(r"<[^>]+>", "", inner).strip()
        plain = html.unescape(plain)
        counter += 1
        sec_id = f"sec-{counter:04d}"

        extra_cls = ""
        level = int(tag[1])
        if level == 1 and re.match(r"^第[一二三四五六七八九十\d]+卷", plain):
            extra_cls = " volume-title"
        elif level == 2 and re.match(r"^第\s*\d+\s*章", plain):
            extra_cls = " chapter-title"

        toc.append({"id": sec_id, "level": str(level), "title": plain, "inner_html": inner, "kind": extra_cls.strip()})
        if "class=" in attrs:
            attrs = re.sub(r'class="([^"]*)"', rf'class="\1{extra_cls}"', attrs)
        else:
            attrs = f'{attrs} class="{extra_cls.strip()}"' if extra_cls else attrs
        return f'<{tag} id="{sec_id}"{attrs}>{inner}</{tag}>'

    new_html = heading_re.sub(_repl, body_html)

    # 若 <h1 class="volume-title"> 与其后的 <h2 class="chapter-title"> 之间没有任何正文，
    # 给紧随其后的首个 chapter-title 加上 `no-page-break` 类，防止卷标题孤零零独占一页空白页！
    vol_h1_re = re.compile(r'<h1\b[^>]*class="[^"]*volume-title[^"]*"[^>]*>.*?</h1>', flags=re.S)
    next_ch_re = re.compile(r'\s*<h2\b([^>]*)class="([^"]*chapter-title[^"]*)"')
    chunks: List[str] = []
    last_pos = 0
    for m in vol_h1_re.finditer(new_html):
        ch_m = next_ch_re.match(new_html, m.end())
        if ch_m:
            cls_val = ch_m.group(2)
            if "no-page-break" not in cls_val.split():
                chunks.append(new_html[last_pos:ch_m.start(2)])
                chunks.append(f"{cls_val} no-page-break")
                last_pos = ch_m.end(2)
    if chunks:
        chunks.append(new_html[last_pos:])
        new_html = "".join(chunks)
    return new_html, toc


def build_toc_html(toc_items: List[Dict[str, str]]) -> str:
    """构建印刷级多级目录页 HTML。"""
    rows: List[str] = []
    for item in toc_items:
        lvl = item["level"]
        cls = f"toc-item toc-l{lvl}"
        if item.get("kind") == "volume-title":
            cls += " toc-volume"
        elif item.get("kind") == "chapter-title":
            cls += " toc-chapter"
        label_html = re.sub(r"</?a\b[^>]*>", "", item.get("inner_html") or html.escape(item["title"]))
        rows.append(
            f'<li class="{cls}"><a href="#{item["id"]}">{label_html}</a></li>'
        )
    return (
        '<section class="toc-section">\n'
        '<div class="toc-banner">CONTENTS · 全书目录</div>\n'
        '<h1 class="toc-heading">目录（五卷十八章全栈学习导航）</h1>\n'
        f'<ul class="toc-list">{"".join(rows)}</ul>\n'
        '</section>\n'
    )


def build_cover_and_guide_html(is_sample: bool = False) -> str:
    """生成高颜值、零 CSS 渐变的封面页与《本书使用指南（零基础学习路径）》。"""
    sample_badge = '<span class="badge-warn">SAMPLE PREVIEW · 排版与公式单元测试样张</span>' if is_sample else ""
    cover_html = f"""
<section class="cover-page">
  <div class="cover-top-bar"></div>
  <div class="cover-meta">
    <span class="badge-nv">NVIDIA CERTIFICATION STANDARD TEXTBOOK · 2026 EDITION</span>
    {sample_badge}
  </div>
  <div class="cover-main">
    <div class="cover-kicker">FROM ZERO TO HERO · 零基础直通高阶工程架构师</div>
    <h1 class="cover-title">{BOOK_TITLE}</h1>
    <div class="cover-subtitle">{BOOK_SUBTITLE}</div>
    <div class="cover-divider"></div>
    <p class="cover-lead">
      严格对标 2026 年英伟达官方认证体系（Associate 入门双轨：<strong>NCA-GENL / NCA-AIIO</strong>；
      Professional 专业级核心：<strong>NCP-GENL / NCP-AAI / NCP-AII</strong>）。坚持“生活化直觉引入 → 极简具体数字手推 →
      结构图解 → 完整可运行代码实验 → 高频误区诊所 → 双轨考点映射”教学闭环，绝不跳步。
    </p>
  </div>
  <div class="cover-roadmap">
    <div class="roadmap-title">五卷十八章全栈知识版图</div>
    <table class="roadmap-table">
      <thead>
        <tr><th>卷次</th><th>核心主题</th><th>章节范围</th><th>对标官方认证</th></tr>
      </thead>
      <tbody>
        <tr><td><strong>第一卷</strong></td><td>零基础破冰：认证体系、AI 数学与浮点数、GPU/CUDA 与 Roofline、PyTorch 实战</td><td>第 1～4 章</td><td>NCA-GENL / NCA-AIIO / NCP-GENL</td></tr>
        <tr><td><strong>第二卷</strong></td><td>大模型底座与万卡训练：分词器、Transformer 手推、3D/5D 并行、SFT/LoRA/GRPO</td><td>第 5～8 章</td><td>NCA-GENL / NCP-GENL / NCP-AAI</td></tr>
        <tr><td><strong>第三卷</strong></td><td>企业 RAG 与智能体：NV-Ingest/RRF/CAGRA、LangGraph/HITL、MCP/受限解码、护栏评估</td><td>第 9～12 章</td><td>NCP-AAI (最高权重) / NCP-GENL</td></tr>
        <tr><td><strong>第四卷</strong></td><td>高性能推理与云原生：Prefill/Decode、PagedAttention、量化手算、TensorRT-LLM/Triton/NIM</td><td>第 13～15 章</td><td>NCP-GENL / NCP-AAI / NCP-AII</td></tr>
        <tr><td><strong>第五卷</strong></td><td>智算集群运维与实战冲刺：NVLink/Spectrum-X/GDS、MIG/DCGM/Xid 排障、50 题全真模拟</td><td>第 16～18 章</td><td>NCA-AIIO / NCP-AII / 全科冲刺</td></tr>
      </tbody>
    </table>
  </div>
  <div class="cover-footer">
    <div>出版编排：NVIDIA Full-Stack Engineering Curriculum Committee (2026)</div>
    <div>排版引擎：Offline KaTeX + Pygments + Chromium Skia Print Engine · A4 Print Edition</div>
  </div>
</section>
"""

    guide_html = """
<section class="guide-page">
  <div class="guide-eyebrow">HOW TO USE THIS TEXTBOOK</div>
  <h1 class="guide-title">本书使用指南（零基础学习路径与考证导航）</h1>
  <p>
    无论你是刚入门 AI 的软件/运维工程师，还是准备冲击 <strong>NVIDIA Certified Professional (NCP)</strong> 的资深开发者，
    本书均为你设计了“不跳步、可手算、能落地”的渐进式路径：
  </p>
  <div class="callout callout-green">
    <div class="callout-title">一、每章标准七步教学闭环</div>
    <ol>
      <li><strong>生活化直觉引入</strong>：先用日常比喻讲透“为什么需要这项技术”，消除专业黑话恐惧。</li>
      <li><strong>极简具体数字手推</strong>：用 2×2 或 3×4 小矩阵、具体小数与字节数逐行加减乘除，看清每一步公式背后的物理意义。</li>
      <li><strong>结构图解与对比表</strong>：通过架构框图与量化对比表建立全局空间坐标。</li>
      <li><strong>完整可运行代码实验室</strong>：全部代码附带逐行中文注释与预期输出，可直接复制到本地环境验证。</li>
      <li><strong>初学者高频误区诊所</strong>：集中拆解 4 个以上最容易栽跟头的真实考题/工程陷阱。</li>
      <li><strong>双轨考点映射</strong>：明确标注哪些概念属于入门级（<code>NCA</code>）必会，哪些深入到专业级（<code>NCP</code>）精算与排障。</li>
      <li><strong>卷末实战习题与逐项解析</strong>：每卷末尾均附带完整分步推导答案，巩固闭环。</li>
    </ol>
  </div>
  <div class="callout callout-blue">
    <div class="callout-title">二、两条推荐通关学习路线</div>
    <ul>
      <li><strong>路线 A（零基础双证稳妥线 · 8 周）</strong>：第 1～4 章（筑基）→ 第 5～6 章 → 第 9 章 → 第 16～17 章核心概念 → 第 18 章 Q1～Q15，先取 <code>NCA-GENL</code> 或 <code>NCA-AIIO</code>；随后精读第 7～8、10～15 章并刷通 Q16～Q50，拿下 <code>NCP-GENL</code> / <code>NCP-AAI</code> / <code>NCP-AII</code>。</li>
      <li><strong>路线 B（工程基础直冲专业级 · 4 周）</strong>：略过第 1～2 章基础算术部分，重点突破第 3 章 Roofline、第 6 章 KV Cache/RoPE/FlashAttention、第 7 章 16&Psi; 与 3D 并行、第 9～12 章 Agentic 全栈、第 13～15 章 PagedAttention/量化/NIM、第 16～17 章集群网络与 Xid 排障，直冲 <code>NCP</code> 全真 50 题。</li>
    </ul>
  </div>
  <div class="callout callout-slate">
    <div class="callout-title">三、符号与排印约定</div>
    <p>
      书中行内参数与 CLI 命令以等宽代码块（如 <code>--mig 1</code>、<code>n_kv_heads</code>）标出；所有数学推演采用标准印刷公式排版；
      涉及生产集群主机或配置示例时统一采用规范占位符（如 <code>MisakaMikoto</code>）及 RFC 示例地址，确保开箱安全。
    </p>
  </div>
</section>
"""
    return cover_html, guide_html


def get_css_stylesheet(katex_dir: Optional[Path]) -> str:
    """组装全套印刷级 CSS（严格零 gradient，确保文件体积轻量锐利）。"""
    katex_css = ""
    if katex_dir is not None:
        raw_katex = (katex_dir / "dist" / "katex.min.css").read_text(encoding="utf-8")
        fonts_uri = (katex_dir / "dist" / "fonts").resolve().as_uri()
        katex_css = raw_katex.replace("url(fonts/", f"url({fonts_uri}/")

    pygments_css = HtmlFormatter(style="friendly", cssclass="highlight").get_style_defs(".highlight")

    custom_css = f"""
@page {{
  size: A4;
  margin: 18mm 15mm 19mm 15mm;
  @top-left {{
    content: "《{BOOK_TITLE}》（2026 全栈详尽版）";
    font-family: "Noto Sans CJK SC", sans-serif;
    font-size: 7.8pt;
    color: #64748b;
    vertical-align: bottom;
    padding-bottom: 2.5mm;
    border-bottom: 0.6pt solid #cbd5e1;
  }}
  @top-right {{
    content: "NVIDIA Associate & Professional Certification Guide";
    font-family: "Noto Sans CJK SC", sans-serif;
    font-size: 7.5pt;
    color: #15803d;
    font-weight: 600;
    vertical-align: bottom;
    padding-bottom: 2.5mm;
    border-bottom: 0.6pt solid #cbd5e1;
  }}
  @bottom-center {{
    content: "第 " counter(page) " 页 / 共 " counter(pages) " 页";
    font-family: "Noto Sans CJK SC", sans-serif;
    font-size: 8.2pt;
    color: #475569;
    vertical-align: top;
    padding-top: 2mm;
  }}
}}

@page :first {{
  margin: 14mm 15mm 14mm 15mm;
  @top-left {{ content: none; border-bottom: none; }}
  @top-right {{ content: none; border-bottom: none; }}
  @bottom-center {{ content: none; }}
}}

* {{
  box-sizing: border-box;
}}

html, body {{
  margin: 0;
  padding: 0;
  font-family: "Noto Serif CJK SC", "Noto Sans CJK SC", serif;
  font-size: 10pt;
  line-height: 1.68;
  color: #1e293b;
  background: #ffffff;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}}

/* 封面页 */
.cover-page {{
  page-break-after: always;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 262mm;
  padding: 6mm 2mm;
}}
.cover-top-bar {{
  height: 6pt;
  background-color: #16a34a;
  margin-bottom: 8mm;
}}
.cover-meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8mm;
}}
.badge-nv {{
  display: inline-block;
  background-color: #0f172a;
  color: #ffffff;
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 8.5pt;
  font-weight: 700;
  letter-spacing: 0.06em;
  padding: 4px 12px;
  border-left: 4px solid #16a34a;
}}
.badge-warn {{
  display: inline-block;
  background-color: #fef3c7;
  color: #92400e;
  border: 1px solid #f59e0b;
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 8.5pt;
  font-weight: 700;
  padding: 4px 10px;
}}
.cover-kicker {{
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 10.5pt;
  font-weight: 700;
  color: #15803d;
  letter-spacing: 0.08em;
  margin-bottom: 4mm;
}}
.cover-title {{
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 27pt;
  line-height: 1.24;
  font-weight: 800;
  color: #0f172a;
  margin: 0 0 5mm 0;
  border: none;
  padding: 0;
  page-break-before: auto !important;
}}
.cover-subtitle {{
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 12.5pt;
  line-height: 1.5;
  color: #334155;
  font-weight: 500;
  margin-bottom: 6mm;
}}
.cover-divider {{
  width: 68mm;
  height: 3.5pt;
  background-color: #16a34a;
  margin-bottom: 6mm;
}}
.cover-lead {{
  font-size: 10pt;
  color: #334155;
  background-color: #f8fafc;
  border-left: 4px solid #0f172a;
  padding: 10px 14px;
  margin: 0 0 8mm 0;
}}
.cover-roadmap {{
  margin-top: 2mm;
}}
.roadmap-title {{
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 10.5pt;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 3mm;
}}
.roadmap-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 8.6pt;
  line-height: 1.45;
}}
.roadmap-table th {{
  background-color: #0f172a;
  color: #ffffff;
  font-family: "Noto Sans CJK SC", sans-serif;
  text-align: left;
  padding: 5px 8px;
  border: 1px solid #0f172a;
}}
.roadmap-table td {{
  padding: 5px 8px;
  border: 1px solid #cbd5e1;
  background-color: #ffffff;
}}
.roadmap-table tr:nth-child(even) td {{
  background-color: #f8fafc;
}}
.cover-footer {{
  margin-top: 8mm;
  padding-top: 4mm;
  border-top: 1px solid #cbd5e1;
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 8.2pt;
  color: #64748b;
  display: flex;
  justify-content: space-between;
}}

/* 使用指南页 */
.guide-page {{
  page-break-after: always;
}}
.guide-eyebrow {{
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 8.5pt;
  font-weight: 700;
  color: #15803d;
  letter-spacing: 0.1em;
}}
.guide-title {{
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 18pt;
  font-weight: 800;
  color: #0f172a;
  border-bottom: 2.5px solid #16a34a;
  padding-bottom: 4px;
  margin: 3px 0 12px 0;
  page-break-before: auto !important;
}}

/* 目录页 */
.toc-section {{
  page-break-after: always;
}}
.toc-banner {{
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 8.5pt;
  font-weight: 700;
  color: #15803d;
  letter-spacing: 0.1em;
}}
.toc-heading {{
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 18pt;
  font-weight: 800;
  color: #0f172a;
  border-bottom: 2.5px solid #0f172a;
  padding-bottom: 4px;
  margin: 3px 0 10px 0;
  page-break-before: auto !important;
}}
.toc-list {{
  list-style: none;
  padding-left: 0;
  margin: 0;
}}
.toc-item {{
  margin: 2px 0;
  line-height: 1.42;
}}
.toc-item a {{
  color: #1e293b;
  text-decoration: none;
}}
.toc-l1, .toc-volume {{
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 10.2pt;
  font-weight: 700;
  color: #0f172a;
  background-color: #f0fdf4;
  border-left: 3.5px solid #16a34a;
  padding: 4px 8px;
  margin-top: 8px;
  margin-bottom: 3px;
}}
.toc-l1 a, .toc-volume a {{
  color: #0f172a;
}}
.toc-l2, .toc-chapter {{
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 9.4pt;
  font-weight: 700;
  padding-left: 12px;
  margin-top: 4px;
}}
.toc-l2 a, .toc-chapter a {{
  color: #15803d;
}}
.toc-l3 {{
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 8.6pt;
  padding-left: 26px;
  color: #475569;
}}

/* 正文标题体系 */
h1, h2, h3, h4 {{
  font-family: "Noto Sans CJK SC", sans-serif;
  color: #0f172a;
  break-after: avoid;
  page-break-after: avoid;
}}
h1.volume-title {{
  page-break-before: always;
  background-color: #0f172a;
  color: #ffffff;
  font-size: 17pt;
  line-height: 1.35;
  font-weight: 800;
  padding: 12px 16px;
  border-left: 6px solid #16a34a;
  margin: 0 0 14px 0;
}}
h2.chapter-title {{
  page-break-before: always;
  font-size: 15pt;
  line-height: 1.35;
  font-weight: 800;
  color: #0f172a;
  background-color: #f0fdf4;
  border-left: 5px solid #16a34a;
  border-bottom: 1.5px solid #bbf7d0;
  padding: 8px 12px;
  margin: 0 0 12px 0;
}}
h2.chapter-title.no-page-break {{
  page-break-before: auto;
  margin-top: 10px;
}}
h2 {{
  font-size: 13.5pt;
  border-bottom: 1.5px solid #cbd5e1;
  padding-bottom: 3px;
  margin: 18px 0 10px 0;
}}
h3 {{
  font-size: 11.4pt;
  font-weight: 700;
  color: #15803d;
  margin: 14px 0 6px 0;
}}
h4 {{
  font-size: 10.3pt;
  font-weight: 700;
  color: #1e293b;
  margin: 10px 0 4px 0;
}}
p {{
  margin: 0 0 8px 0;
  text-align: justify;
}}
ul, ol {{
  margin: 4px 0 10px 0;
  padding-left: 20px;
}}
li {{
  margin-bottom: 3px;
}}
hr {{
  border: none;
  border-top: 1px solid #cbd5e1;
  margin: 14px 0;
}}

/* Callout 与引用框（纯色边框+底色） */
.callout {{
  padding: 10px 13px;
  margin: 10px 0;
  border-radius: 3px;
  break-inside: avoid;
}}
.callout-title {{
  font-family: "Noto Sans CJK SC", sans-serif;
  font-weight: 700;
  font-size: 10pt;
  margin-bottom: 5px;
  color: #0f172a;
}}
.callout-green {{
  background-color: #f0fdf4;
  border: 1px solid #bbf7d0;
  border-left: 4px solid #16a34a;
}}
.callout-blue {{
  background-color: #eff6ff;
  border: 1px solid #bfdbfe;
  border-left: 4px solid #2563eb;
}}
.callout-slate {{
  background-color: #f8fafc;
  border: 1px solid #cbd5e1;
  border-left: 4px solid #0f172a;
}}
blockquote {{
  margin: 10px 0;
  padding: 8px 12px;
  background-color: #f0fdf4;
  border: 1px solid #dcfce7;
  border-left: 4px solid #16a34a;
  color: #1e293b;
}}
blockquote p:last-child {{
  margin-bottom: 0;
}}

/* 表格样式：多页表格自动重复表头、单行不断裂 */
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 10px 0 12px 0;
  font-size: 8.9pt;
  line-height: 1.48;
  table-layout: auto;
  word-wrap: break-word;
}}
thead {{
  display: table-header-group;
}}
tr {{
  break-inside: avoid;
  page-break-inside: avoid;
}}
th {{
  background-color: #0f172a;
  color: #ffffff;
  font-family: "Noto Sans CJK SC", sans-serif;
  font-weight: 700;
  text-align: left;
  padding: 6px 8px;
  border: 1px solid #334155;
}}
td {{
  padding: 5.5px 8px;
  border: 1px solid #cbd5e1;
  vertical-align: top;
  overflow-wrap: break-word;
}}
tbody tr:nth-child(even) td {{
  background-color: #f8fafc;
}}

/* 行内代码与代码实验室卡片（长代码块自动折行绝不溢出版心） */
code.inline-code, p code, li code, td code, th code {{
  font-family: "Noto Sans Mono CJK SC", "Noto Sans Mono", monospace;
  font-size: 8.7pt;
  background-color: #f1f5f9;
  color: #0f172a;
  border: 0.5pt solid #cbd5e1;
  padding: 1px 4px;
  border-radius: 2px;
  word-break: break-word;
}}
.code-card {{
  margin: 10px 0 12px 0;
  border: 1px solid #cbd5e1;
  border-left: 3.5px solid #16a34a;
  background-color: #f8fafc;
}}
.code-card.short-block, .diagram-box.short-block {{
  break-inside: avoid;
  page-break-inside: avoid;
}}
.code-header {{
  display: flex;
  justify-content: space-between;
  background-color: #0f172a;
  color: #e2e8f0;
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 7.8pt;
  font-weight: 700;
  padding: 3px 10px;
  letter-spacing: 0.04em;
  break-after: avoid;
}}
.code-card .highlight {{
  background-color: #f8fafc;
  margin: 0;
  padding: 8px 10px;
}}
.code-card pre, pre {{
  margin: 0;
  padding: 0;
  font-family: "Noto Sans Mono CJK SC", "Noto Sans Mono", monospace;
  font-size: 8.1pt;
  line-height: 1.44;
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: anywhere;
}}

/* ASCII 框图 */
.diagram-box {{
  margin: 10px 0 12px 0;
  border: 1px solid #94a3b8;
  background-color: #f8fafc;
}}
.diagram-header {{
  background-color: #15803d;
  color: #ffffff;
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 7.8pt;
  font-weight: 700;
  padding: 3px 10px;
  letter-spacing: 0.05em;
  break-after: avoid;
}}
.ascii-diagram {{
  padding: 8px 10px;
  font-family: "Noto Sans Mono CJK SC", "Noto Sans Mono", monospace;
  font-size: 8pt;
  line-height: 1.22;
  white-space: pre-wrap;
  word-break: break-word;
  color: #0f172a;
}}

/* 内联 SVG 居中与自适应 */
.svg-wrap {{
  margin: 10px 0;
  text-align: center;
  break-inside: avoid;
}}
.svg-wrap svg {{
  max-width: 100%;
  height: auto;
}}

/* 数学公式容器与 Python 兜底样式 */
.katex-display {{
  margin: 0.7em 0 !important;
  overflow-x: visible;
  break-inside: avoid;
}}
.katex {{
  font-size: 1.03em;
}}
.math-block {{
  margin: 8px 0;
  text-align: center;
  break-inside: avoid;
}}
.pymath-frac {{
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  vertical-align: middle;
  margin: 0 2px;
  font-size: 0.9em;
}}
.pymath-num {{
  border-bottom: 1px solid #1e293b;
  padding: 0 3px 1px 3px;
}}
.pymath-den {{
  padding: 1px 3px 0 3px;
}}
.pymath-sqrt-body {{
  border-top: 1px solid #1e293b;
  padding: 0 2px;
}}
.pymath-matrix table {{
  display: inline-table;
  width: auto;
  margin: 0 3px;
  border-left: 1.5px solid #1e293b;
  border-right: 1.5px solid #1e293b;
  vertical-align: middle;
}}
.pymath-matrix td {{
  border: none;
  padding: 1px 5px;
  text-align: center;
  background: transparent !important;
}}

/* Pygments 语法高亮基础样式 */
{pygments_css}
.highlight {{ background: #f8fafc !important; }}
"""
    return f"<style>\n{katex_css}\n</style>\n<style>\n{custom_css}\n</style>"


# ---------------------------------------------------------------------------
# 核心转换函数：Markdown -> 完整 HTML
# ---------------------------------------------------------------------------
def markdown_to_html(
    md_text: str,
    katex_dir: Optional[Path],
    include_frontmatter: bool = True,
    is_sample: bool = False,
) -> str:
    """将 Markdown 全书文本安全编译为带封面、指南、自动目录与离线公式的 HTML。"""
    check_plaintext_ips(md_text)

    # 1. 保护围栏代码块
    step1_md, code_blocks = _extract_fenced_code_blocks(md_text)
    # 2. 保护行内代码
    step2_md, inline_codes = _extract_inline_codes(step1_md)
    # 3. 保护内联 SVG
    step3_md, raw_svgs = _extract_raw_svgs(step2_md)
    # 4. 提取块级和行内数学公式
    step4_md, math_items = _extract_math_expressions(step3_md)
    # 5. 转义非标准 HTML 标签（防止 <UNK>/<BOS>/<d> 消失）
    step5_md = _sanitize_unknown_angle_tags(step4_md)

    # 6. 运行 markdown-it-py
    md_parser = MarkdownIt("commonmark", {"html": True, "linkify": False, "typographer": False}).enable(
        ["table", "strikethrough"]
    )
    body_html = md_parser.render(step5_md)

    # 7. 批量渲染所有公式
    rendered_maths = batch_render_math(math_items, katex_dir)

    # 8. 还原占位符（O(N) 预计算 + 双趟正则替换）
    body_html = body_html.replace("@@ESCAPED_DOLLAR@@", "$")
    rendered_code_blocks = [_render_code_block_html(lang, code) for lang, code, _ in code_blocks]

    def _replace_wrapped_block(m: re.Match) -> str:
        kind, idx = m.group(1), int(m.group(2))
        if kind == "MATH_BLOCK" and idx < len(rendered_maths):
            return f'<div class="math-block">{rendered_maths[idx]}</div>'
        if kind == "RAW_SVG" and idx < len(raw_svgs):
            return f'<div class="svg-wrap">{raw_svgs[idx]}</div>'
        if kind == "CODE_BLOCK" and idx < len(rendered_code_blocks):
            return rendered_code_blocks[idx]
        return m.group(0)

    def _replace_remaining_token(m: re.Match) -> str:
        kind, idx = m.group(1), int(m.group(2))
        if kind in ("MATH_BLOCK", "MATH_INLINE") and idx < len(rendered_maths):
            return rendered_maths[idx]
        if kind == "RAW_SVG" and idx < len(raw_svgs):
            return f'<div class="svg-wrap">{raw_svgs[idx]}</div>'
        if kind == "CODE_BLOCK" and idx < len(rendered_code_blocks):
            return rendered_code_blocks[idx]
        if kind == "INLINE_CODE" and idx < len(inline_codes):
            return f'<code class="inline-code">{html.escape(inline_codes[idx])}</code>'
        return m.group(0)

    # Pass 1: 剥离 markdown-it-py 为独立块级占位符包裹的孤立 <p>...</p>
    body_html = re.sub(
        r"<p>@@(MATH_BLOCK|RAW_SVG|CODE_BLOCK)_(\d+)@@</p>",
        _replace_wrapped_block,
        body_html,
    )
    # Pass 2: 替换剩余行内与嵌套占位符
    body_html = re.sub(
        r"@@(MATH_BLOCK|MATH_INLINE|RAW_SVG|CODE_BLOCK|INLINE_CODE)_(\d+)@@",
        _replace_remaining_token,
        body_html,
    )

    # 9. 注入标题锚点并自动生成多级目录
    body_html, toc_items = _add_heading_ids_and_extract_toc(body_html)

    front_html = ""
    if include_frontmatter:
        cover_html, guide_html = build_cover_and_guide_html(is_sample=is_sample)
        toc_html = build_toc_html(toc_items)
        front_html = cover_html + "\n" + guide_html + "\n" + toc_html

    styles = get_css_stylesheet(katex_dir)
    full_html = (
        "<!DOCTYPE html>\n"
        '<html lang="zh-CN">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{BOOK_TITLE}（2026 全栈详尽版）</title>\n"
        f"{styles}\n"
        "</head>\n"
        "<body>\n"
        f"{front_html}\n"
        '<main class="book-content">\n'
        f"{body_html}\n"
        "</main>\n"
        "</body>\n"
        "</html>\n"
    )
    return full_html


# ---------------------------------------------------------------------------
# Chrome Headless PDF 编译 + 页数/体积门禁 + Ghostscript 保底压缩
# ---------------------------------------------------------------------------
def render_html_to_pdf(html_content: str, output_pdf: Path) -> None:
    """使用无头 Chrome 将 HTML 渲染为 A4 PDF，若体积 >= 24 MB 自动启用 Ghostscript 压缩。"""
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".html", encoding="utf-8", delete=False) as tf:
        tf.write(html_content)
        html_path = Path(tf.name)

    raw_pdf = output_pdf.with_suffix(".raw.pdf")
    try:
        cmd = [
            CHROME_BIN,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--allow-file-access-from-files",
            "--no-pdf-header-footer",
            "--virtual-time-budget=2000",
            f"--print-to-pdf={raw_pdf}",
            html_path.resolve().as_uri(),
        ]
        subprocess.run(cmd, check=True, timeout=300)

        size_bytes = raw_pdf.stat().st_size
        print(f"[INFO] Chrome 初始输出 PDF 体积: {size_bytes:,} bytes ({size_bytes / 1_048_576:.2f} MB)")

        if size_bytes < MAX_PDF_BYTES:
            shutil.move(str(raw_pdf), str(output_pdf))
        else:
            gs_bin = shutil.which("gs")
            if not gs_bin:
                raise RuntimeError(f"PDF 体积 {size_bytes} >= {MAX_PDF_BYTES} 且未找到 Ghostscript (gs)！")
            compressed = False
            for preset in ("/prepress", "/ebook", "/screen"):
                tmp_gs = output_pdf.with_suffix(f".gs{preset.strip('/')}.pdf")
                print(f"[INFO] 体积超限，尝试 Ghostscript ({preset}) 压缩...")
                subprocess.run(
                    [
                        gs_bin, "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5",
                        f"-dPDFSETTINGS={preset}", "-dCompressFonts=true",
                        "-dSubsetFonts=true", "-dNOPAUSE", "-dQUIET", "-dBATCH",
                        f"-sOutputFile={tmp_gs}", str(raw_pdf),
                    ],
                    check=True, timeout=600,
                )
                gs_size = tmp_gs.stat().st_size
                print(f"[INFO] Ghostscript ({preset}) 后体积: {gs_size:,} bytes ({gs_size / 1_048_576:.2f} MB)")
                if gs_size < MAX_PDF_BYTES:
                    shutil.move(str(tmp_gs), str(output_pdf))
                    compressed = True
                    break
                tmp_gs.unlink(missing_ok=True)
            raw_pdf.unlink(missing_ok=True)
            if not compressed:
                raise RuntimeError("即使使用最高压缩比，PDF 仍 >= 24,000,000 bytes！")
    finally:
        html_path.unlink(missing_ok=True)
        if raw_pdf.exists():
            raw_pdf.unlink(missing_ok=True)


def get_pdf_pages(pdf_path: Path) -> int:
    """调用 pdfinfo 获取 PDF 总页数。"""
    out = subprocess.check_output(["pdfinfo", str(pdf_path)], text=True)
    m = re.search(r"^Pages:\s+(\d+)", out, flags=re.M)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Phase 1 单元测试样张内容
# ---------------------------------------------------------------------------
SAMPLE_TEST_MARKDOWN = r"""# 第一卷：零基础破冰篇——数学直觉、Transformer 与 GPU 底层手推

## 第 1 章：排版、中文、公式与代码综合单元测试

> **核心结论（Callout 样式测试）**：在现代大语言模型（LLM）推理中，一次前向注意力计算的缩放点积公式为 $\text{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$，其中 $d_k$ 和 $d_{kv}$ 绝对不能出现下划线断裂成 `<em>` 的情况！

### 1.1 货币符号保护 vs 行内与块级公式测试

假设一台 H100 云主机租金从 $2.5/小时 降到 $1.8/小时（或者预算区间在 $100 到 $250 之间），这属于纯货币文本，绝对不能被错误识别为数学公式！而下面的符号则是真正的数学公式：混合精度静态显存为 $16\Psi$，KV 头维度记为 $d_{kv}$，组相对优势为 $A_i = \frac{r_i - \mu}{\sigma}$，RRF 融合得分为 $\sum \frac{1}{k + r_i}$。

缩放点积注意力与 KV Cache 显存精确公式（多行块级公式）：

$$
\text{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

$$
\text{Mem}_{KV} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times s \times b \times p_{\text{bytes}}
$$

SmoothQuant 数学等价平滑变换与 RoPE 旋转矩阵：

$$
Y = (X \operatorname{diag}(s)^{-1})(\operatorname{diag}(s) W), \qquad
R_{\Theta,m} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}
$$

### 1.2 表格、特殊 Token 与 ASCII 框图测试

下表展示 `FP16`、`BF16` 与 `FP8`（`E4M3` / `E5M2`）精度对比，并验证 `<UNK>`、`<BOS>`、`<EOS>` 等特殊标记不会在 HTML 渲染中被隐藏：

| 数值格式 | 符号位 (S) | 指数位 (E) | 尾数位 (M) | 核心适用场景与工程特性 |
| :--- | :---: | :---: | :---: | :--- |
| `FP32` | 1 | 8 | 23 | Master Weights 与 Optimizer States ($m, v$)，数值极稳 |
| `BF16` | 1 | 8 | 7 | 大模型训练黄金格式，动态范围与 FP32 完全一致 |
| `FP8 (E4M3)` | 1 | 4 | 3 | Hopper/Blackwell 前向推理与权重激活，精度优先 |
| `FP8 (E5M2)` | 1 | 5 | 2 | 反向梯度传输，动态范围更广防溢出 |

```text
┌───────────────────────────┐       NVLink 900 GB/s        ┌───────────────────────────┐
│   GPU 0 (H100 80GB SXM)   │ ◄──────────────────────────► │   GPU 1 (H100 80GB SXM)   │
│  [TP Rank 0 | KV Slice 0] │     AllReduce (2x / Layer)   │  [TP Rank 1 | KV Slice 1] │
└───────────────────────────┘                              └───────────────────────────┘
```

### 1.3 长行代码自动折行与语法高亮测试

```python
import math
import torch

def scaled_dot_product_attention_demo(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    '''手写验证缩放点积注意力：包含超长注释以测试自动折行 word-break 绝不溢出 A4 右边界，同时验证 d_k 与 KV Cache 维度对齐'''
    d_k = q.size(-1)
    # 注意这里也有 $100 和 _underscore_ 变量名，代码块内绝对不能触发数学公式替换！
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)
```
"""


def build_combined_markdown() -> Path:
    """拼接卷首导言与 5 卷正文，写入合并后的全书 Markdown 文件。"""
    missing = [v for v in VOLUME_FILES if not (CHAPTERS_DIR / v).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少以下分卷 Markdown 文件，无法进行全书合订: {missing}")

    front_md = f"""# 《{BOOK_TITLE}（2026 全栈详尽版）》

> **{BOOK_EDITION}**
>
> 全书涵盖五卷十八章：**第一卷（零基础破冰）**、**第二卷（大模型底座与万卡训练）**、**第三卷（企业级 RAG 与 Agentic AI）**、**第四卷（高性能推理、量化与 Triton/NIM 部署）**、**第五卷（智算集群硬件网络、云原生排障与 50 题全真通关题库）**。

---

"""
    parts = [front_md]
    for fname in VOLUME_FILES:
        vol_text = (CHAPTERS_DIR / fname).read_text(encoding="utf-8").strip()
        parts.append(vol_text)
        parts.append("\n\n---\n\n")

    combined_text = "\n".join(parts).strip() + "\n"
    out_md = OUTPUT_DIR_DRAFT / f"{BOOK_FILENAME_BASE}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(combined_text, encoding="utf-8")
    print(f"[INFO] 已生成全书合订 Markdown: {out_md} ({out_md.stat().st_size:,} bytes)")
    return out_md


def fix_permissions(paths: List[Path]) -> None:
    """尝试将产出文件与工程目录归属设为 magnus51235608:magnus51235608。"""
    if os.geteuid() != 0:
        return
    for p in paths:
        if p.exists():
            subprocess.run(["chown", "-R", "magnus51235608:magnus51235608", str(p)], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="《英伟达认证工程师从零精通标准课本》PDF 编译脚本")
    parser.add_argument("--test-sample", action="store_true", help="运行 Phase 1 样张单元测试 -> /tmp/test_sample_math.pdf")
    parser.add_argument("--input", type=Path, help="指定单个输入 Markdown 路径")
    parser.add_argument("--output-pdf", type=Path, help="指定输出 PDF 路径")
    parser.add_argument("--dump-html", type=Path, help="可选：保存中间 HTML 供排查")
    args = parser.parse_args()

    katex_dir = find_katex_dir()
    print(f"[INFO] KaTeX 离线路径: {katex_dir if katex_dir else '未找到（启用纯 Python 兜底）'}")

    if args.test_sample:
        out_pdf = args.output_pdf or Path("/tmp/test_sample_math.pdf")
        html_doc = markdown_to_html(SAMPLE_TEST_MARKDOWN, katex_dir, include_frontmatter=True, is_sample=True)
        if args.dump_html:
            args.dump_html.write_text(html_doc, encoding="utf-8")
        render_html_to_pdf(html_doc, out_pdf)
        pages = get_pdf_pages(out_pdf)
        size_b = out_pdf.stat().st_size
        print(f"[PASS] Sample PDF 编译成功: {out_pdf} | {pages} pages | {size_b:,} bytes ({size_b / 1_048_576:.2f} MB)")
        return 0

    if args.input:
        md_text = args.input.read_text(encoding="utf-8")
        out_pdf = args.output_pdf or args.input.with_suffix(".pdf")
        html_doc = markdown_to_html(md_text, katex_dir, include_frontmatter=True, is_sample=False)
        if args.dump_html:
            args.dump_html.write_text(html_doc, encoding="utf-8")
        render_html_to_pdf(html_doc, out_pdf)
        pages = get_pdf_pages(out_pdf)
        size_b = out_pdf.stat().st_size
        print(f"[PASS] PDF 编译成功: {out_pdf} | {pages} pages | {size_b:,} bytes ({size_b / 1_048_576:.2f} MB)")
        return 0

    # Phase 2：全书 5 卷合订编译
    out_md = build_combined_markdown()
    # 注意：正文渲染不包含 front_md 顶部重复的 `# 《书名》` 避免重复封面，直接由 cover + guide + toc + 5 卷构成
    vol_parts = [(CHAPTERS_DIR / f).read_text(encoding="utf-8").strip() for f in VOLUME_FILES]
    full_body_md = "\n\n".join(vol_parts)

    html_doc = markdown_to_html(full_body_md, katex_dir, include_frontmatter=True, is_sample=False)
    if args.dump_html:
        args.dump_html.write_text(html_doc, encoding="utf-8")

    out_pdf_draft = OUTPUT_DIR_DRAFT / f"{BOOK_FILENAME_BASE}.pdf"
    out_pdf_review = OUTPUT_DIR_REVIEW / f"{BOOK_FILENAME_BASE}.pdf"

    render_html_to_pdf(html_doc, out_pdf_draft)
    pages = get_pdf_pages(out_pdf_draft)
    size_b = out_pdf_draft.stat().st_size

    out_pdf_review.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(out_pdf_draft), str(out_pdf_review))
    fix_permissions([PROJECT_DIR, out_md, out_pdf_draft, out_pdf_review])

    print(
        f"[PASS] 全书合订完成: MD={out_md.stat().st_size:,} bytes | "
        f"PDF={pages} pages, {size_b:,} bytes ({size_b / 1_048_576:.2f} MB) | "
        f"已同步至 {out_pdf_review}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
