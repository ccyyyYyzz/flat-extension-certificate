import os
import re
import shutil

ROOT = r"D:\woooorld\paper\prl"
OUT = os.path.join(ROOT, "arxiv")
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

main = open(os.path.join(ROOT, "main.tex"), encoding="utf-8").read()
supp = open(os.path.join(ROOT, "supplement.tex"), encoding="utf-8").read()

# --- main: split preamble / body; keep its bibliography in place ---
m_pre, m_body = main.split("\\begin{document}", 1)
m_body = m_body.split("\\end{document}")[0]

# --- supplement: body between \maketitle and its bibliography ---
s_body = supp.split("\\maketitle", 1)[1]
s_body = s_body.split("\\bibliography{references}")[0]

# --- supplement-only preamble extras ---
extras = [
    r"\usepackage{array}",
    r"\newcommand{\F}{\mathbb F}",
    r"\newcommand{\op}{\mathrm{op}}",
    r"\newcommand{\id}{\mathrm{id}}",
    r"\newcommand{\eps}{\varepsilon}",
    r"\newcommand{\cN}{\mathcal N}",
    r"\newcommand{\cR}{\mathcal R}",
    r"\newcommand{\cP}{\mathcal P}",
    r"\newcommand{\cS}{\mathcal S}",
    r"\newcommand{\cC}{\mathcal C}",
    r"\newcommand{\cD}{\mathcal D}",
    r"\newcommand{\cH}{\mathcal H}",
    r"\newcommand{\cO}{\mathcal O}",
    (r"\newcommand{\hashfour}[4]{\texttt{#1}\allowbreak\texttt{#2}"
     r"\allowbreak\texttt{#3}\allowbreak\texttt{#4}}"),
    r"\newtheorem{proposition}{Proposition}",
    r"\newtheorem{theorem}{Theorem}",
    r"\newtheorem{corollary}{Corollary}",
    r"\renewcommand{\theproposition}{S\arabic{proposition}}",
    r"\renewcommand{\thetheorem}{S\arabic{theorem}}",
    r"\renewcommand{\thecorollary}{S\arabic{corollary}}",
]

seam = "\n".join([
    "",
    r"\clearpage",
    r"\onecolumngrid",
    r"\begin{center}",
    (r"{\large\bfseries Supplemental Material for ``Single-Slot Tomography "
     r"Cannot Self-Certify Multitime Response Dimension''}"),
    r"\end{center}",
    r"\setcounter{equation}{0}",
    r"\setcounter{figure}{0}",
    r"\setcounter{table}{0}",
    r"\setcounter{section}{0}",
    r"\renewcommand{\theequation}{S\arabic{equation}}",
    r"\renewcommand{\thetable}{S\Roman{table}}",
    r"\renewcommand{\thefigure}{S\arabic{figure}}",
    r"\renewcommand{\thesection}{S\arabic{section}}",
    "",
])

merged = (
    m_pre.rstrip()
    + "\n" + "\n".join(extras) + "\n"
    + "\\begin{document}"
    + m_body.rstrip()
    + "\n" + seam
    + s_body.rstrip()
    + "\n\\end{document}\n"
)

# sanity: no leftover xr machinery, exactly one documentclass
assert "externaldocument" not in merged
assert merged.count("\\documentclass") == 1
assert merged.count("\\maketitle") == 1
dups = [n for n in re.findall(r"\\newcommand\{\\(\w+)\}", merged)
        if merged.count("\\newcommand{\\%s}" % n) > 1]
assert not dups, dups

with open(os.path.join(OUT, "ms.tex"), "w", encoding="utf-8",
          newline="\n") as f:
    f.write(merged)

shutil.copy(os.path.join(ROOT, "references.bib"),
            os.path.join(OUT, "references.bib"))
shutil.copy(os.path.join(ROOT, "figures", "c7_flat_extension.pdf"),
            os.path.join(OUT, "figures", "c7_flat_extension.pdf"))
print("wrote", os.path.join(OUT, "ms.tex"), len(merged), "chars")
