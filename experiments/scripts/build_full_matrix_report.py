from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(r"C:\Users\isclin\Desktop\Summer Project")
REPORT_DIR = ROOT / "04_reports_and_deliverables" / "final_experiment_report_2026-07-07"
TABLE_DIR = REPORT_DIR / "tables"
FIG_DIR = REPORT_DIR / "figures"
MD_PATH = REPORT_DIR / "PoisonedRAG_full_report_2026-07-07.md"
PDF_PATH = REPORT_DIR / "PoisonedRAG_full_report_2026-07-07.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")

DATASET_ORDER = ["NQ", "HotpotQA", "MS MARCO"]
MODEL_ORDER = ["qwen3.5:9b", "llama3.1:8b", "mistral:7b"]

METRICS = [
    ("asr", "ASR"),
    ("correct_answer_rate", "Correct-answer rate"),
    ("retrieval_success_rate", "Retrieval success"),
    ("avg_poison_in_topk", "Avg poison in top-5"),
    ("avg_poison_topk_share", "Poison top-k share"),
]


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


def read_tables():
    paths = {
        "paper": TABLE_DIR / "paper_style_full_matrix_summary.csv",
        "candidate": TABLE_DIR / "candidate_pool_full_matrix_summary.csv",
        "corpus": TABLE_DIR / "corpus_level_full_matrix_summary.csv",
    }
    for path in paths.values():
        require(path)
    return {key: pd.read_csv(path) for key, path in paths.items()}


def fmt(value, digits=2):
    return f"{float(value):.{digits}f}"


def pct(value, digits=4):
    return f"{float(value):.{digits}f}%"


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def paper_rows(paper):
    rows = []
    ordered = paper.copy()
    ordered["dataset_label"] = pd.Categorical(ordered["dataset_label"], DATASET_ORDER, ordered=True)
    ordered["model_label"] = pd.Categorical(ordered["model_label"], MODEL_ORDER, ordered=True)
    for row in ordered.sort_values(["dataset_label", "model_label"]).itertuples(index=False):
        rows.append(
            [
                row.dataset_label,
                row.model_label,
                fmt(row.asr_mean),
                fmt(row.precision_mean),
                fmt(row.recall_mean),
                fmt(row.f1_mean),
            ]
        )
    return rows


def candidate_key_rows(candidate):
    rows = []
    clean = candidate[candidate["variant"] == "clean"].copy()
    focus = candidate[candidate["variant"].isin(["original", "query_first_instruction_aware"])].copy()
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            clean_row = clean[(clean["dataset_label"] == dataset) & (clean["model_label"] == model)].iloc[0]
            subset = focus[(focus["dataset_label"] == dataset) & (focus["model_label"] == model)]
            def value(variant, rate):
                hit = subset[(subset["variant"] == variant) & (subset["poison_rate_percent"].round(6) == rate)]
                return fmt(hit["asr"].iloc[0]) if not hit.empty else "NA"

            rows.append(
                [
                    dataset,
                    model,
                    fmt(clean_row.asr),
                    value("original", 1.0),
                    value("query_first_instruction_aware", 1.0),
                    value("original", 10.0),
                    value("query_first_instruction_aware", 10.0),
                ]
            )
    return rows


def corpus_key_rows(corpus):
    rows = []
    clean = corpus[corpus["variant"] == "clean"].copy()
    focus = corpus[corpus["variant"].isin(["original", "query_first_instruction_aware"])].copy()
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            clean_row = clean[(clean["dataset_label"] == dataset) & (clean["model_label"] == model)].iloc[0]
            subset = focus[(focus["dataset_label"] == dataset) & (focus["model_label"] == model)]
            def value(variant, rate):
                rounded = subset["requested_corpus_poison_rate_percent"].round(8)
                hit = subset[(subset["variant"] == variant) & (rounded == rate)]
                return fmt(hit["asr"].iloc[0]) if not hit.empty else "NA"

            rows.append(
                [
                    dataset,
                    model,
                    fmt(clean_row.asr),
                    value("original", 0.0001),
                    value("query_first_instruction_aware", 0.0001),
                    value("original", 0.005),
                    value("query_first_instruction_aware", 0.005),
                ]
            )
    return rows


def build_markdown(paper, candidate, corpus):
    candidate_rows = candidate_key_rows(candidate)
    corpus_rows = corpus_key_rows(corpus)
    paper_table_rows = paper_rows(paper)

    candidate_original_1 = candidate[
        (candidate["variant"] == "original") & (candidate["poison_rate_percent"].round(6) == 1.0)
    ]["asr"]
    corpus_original_lowest = corpus[
        (corpus["variant"] == "original")
        & (corpus["requested_corpus_poison_rate_percent"].round(8) == 0.0001)
    ]["asr"]

    asset_rows = [
        ["Paper-style", "3 datasets x 3 LLMs", len(paper), "ASR / Precision / Recall / F1"],
        [
            "Candidate-pool",
            "3 datasets x 3 LLMs x (clean + 4 variants x 4 rates)",
            len(candidate),
            "ASR / correct-answer / retrieval / poison top-k count / poison top-k share",
        ],
        [
            "Corpus-level true ASR",
            "3 datasets x 3 LLMs x (clean + 2 variants x 4 true corpus rates)",
            len(corpus),
            "ASR / correct-answer / retrieval / poison top-k count / poison top-k share",
        ],
    ]

    metric_files = []
    for prefix in ["candidate_pool_full_matrix", "corpus_level_full_matrix"]:
        for metric, _ in METRICS:
            metric_files.append(f"tables/{prefix}_{metric}_pivot.csv")

    figure_files = [
        "figures/paper_style_fullmatrix_metrics.png",
        "figures/candidate_fullmatrix_asr_grid.png",
        "figures/candidate_fullmatrix_correct_answer_rate_grid.png",
        "figures/candidate_fullmatrix_retrieval_success_grid.png",
        "figures/candidate_fullmatrix_avg_poison_in_topk_grid.png",
        "figures/candidate_fullmatrix_avg_poison_topk_share_grid.png",
        "figures/candidate_fullmatrix_clean_baseline.png",
        "figures/corpus_fullmatrix_asr_grid.png",
        "figures/corpus_fullmatrix_correct_answer_rate_grid.png",
        "figures/corpus_fullmatrix_retrieval_success_grid.png",
        "figures/corpus_fullmatrix_avg_poison_in_topk_grid.png",
        "figures/corpus_fullmatrix_avg_poison_topk_share_grid.png",
        "figures/corpus_fullmatrix_clean_baseline.png",
    ]

    md = f"""# PoisonedRAG 本机完整复现实验报告

日期：2026-07-07  
项目目录：`{ROOT}`  
报告范围：GitHub 交付仓库和 slides 之前的论文级复现实验、低投毒率实验、corpus-level true ASR 与改进毒文档评估。

## 结论摘要

1. 已完成三数据集 x 三本地 LLM 的论文式复现矩阵：NQ、HotpotQA、MS MARCO 均覆盖 `qwen3.5:9b`、`llama3.1:8b`、`mistral:7b`。
2. 已完成三数据集 x 三本地 LLM 的 candidate-pool low-rate 矩阵：共 {len(candidate)} 行，覆盖 clean、original、authority、instruction-aware、query-first 四类攻击变体和 1%、3%、5%、10%。
3. 已完成三数据集 x 三本地 LLM 的 corpus-level true ASR 矩阵：共 {len(corpus)} 行，覆盖 clean、original、query-first 和 0.0001%、0.0005%、0.001%、0.005% 真实 corpus 投毒率。
4. candidate-pool 中 original 1% ASR 的范围为 {fmt(candidate_original_1.min())}-{fmt(candidate_original_1.max())}；低投毒率一旦进入候选池，三种 LLM 都很容易被带偏。
5. corpus-level 0.0001% true poison rate 下 original ASR 的范围为 {fmt(corpus_original_lowest.min())}-{fmt(corpus_original_lowest.max())}；这说明即使用完整语料库作分母，极少量毒文档也可能产生明显攻击效果。
6. 改进版 `query_first_instruction_aware` 有研究价值，但不是稳定最强 baseline；当前最强、最稳的 baseline 通常仍是 `original`，原因是 query anchor 对 Contriever 检索排序非常关键。

## 数据矩阵完整性

{markdown_table(["实验块", "矩阵定义", "实际行数", "指标"], asset_rows)}

所有详细矩阵已经输出为 CSV，图表也按三数据集 x 三 LLM 的 3x3 面板生成。PDF 正文展示核心表和所有指标图，完整数值以 `tables/` 下 CSV 为准。

## 论文式复现实验

设置：官方 PoisonedRAG pipeline、Contriever 检索器、top_k=5、每个数据集 100 条目标问题记录；响应 LLM 替换为本机 Ollama 模型。

{markdown_table(["Dataset", "LLM", "ASR", "Precision", "Recall", "F1"], paper_table_rows)}

![论文式三数据集三 LLM 指标](figures/paper_style_fullmatrix_metrics.png)

## Candidate-pool 低投毒率实验

candidate-pool 投毒率定义为 `n_poison / (n_poison + n_benign)`。这组实验衡量毒文档已经进入候选池后的攻击强度，不等同于完整语料库分母。

核心 ASR 摘要如下；完整 153 行矩阵见 `tables/candidate_pool_full_matrix_summary.csv`，五项指标 pivot 见下方附件表。

{markdown_table(["Dataset", "LLM", "Clean ASR", "Original 1%", "Query-first 1%", "Original 10%", "Query-first 10%"], candidate_rows)}

![Candidate ASR](figures/candidate_fullmatrix_asr_grid.png)
![Candidate correct-answer rate](figures/candidate_fullmatrix_correct_answer_rate_grid.png)
![Candidate retrieval success](figures/candidate_fullmatrix_retrieval_success_grid.png)
![Candidate avg poison in top-5](figures/candidate_fullmatrix_avg_poison_in_topk_grid.png)
![Candidate poison top-k share](figures/candidate_fullmatrix_avg_poison_topk_share_grid.png)
![Candidate clean baseline](figures/candidate_fullmatrix_clean_baseline.png)

## Corpus-level low-rate true ASR

true ASR 使用完整 BEIR corpus size 作分母。0.0001%、0.0005%、0.001%、0.005% 分别对应每个数据集语料规模下的极少量毒文档。`qwen3.5:9b` 和 `llama3.1:8b` 为避免长上下文 timeout 使用 ultrafast generation profile，`mistral:7b` 使用 fast profile；因此 corpus-level 的跨模型对比应理解为本机可复现配置下的对比。

核心 ASR 摘要如下；完整 81 行矩阵见 `tables/corpus_level_full_matrix_summary.csv`，五项指标 pivot 见下方附件表。

{markdown_table(["Dataset", "LLM", "Clean ASR", "Original 0.0001%", "Query-first 0.0001%", "Original 0.005%", "Query-first 0.005%"], corpus_rows)}

![Corpus true ASR](figures/corpus_fullmatrix_asr_grid.png)
![Corpus correct-answer rate](figures/corpus_fullmatrix_correct_answer_rate_grid.png)
![Corpus retrieval success](figures/corpus_fullmatrix_retrieval_success_grid.png)
![Corpus avg poison in top-5](figures/corpus_fullmatrix_avg_poison_in_topk_grid.png)
![Corpus poison top-k share](figures/corpus_fullmatrix_avg_poison_topk_share_grid.png)
![Corpus clean baseline](figures/corpus_fullmatrix_clean_baseline.png)

## 改进版毒文档结论

`authority` 和旧 `instruction_aware` 的长前缀会削弱 query-document similarity，导致毒文档更难进入 top-k。`query_first_instruction_aware` 把 query anchor 放回文档开头后，检索表现明显改善，并且更接近真实 RAG 攻击中“updated / verified / preferred answer”的写法。

但是，跨三数据集和三 LLM 后，query-first 并没有稳定超过 original。当前最稳的论文级 baseline 仍是 original；query-first 更适合作为“更自然、更真实但不一定更强”的研究变体，用于说明低投毒率攻击不只依赖原始短毒文档。

## 不能过度声称

- 不能说已经用原论文同一个响应 LLM 完全复现；这里是本地替代 LLM 复现。
- 不能把 candidate-pool poison rate 当作 corpus-level poison rate。
- corpus-level 中 qwen3.5 和 llama3.1 使用 ultrafast generation profile，适合本机完成矩阵，但与普通长上下文生成配置不是同一口径。
- 不能声称 query-first 改进版优于 original；当前结论是它更真实、更有研究价值，但攻击强度通常略弱。

## 后续任务

GitHub-ready 交付仓库和 slide deck 是最后两项交付工作。研究层面后续可以补 repeated seed、置信区间、minimum poison rate 曲线拟合，以及更多自然语言攻击模板。

## 附件表和图

核心 CSV：

- `tables/paper_style_full_matrix_summary.csv`
- `tables/candidate_pool_full_matrix_summary.csv`
- `tables/corpus_level_full_matrix_summary.csv`
- `tables/experiment_matrix_completion_status.csv`
- `tables/corpus_level_matrix_completion_status.csv`

指标 pivot：

{chr(10).join("- `" + item + "`" for item in metric_files)}

图表：

{chr(10).join("- `" + item + "`" for item in figure_files)}
"""
    MD_PATH.write_text(md, encoding="utf-8-sig")
    return md


def pdf_cell(text, style):
    return Paragraph(str(text), style)


def build_table(data, widths=None, font_size=7.6):
    styles = getSampleStyleSheet()
    cell = ParagraphStyle(
        "TableCellCN",
        parent=styles["BodyText"],
        fontName="SimHei",
        fontSize=font_size,
        leading=font_size + 2.2,
        wordWrap="CJK",
    )
    rows = [[pdf_cell(item, cell) for item in row] for row in data]
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "SimHei"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C7CDD8")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("SimHei", 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawString(1.5 * cm, 1.05 * cm, "PoisonedRAG 本机完整复现实验报告")
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.05 * cm, f"第 {doc.page} 页")
    canvas.restoreState()


def figure(path, width=17.2 * cm):
    require(path)
    img = Image(str(path))
    aspect = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * aspect
    return img


def build_pdf(paper, candidate, corpus):
    if not FONT_PATH.exists():
        raise FileNotFoundError(FONT_PATH)
    pdfmetrics.registerFont(TTFont("SimHei", str(FONT_PATH)))

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName="SimHei",
        fontSize=20,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    h1 = ParagraphStyle(
        "H1CN",
        parent=styles["Heading1"],
        fontName="SimHei",
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=7,
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName="SimHei",
        fontSize=9.2,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=5,
        wordWrap="CJK",
    )
    note = ParagraphStyle(
        "NoteCN",
        parent=body,
        fontSize=8.4,
        leading=12,
        textColor=colors.HexColor("#4B5563"),
    )

    story = []
    story.append(Paragraph("PoisonedRAG 本机完整复现实验报告", title))
    story.append(Paragraph("日期：2026-07-07。范围：论文级复现、低投毒率实验、corpus-level true ASR、改进毒文档评估；不含最终 slide deck。", body))

    story.append(Paragraph("结论摘要", h1))
    for text in [
        "完成 paper-style、candidate-pool、corpus-level 三个实验块的 3 数据集 x 3 LLM 矩阵。",
        f"candidate-pool 详细矩阵 {len(candidate)} 行；corpus-level true ASR 详细矩阵 {len(corpus)} 行；paper-style 复现矩阵 {len(paper)} 行。",
        "所有核心指标均有三数据集 x 三 LLM 图：ASR、正确答案率、检索成功率、top-5 毒文档数量、top-5 毒文档占比。",
        "query-first 改进版更自然，但没有稳定超过 original；当前 original 仍是最稳 baseline。",
    ]:
        story.append(Paragraph("· " + text, body))

    story.append(Paragraph("数据矩阵完整性", h1))
    story.append(
        build_table(
            [
                ["实验块", "矩阵定义", "实际行数"],
                ["Paper-style", "3 datasets x 3 LLMs", len(paper)],
                ["Candidate-pool", "3 datasets x 3 LLMs x (clean + 4 variants x 4 rates)", len(candidate)],
                ["Corpus-level", "3 datasets x 3 LLMs x (clean + 2 variants x 4 true rates)", len(corpus)],
            ],
            widths=[3.4 * cm, 10.2 * cm, 2.2 * cm],
            font_size=8,
        )
    )

    story.append(Paragraph("Paper-style 复现矩阵", h1))
    story.append(
        build_table(
            [["Dataset", "LLM", "ASR", "Precision", "Recall", "F1"]] + paper_rows(paper),
            widths=[2.7 * cm, 3.3 * cm, 2.2 * cm, 2.4 * cm, 2.2 * cm, 2.0 * cm],
            font_size=7.4,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(figure(FIG_DIR / "paper_style_fullmatrix_metrics.png"))

    story.append(PageBreak())
    story.append(Paragraph("Candidate-pool 核心 ASR 摘要", h1))
    story.append(Paragraph("完整 153 行矩阵在 tables/candidate_pool_full_matrix_summary.csv；下表只抽取 clean、1%、10% 的 ASR 方便快速阅读。", note))
    story.append(
        build_table(
            [["Dataset", "LLM", "Clean", "Orig 1%", "QF 1%", "Orig 10%", "QF 10%"]] + candidate_key_rows(candidate),
            widths=[2.2 * cm, 3.0 * cm, 1.8 * cm, 1.9 * cm, 1.9 * cm, 2.0 * cm, 2.0 * cm],
            font_size=7.0,
        )
    )

    for filename, title_text in [
        ("candidate_fullmatrix_asr_grid.png", "Candidate-pool ASR"),
        ("candidate_fullmatrix_correct_answer_rate_grid.png", "Candidate-pool 正确答案率"),
        ("candidate_fullmatrix_retrieval_success_grid.png", "Candidate-pool 检索成功率"),
        ("candidate_fullmatrix_avg_poison_in_topk_grid.png", "Candidate-pool top-5 毒文档数量"),
        ("candidate_fullmatrix_avg_poison_topk_share_grid.png", "Candidate-pool top-5 毒文档占比"),
        ("candidate_fullmatrix_clean_baseline.png", "Candidate-pool clean baseline"),
    ]:
        story.append(PageBreak())
        story.append(Paragraph(title_text, h1))
        story.append(figure(FIG_DIR / filename))

    story.append(PageBreak())
    story.append(Paragraph("Corpus-level true ASR 核心摘要", h1))
    story.append(Paragraph("完整 81 行矩阵在 tables/corpus_level_full_matrix_summary.csv；qwen3.5 和 llama3.1 使用 ultrafast profile，mistral 使用 fast profile。", note))
    story.append(
        build_table(
            [["Dataset", "LLM", "Clean", "Orig 0.0001%", "QF 0.0001%", "Orig 0.005%", "QF 0.005%"]] + corpus_key_rows(corpus),
            widths=[2.2 * cm, 3.0 * cm, 1.7 * cm, 2.25 * cm, 2.25 * cm, 2.25 * cm, 2.25 * cm],
            font_size=6.5,
        )
    )

    for filename, title_text in [
        ("corpus_fullmatrix_asr_grid.png", "Corpus-level true ASR"),
        ("corpus_fullmatrix_correct_answer_rate_grid.png", "Corpus-level 正确答案率"),
        ("corpus_fullmatrix_retrieval_success_grid.png", "Corpus-level 检索成功率"),
        ("corpus_fullmatrix_avg_poison_in_topk_grid.png", "Corpus-level top-5 毒文档数量"),
        ("corpus_fullmatrix_avg_poison_topk_share_grid.png", "Corpus-level top-5 毒文档占比"),
        ("corpus_fullmatrix_clean_baseline.png", "Corpus-level clean baseline"),
    ]:
        story.append(PageBreak())
        story.append(Paragraph(title_text, h1))
        story.append(figure(FIG_DIR / filename))

    story.append(PageBreak())
    story.append(Paragraph("结论和后续", h1))
    for text in [
        "candidate-pool 和 corpus-level 的结果共同说明：低比例投毒在检索进入 top-k 后可以显著影响本地 LLM 输出。",
        "query-first 改进版比旧 instruction-aware 更像真实攻击文本，但强度通常略低于 original。",
        "下一步交付层面只剩 GitHub-ready 仓库和 slide deck；研究层面可补 repeated seed、confidence interval 和 minimum poison rate 曲线拟合。",
    ]:
        story.append(Paragraph("· " + text, body))

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=1.35 * cm,
        rightMargin=1.35 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.65 * cm,
    )
    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    return PDF_PATH


def main():
    tables = read_tables()
    build_markdown(tables["paper"], tables["candidate"], tables["corpus"])
    build_pdf(tables["paper"], tables["candidate"], tables["corpus"])
    print(MD_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
