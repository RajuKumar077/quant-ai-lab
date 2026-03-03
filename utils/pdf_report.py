from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _safe_text(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    if isinstance(value, (float, np.floating)):
        if np.isfinite(float(value)):
            return f"{float(value):,.4f}"
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def _normalize_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        out = value.to_frame(name=value.name or "value")
    elif isinstance(value, pd.DataFrame):
        out = value.copy()
    else:
        out = pd.DataFrame()
    if out.empty:
        return out
    if out.index.name is not None or not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index()
    return out


def _to_cell_paragraph(value: Any, style: ParagraphStyle) -> Paragraph:
    text = escape(_safe_text(value))
    return Paragraph(text, style)


def _build_table(
    frame: pd.DataFrame,
    max_rows: int,
    body_style: ParagraphStyle,
    header_style: ParagraphStyle,
    table_header_bg=colors.HexColor("#1F2937"),
) -> list[Table]:
    shown = frame.head(max_rows)
    max_cols_per_table = 6
    col_groups = [shown.columns[i : i + max_cols_per_table] for i in range(0, len(shown.columns), max_cols_per_table)]
    tables = []

    for cols in col_groups:
        data = [[_to_cell_paragraph(col, header_style) for col in cols]]
        for _, row in shown[cols].iterrows():
            data.append([_to_cell_paragraph(v, body_style) for v in row.values])

        usable_width = A4[0] - 3.0 * cm
        col_widths = [usable_width / len(cols)] * len(cols)
        tbl = Table(data, colWidths=col_widths, repeatRows=1, splitByRow=True)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), table_header_bg),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        tables.append(tbl)
    return tables


def _chunked_table_story(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    max_rows: int,
    body_style: ParagraphStyle,
    heading_style: ParagraphStyle,
    subheading_style: ParagraphStyle,
    header_cell_style: ParagraphStyle,
) -> list:
    story = [Paragraph(title, heading_style), Paragraph(subtitle, subheading_style)]
    frame = _normalize_frame(df)
    if frame.empty:
        story.extend([Paragraph("No records available.", body_style), Spacer(1, 0.3 * cm)])
        return story

    tables = _build_table(frame, max_rows=max_rows, body_style=body_style, header_style=header_cell_style)
    for idx, tbl in enumerate(tables, start=1):
        if len(tables) > 1:
            story.append(Paragraph(f"Column view {idx}/{len(tables)}", body_style))
        story.append(tbl)
        story.append(Spacer(1, 0.22 * cm))

    if len(frame) > max_rows:
        story.append(Paragraph(f"Rows shown: {max_rows} of {len(frame)}.", body_style))
    story.append(Spacer(1, 0.32 * cm))
    return story


def _render_fig_to_image(fig, width_cm=17.2, height_cm=6.0) -> Image:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width_cm * cm, height=height_cm * cm)


def _extract_reco_counts(reco_df: pd.DataFrame) -> pd.Series:
    df = _normalize_frame(reco_df)
    if df.empty:
        return pd.Series(dtype=float)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.Series(dtype=float)
    cols = [c for c in numeric.columns if str(c).lower() in {"strongbuy", "buy", "hold", "sell", "strongsell"}]
    target = numeric[cols] if cols else numeric
    sums = target.sum().sort_values(ascending=False)
    return sums[sums > 0]


def _extract_holders_split(major_holders_df: pd.DataFrame) -> pd.Series:
    df = _normalize_frame(major_holders_df)
    if df.empty:
        return pd.Series(dtype=float)
    percentages = {}
    for _, row in df.head(8).iterrows():
        vals = [str(v) for v in row.values if isinstance(v, str)]
        if len(vals) < 2:
            continue
        pct = next((v for v in vals if "%" in v), None)
        label = next((v for v in vals if "%" not in v), None)
        if pct and label:
            try:
                percentages[label[:40]] = float(pct.replace("%", "").strip())
            except Exception:
                continue
    if not percentages:
        return pd.Series(dtype=float)
    return pd.Series(percentages)


def _visual_story(chart_data: dict[str, Any], heading_style: ParagraphStyle, body_style: ParagraphStyle) -> list:
    story: list[Any] = [Paragraph("Visual Analytics", heading_style), Spacer(1, 0.1 * cm)]
    close_df = _normalize_frame(chart_data.get("close_prices"))
    returns_df = _normalize_frame(chart_data.get("returns"))
    scorecard_df = _normalize_frame(chart_data.get("scorecard"))
    risk_metrics = chart_data.get("risk_metrics", {}) or {}
    ratio_snapshot = chart_data.get("ratio_snapshot", {}) or {}
    recommendations = _normalize_frame(chart_data.get("recommendations"))
    major_holders = _normalize_frame(chart_data.get("major_holders"))

    close_col = "close" if "close" in close_df.columns else (close_df.columns[-1] if not close_df.empty else None)
    ret_col = "return" if "return" in returns_df.columns else (returns_df.columns[-1] if not returns_df.empty else None)

    chart_count = 0
    if close_col is not None and len(close_df) > 5:
        c = pd.to_numeric(close_df[close_col], errors="coerce").dropna()
        x = pd.to_datetime(close_df.loc[c.index, close_df.columns[0]], errors="coerce") if len(close_df.columns) > 1 else np.arange(len(c))
        x = x if hasattr(x, "__len__") and len(x) == len(c) else np.arange(len(c))

        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(x, c.values, color="#2563EB", linewidth=2)
        ax.set_title("1) Closing Price Trend", fontsize=11, loc="left")
        ax.grid(alpha=0.2)
        story.extend([_render_fig_to_image(fig), Spacer(1, 0.12 * cm)])
        chart_count += 1

        roll_vol = c.pct_change().rolling(30).std() * np.sqrt(252)
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(roll_vol.index, roll_vol.values, color="#0EA5E9", linewidth=2)
        ax.set_title("2) 30-Day Rolling Volatility", fontsize=11, loc="left")
        ax.grid(alpha=0.2)
        story.extend([_render_fig_to_image(fig), Spacer(1, 0.12 * cm)])
        chart_count += 1

        drawdown = c / c.cummax() - 1
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.fill_between(drawdown.index, drawdown.values, 0, color="#EF4444", alpha=0.6)
        ax.set_title("3) Drawdown Curve", fontsize=11, loc="left")
        ax.grid(alpha=0.2)
        story.extend([_render_fig_to_image(fig), Spacer(1, 0.12 * cm)])
        chart_count += 1

    if ret_col is not None and len(returns_df) > 20:
        r = pd.to_numeric(returns_df[ret_col], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.hist(r.values, bins=35, color="#7C3AED", alpha=0.8, edgecolor="white")
        ax.set_title("4) Return Distribution Histogram", fontsize=11, loc="left")
        ax.grid(alpha=0.2)
        story.extend([_render_fig_to_image(fig), Spacer(1, 0.12 * cm)])
        chart_count += 1

        mx = (1 + r).resample("M").prod() - 1 if isinstance(r.index, pd.DatetimeIndex) else pd.Series(dtype=float)
        if not mx.empty:
            fig, ax = plt.subplots(figsize=(10, 3.2))
            colors_arr = np.where(mx.values >= 0, "#16A34A", "#DC2626")
            ax.bar(mx.index.astype(str), mx.values, color=colors_arr)
            ax.set_title("5) Monthly Return Bars", fontsize=11, loc="left")
            ax.tick_params(axis="x", rotation=60, labelsize=7)
            ax.grid(alpha=0.2, axis="y")
            story.extend([_render_fig_to_image(fig), Spacer(1, 0.12 * cm)])
            chart_count += 1

    if not scorecard_df.empty and {"Category", "Score"}.issubset(scorecard_df.columns):
        s = scorecard_df.copy()
        s["Score"] = pd.to_numeric(s["Score"], errors="coerce")
        s = s.dropna(subset=["Score"]).sort_values("Score", ascending=False)
        if not s.empty:
            fig, ax = plt.subplots(figsize=(10, 3.2))
            ax.barh(s["Category"], s["Score"], color="#F59E0B")
            ax.set_title("6) Hedge Fund Scorecard", fontsize=11, loc="left")
            ax.invert_yaxis()
            ax.grid(alpha=0.2, axis="x")
            story.extend([_render_fig_to_image(fig), Spacer(1, 0.12 * cm)])
            chart_count += 1

    risk_keys = ["historical_var", "parametric_var", "annualized_volatility", "max_drawdown"]
    risk_vals = {k: abs(float(risk_metrics.get(k))) for k in risk_keys if pd.notna(risk_metrics.get(k))}
    if risk_vals:
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.bar(list(risk_vals.keys()), list(risk_vals.values()), color="#334155")
        ax.set_title("7) Risk Metric Magnitudes", fontsize=11, loc="left")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(alpha=0.2, axis="y")
        story.extend([_render_fig_to_image(fig), Spacer(1, 0.12 * cm)])
        chart_count += 1

    ratio_labels = ["gross_margin", "operating_margin", "net_margin", "roe", "roic"]
    ratio_vals = {k: float(ratio_snapshot.get(k)) for k in ratio_labels if pd.notna(ratio_snapshot.get(k))}
    if ratio_vals:
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(list(ratio_vals.keys()), list(ratio_vals.values()), marker="o", color="#0F766E", linewidth=2)
        ax.set_title("8) Profitability Profile", fontsize=11, loc="left")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(alpha=0.2)
        story.extend([_render_fig_to_image(fig), Spacer(1, 0.12 * cm)])
        chart_count += 1

    reco_counts = _extract_reco_counts(recommendations)
    if not reco_counts.empty:
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.bar(reco_counts.index.astype(str), reco_counts.values, color="#3B82F6")
        ax.set_title("9) Analyst Recommendation Mix", fontsize=11, loc="left")
        ax.grid(alpha=0.2, axis="y")
        story.extend([_render_fig_to_image(fig), Spacer(1, 0.12 * cm)])
        chart_count += 1
    else:
        holders = _extract_holders_split(major_holders)
        if not holders.empty:
            fig, ax = plt.subplots(figsize=(10, 3.2))
            ax.pie(holders.values, labels=holders.index, autopct="%1.1f%%", startangle=90)
            ax.set_title("9) Holder Structure Split", fontsize=11, loc="left")
            story.extend([_render_fig_to_image(fig), Spacer(1, 0.12 * cm)])
            chart_count += 1

    story.append(Paragraph(f"Charts included: {chart_count}", body_style))
    story.append(PageBreak())
    return story


def generate_company_findings_pdf(report: dict[str, Any]) -> bytes:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleApple",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "SectionApple",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#111827"),
        spaceBefore=6,
        spaceAfter=6,
    )
    subheading_style = ParagraphStyle(
        "Subheading",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#475569"),
        leading=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyApple",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1F2937"),
        wordWrap="CJK",
    )
    header_cell_style = ParagraphStyle(
        "HeaderCell",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.2 * cm,
        rightMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )

    story = []
    ticker = report.get("ticker", "N/A")
    date_span = report.get("date_span", "N/A")
    fy_label = report.get("fiscal_period", "Custom")

    story.append(Paragraph(f"{ticker} - Corporate Findings Report", title_style))
    story.append(Paragraph(f"Fiscal Window: {fy_label} | Market Data Range: {date_span}", subheading_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subheading_style))
    story.append(Spacer(1, 0.35 * cm))

    summary_items = report.get("summary_items", [])
    summary_data = [[_to_cell_paragraph("Key Metric", header_cell_style), _to_cell_paragraph("Value", header_cell_style)]]
    for k, v in summary_items:
        summary_data.append([_to_cell_paragraph(k, body_style), _to_cell_paragraph(v, body_style)])
    summary_table = Table(summary_data, colWidths=[8.5 * cm, 8.5 * cm], repeatRows=1, splitByRow=True)
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#94A3B8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 0.35 * cm)])

    analyst_summary = report.get("analyst_summary", "N/A")
    story.append(Paragraph("Analyst Summary", section_style))
    story.append(Paragraph(escape(_safe_text(analyst_summary)), body_style))
    story.append(PageBreak())

    chart_data = report.get("chart_data", {})
    story.extend(_visual_story(chart_data, section_style, body_style))

    core_sections = report.get("core_sections", {})
    for title, frame in core_sections.items():
        story.extend(
            _chunked_table_story(
                df=frame,
                title=title,
                subtitle="Core analysis output",
                max_rows=40,
                body_style=body_style,
                heading_style=section_style,
                subheading_style=subheading_style,
                header_cell_style=header_cell_style,
            )
        )

    story.append(PageBreak())
    story.append(Paragraph("Raw Findings Appendix", section_style))
    story.append(Paragraph("Source extracts by dataset for audit and archival use.", body_style))
    story.append(Spacer(1, 0.2 * cm))

    raw_sections = report.get("raw_sections", {})
    max_rows = int(report.get("raw_max_rows", 60))
    for name, frame in raw_sections.items():
        normalized = _normalize_frame(frame)
        subtitle = f"Rows: {len(normalized):,} | Columns: {len(normalized.columns):,}" if not normalized.empty else "No records"
        story.extend(
            _chunked_table_story(
                df=normalized,
                title=name,
                subtitle=subtitle,
                max_rows=max_rows,
                body_style=body_style,
                heading_style=section_style,
                subheading_style=subheading_style,
                header_cell_style=header_cell_style,
            )
        )

    def _draw_footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(1.2 * cm, 0.7 * cm, f"AlphaForge Quant Terminal | {ticker}")
        canvas.drawRightString(A4[0] - 1.2 * cm, 0.7 * cm, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    buffer.seek(0)
    return buffer.getvalue()
