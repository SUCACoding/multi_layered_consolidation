# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 12:17:14 2025

@author: SUCA
"""

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import matplotlib.pyplot as plt

def fig_to_img_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def create_pdf_report(strat, set_frac, idx_frac, UPore_Calc, sett, t_step_sum, idx_cons,
                      figs_and_titles, soil_layers_table, cons_deg_table, boundary, dZ):
    pdf_buffer = BytesIO()

    # Page size and margins
    left_margin = right_margin = 72  # 1 inch margins
    PAGE_WIDTH, PAGE_HEIGHT = A4
    available_width = PAGE_WIDTH - left_margin - right_margin

    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4,
                            leftMargin=left_margin, rightMargin=right_margin)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Consolidation Calculation Report", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "This tool allows engineers to simulate and predict time-dependent settlements in multi-layered soils "
        "under applied loads. It accepts input parameters such as layer thickness, compressibility, permeability, "
        "initial void ratio, and loading conditions. By solving consolidation equations for each layer while "
        "accounting for varying drainage conditions and layer interactions, this tool provides accurate predictions "
        "of settlement magnitude and rate.",
        styles['Normal']
    ))
    story.append(Spacer(1, 12))
                 
    text = f"For this calculation, the boundary has been set to <b>{boundary}</b> and the soil layer increments to <i>{dZ:.2f}m</i>."
    story.append(Paragraph(text, styles['Normal']))
    # Soil Layers Table
    story.append(Paragraph("Soil Layers Summary Table", styles['Heading2']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("""
    Below table presents the provided input values for:<br/>
    &#8226; <b>Name</b>: Identifier for the soil layer (e.g., clay, sand, silt).<br/>
    &#8226; <b>Elevation</b>: Top and bottom elevations (meters) defining the thickness. Ensure no overlap.<br/>
    &#8226; <b>Void Ratio (e)</b>: Measure of the volume of voids relative to solids in the soil.<br/>
    &#8226; <b>Permeability (k)</b>: Hydraulic conductivity (m/s), indicating how easily water flows through the soil.
    """, styles['Normal']))
    story.append(Spacer(1, 12)
                 )                
    data = [soil_layers_table.columns.to_list()] + soil_layers_table.values.tolist()
    n_cols = len(data[0])
    col_widths = [available_width / n_cols] * n_cols  # Equal columns
    table = Table(data, colWidths=col_widths)

    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ])
    table.setStyle(style)
    story.append(table)

    # Consolidation Degree Table
    story.append(Paragraph("Consolidation Degree Summary Table", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(
        "Below consolidation rates have been selected for the documentation. The Consolidation Degree is calculated"
        "based on the surface settlement over time divided by the total settlement at full consolidation"
        "The Consolidation Degree times have been used in all future graphs.",
        styles['Normal']
    ))
    story.append(Spacer(1, 12))
    
    cons_deg_table = cons_deg_table.reset_index()
    cons_deg_table = cons_deg_table.rename(columns={'index': 'Degree of Consolidation'})
    data = [cons_deg_table.columns.to_list()] + cons_deg_table.values.tolist()
    n_cols = len(data[0])

    # Wider first column for long text
    first_col_width = 120  # points - adjust if needed
    remaining_width = available_width - first_col_width
    other_col_width = remaining_width / (n_cols - 1)
    col_widths = [first_col_width] + [other_col_width] * (n_cols - 1)

    table2 = Table(data, colWidths=col_widths)
    table2.setStyle(style)
    story.append(table2)

    # Figures
    story.append(Paragraph("Graphs", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(
        "Below, the different graphs for the resulting analysis has been depicted for the different Consolidation Degrees",
        styles['Normal']
    ))
    
    for title, fig in figs_and_titles:
        story.append(Paragraph(title, styles['Heading3']))
        img_buf = fig_to_img_bytes(fig)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            tmpfile.write(img_buf.read())
            tmpfile.flush()
            img = RLImage(tmpfile.name, width=450, height=300)
            story.append(img)
            story.append(Spacer(1, 24))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()