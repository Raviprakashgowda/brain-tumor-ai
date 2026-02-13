from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image, Table, TableStyle, PageBreak
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import os


# ---------------- Page Border ----------------
def draw_page_border(canvas, doc):
    canvas.saveState()
    width, height = letter
    margin = 20
    canvas.setStrokeColor(colors.darkblue)
    canvas.setLineWidth(2)
    canvas.rect(margin, margin, width - 2 * margin, height - 2 * margin)
    canvas.restoreState()


# ---------------- PDF Generator ----------------
def generate_pdf(output_path, patient_data, image_paths):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        fontSize=18,
        textColor=colors.darkblue,
        alignment=1,
        spaceAfter=8
    )

    section_style = ParagraphStyle(
        "Section",
        fontSize=13,
        textColor=colors.darkblue,
        spaceBefore=12,
        spaceAfter=6,
        bold=True
    )

    normal = styles["Normal"]
    small = ParagraphStyle("Small", fontSize=9, textColor=colors.grey)

    elements = []

    # ---------------- Header ----------------
    logo_path = "static/images/logo.png"
    if os.path.exists(logo_path):
        logo = Image(logo_path, width=1 * inch, height=1 * inch)
    else:
        logo = Paragraph("", normal)

    header = Table([
        [logo, Paragraph("TumorDetect AI Diagnostic Center", title_style)]
    ], colWidths=[80, 420])

    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    elements.append(header)
    elements.append(Spacer(1, 10))

    # ---------------- Patient Information ----------------
    elements.append(Paragraph("Patient Information", section_style))

    patient_table = [
        ["Patient ID", patient_data.get("Patient ID", "")],
        ["Patient Name", patient_data.get("Patient Name", "")],
        ["Age", patient_data.get("Age", "")],
        ["Gender", patient_data.get("Gender", "")],
        ["MRI Date", patient_data.get("MRI Date", "")]
    ]

    table = Table(patient_table, colWidths=[200, 300])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
    ]))
    elements.append(table)

    # ---------------- Diagnosis Summary ----------------
    elements.append(Paragraph("Diagnosis Summary", section_style))

    summary_data = [
        ["Tumor Type", patient_data.get("Tumor Type", "N/A")],
        ["Risk Level", patient_data.get("Risk Level", "N/A")],
        ["Model Confidence", patient_data.get("Model Confidence", "N/A")],
    ]

    table = Table(summary_data, colWidths=[200, 300])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
    ]))
    elements.append(table)

    # ---------------- Tumor Measurements ----------------
    elements.append(Paragraph("Tumor Measurements", section_style))

    measurement_data = [
        ["Location", str(patient_data.get("Tumor Location", "N/A"))],
        ["Width", str(patient_data.get("Tumor Width", "N/A"))],
        ["Height", str(patient_data.get("Tumor Height", "N/A"))],
        ["Segmented Area", str(patient_data.get("Tumor Area", "N/A"))],
    ]

    table = Table(measurement_data, colWidths=[200, 300])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
    ]))
    elements.append(table)

    # ---------------- Clinical Impression ----------------
    elements.append(Paragraph("Clinical Impression", section_style))
    impression_text = (
        f"Findings suggest the presence of "
        f"{patient_data.get('Tumor Type', 'a tumor')}."
        " Further radiological and clinical evaluation is recommended."
    )
    elements.append(Paragraph(impression_text, normal))

    # ---------------- AI Explanation ----------------
    elements.append(Paragraph("AI Explanation", section_style))

    explanation_data = [
        ["Description", patient_data.get("Description", "N/A")],
        ["Possible Cause", patient_data.get("Possible Cause", "N/A")],
        ["Treatment", patient_data.get("Treatment", "N/A")]
    ]

    table = Table(explanation_data, colWidths=[200, 300])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
    ]))
    elements.append(table)

    # ---------------- Page Break ----------------
    elements.append(PageBreak())

    # ---------------- MRI Images Page ----------------
    elements.append(Paragraph("MRI Analysis Images", section_style))

    image_cells = []
    temp_row = []

    for label, path in image_paths.items():
        if os.path.exists(path):
            img = Image(path, width=2.3 * inch, height=2.3 * inch)
            temp_row.append([Paragraph(label, small), img])

            if len(temp_row) == 2:
                image_cells.append(temp_row)
                temp_row = []

    if temp_row:
        image_cells.append(temp_row)

    image_table_data = []
    for row in image_cells:
        labels = [cell[0] for cell in row]
        images = [cell[1] for cell in row]
        image_table_data.append(labels)
        image_table_data.append(images)

    image_table = Table(image_table_data, colWidths=[250, 250])
    elements.append(image_table)

    elements.append(Spacer(1, 20))

    # ---------------- Signature ----------------
    elements.append(Paragraph("Authorized By:", section_style))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("AI Diagnostic System", normal))
    elements.append(Paragraph("TumorDetect AI", normal))

    elements.append(Spacer(1, 20))

    # ---------------- Disclaimer ----------------
    elements.append(Paragraph("Disclaimer", section_style))
    elements.append(Paragraph(
        "This report is generated by an AI-based diagnostic system "
        "for academic and research purposes only.",
        small
    ))

    doc.build(elements, onFirstPage=draw_page_border, onLaterPages=draw_page_border)
