# app/report_generator.py

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime

def generate_pdf(name, age, answers_dict, prediction):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - 50, "Mental Health Assessment Report")

    # User Info
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 90, f"Name: {name}")
    c.drawString(300, height - 90, f"Age: {age}")
    c.drawString(50, height - 110, f"Date: {datetime.today().strftime('%B %d, %Y')}")

    # Section Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 140, "Assessment Summary:")

    # Questions and Answers
    y = height - 165
    c.setFont("Helvetica", 11)
    for question, answer in answers_dict.items():
        lines = c.beginText(50, y)
        lines.setFont("Helvetica", 11)
        lines.textLines(f"{question}\nAnswer: {answer}")
        c.drawText(lines)
        y -= 40
        if y < 100:
            c.showPage()
            y = height - 50

    # Final Result
    c.setFont("Helvetica-Bold", 13)
    y -= 30
    if prediction == "Yes":
        result_text = (
            "Based on your responses, it appears that you may benefit from mental health support. "
            "We encourage you to seek guidance from a mental health professional and explore available resources. "
            "You are not alone, and support is available."
        )
        c.setFillColorRGB(1, 0, 0)
        c.drawString(50, y, "Final Prediction: Needs Support")
    else:
        result_text = (
            "Great job! You are currently maintaining good mental health. Keep following healthy habits "
            "and being mindful of your mental well-being."
        )
        c.setFillColorRGB(0, 0.5, 0)
        c.drawString(50, y, "Final Prediction: Mentally Healthy")

    y -= 25
    c.setFillColorRGB(0, 0, 0)
    text = c.beginText(50, y)
    text.setFont("Helvetica", 11)
    text.textLines(result_text)
    c.drawText(text)

    # Save PDF
    c.save()
    buffer.seek(0)
    return buffer
