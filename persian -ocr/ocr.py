import os
import easyocr
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT


desktop = os.path.join(os.path.expanduser("~"), "Desktop")

images = [f for f in os.listdir(desktop) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if not images:
    raise FileNotFoundError("not founds ")
img_path = os.path.join(desktop, images[0])
print(f"proccing: {img_path}")

reader = easyocr.Reader(['fa','ar'], gpu=False)
result = reader.readtext(img_path)

text_lines = [line[1].strip() for line in result if line[1].strip() != '']
text = "\n".join(text_lines)

print("result")
print(text)

txt_path = os.path.join(desktop, "output.txt")
with open(txt_path, "w", encoding="utf-8-sig") as f:
    f.write(text)
print(f"done {txt_path}")

doc = Document()
for line in text_lines:
    p = doc.add_paragraph(line)
    p.alignment = WD_PARAGRAPH_ALIGNMENT.RIGHT 
    run = p.runs[0]
    run.font.name = 'Tahoma'  
    r = run._element
    r.rPr.rFonts.set(qn('w:eastAsia'), 'Tahoma')
    run.font.size = Pt(12)

docx_path = os.path.join(desktop, "output.docx")
doc.save(docx_path)
print(f"docx created {docx_path}")
