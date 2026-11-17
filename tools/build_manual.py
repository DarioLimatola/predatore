"""Render the release manual and validation report (optional ReportLab dependency)."""
from html import escape
from pathlib import Path
import re
import sys
import os

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                Preformatted, Table, TableStyle, KeepTogether)

ROOT=Path(__file__).resolve().parents[1]
BLUE=colors.HexColor('#163a56')
TEAL=colors.HexColor('#007975')
INK=colors.HexColor('#172d3d')
PALE=colors.HexColor('#eff5f7')


font_root=Path(os.environ.get('PREDATORE_FONT_DIR','/usr/share/fonts/truetype/dejavu'))
if (font_root/'DejaVuSans.ttf').is_file():
    faces=('DejaVuSans.ttf','DejaVuSans-Bold.ttf','DejaVuSansMono.ttf')
else:
    font_root=Path(reportlab.__file__).parent/'fonts'
    faces=('Vera.ttf','VeraBd.ttf','Vera.ttf')
for name,face in zip(('BodyFace','HeadingFace','CodeFace'),faces):
    pdfmetrics.registerFont(TTFont(name,str(font_root/face)))


def style(name,size,leading,**kwargs):
    return ParagraphStyle(name,fontName='BodyFace',fontSize=size,
                          leading=leading,textColor=INK,**kwargs)


BODY=style('Body',10.3,15,spaceAfter=8)
TITLE=style('Title',28,34,spaceBefore=10,spaceAfter=18,keepWithNext=True)
H2=style('Heading',17,22,spaceBefore=16,spaceAfter=9,keepWithNext=True)
H3=style('Subheading',12,16,spaceBefore=9,spaceAfter=6,keepWithNext=True)
SMALL=style('Table',8.3,11)
WHITE=style('Table header',8.3,11)
WHITE.textColor=colors.white
CODE=style('Code',8.5,12,spaceBefore=5,spaceAfter=11,backColor=PALE,borderPadding=7)
CODE.fontName='CodeFace'
for st in (TITLE,H2,H3,WHITE):st.fontName='HeadingFace'


def markup(line):
    return re.sub(r'\x60([^\x60]+)\x60',r'<font face="CodeFace">\1</font>',
                  escape(line.replace('→',' > ').replace('≤','<=').replace('≥','>=')))


def content(path, compact=False):
    body=style("ReportBody",9.8,13.6,spaceAfter=6) if compact else BODY
    lines=path.read_text(encoding='utf-8').splitlines()
    i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('#'):
            head,title=line.split(' ',1)
            yield Paragraph(markup(title),{'#':TITLE,'##':H2,'###':H3}.get(head,H3))
            i+=1;continue
        if line.startswith('    '):
            block=[]
            while i<len(lines) and lines[i].startswith('    '):
                block.append(lines[i][4:]);i+=1
            yield KeepTogether([Preformatted('\n'.join(block),CODE,maxLineLength=88)]);continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                row=[v.strip() for v in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',v) for v in row):rows.append(row)
                i+=1
            cols=len(rows[0]);width=487
            table=Table([[Paragraph(markup(v),WHITE if n==0 else SMALL) for v in row]
                         for n,row in enumerate(rows)],repeatRows=1,colWidths=[width/cols]*cols,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BLUE),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[PALE,colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('TOPPADDING',(0,0),(-1,-1),6 if compact else 7),('BOTTOMPADDING',(0,0),(-1,-1),6 if compact else 7),
                ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6)]))
            yield table;yield Spacer(1,9);continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||    |- |\d+\. )',lines[i]):
            paragraph.append(lines[i].strip());i+=1
        yield Paragraph(markup(' '.join(paragraph)),body)


def frame(canvas,doc):
    w,h=A4
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#cbdbe2'));canvas.line(54,47,w-54,47)
    canvas.setFont('BodyFace',8.5);canvas.setFillColor(BLUE)
    canvas.drawString(54,h-33,'MY AI  /  PREDATORE V6.05')
    canvas.drawString(54,32,'Manuale tecnico e rapporto di verifica')
    canvas.drawRightString(w-54,32,str(doc.page))
    canvas.restoreState()


def main():
    dest=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'docs/Predatore_V6_05_Manuale.pdf'
    doc=SimpleDocTemplate(str(dest),pagesize=A4,leftMargin=54,rightMargin=54,
          topMargin=56,bottomMargin=56,title='Predatore V6.05 - Manuale e report',author='Progetto My AI')
    cover=style('Cover',39,46,spaceAfter=20);cover.textColor=BLUE;cover.fontName='HeadingFace'
    label=style('Label',11,16,spaceAfter=18);label.textColor=TEAL
    story=[Spacer(1,94),Paragraph('LABORATORIO LOGICO-BINARIO',label),
           Paragraph('Predatore<br/>V6.05',cover),
           Paragraph('Manuale del simulatore e del motore booleano vettoriale',H2),
           Spacer(1,12),Paragraph('EVA apprende dagli episodi e dai docenti. SMITH organizza esperienza e strategie tra generazioni e run.',BODY),
           Spacer(1,23),Paragraph('INSTALLAZIONE · ARCHITETTURA · API · VALIDAZIONE',label),
           Paragraph('Il report finale include i confronti 2D, la continuità tra run e il profilo dei costi.',BODY),
           PageBreak(),*content(ROOT/'docs/MANUALE.md'),PageBreak(),
           *content(ROOT/'validation/v605/REPORT.md',compact=True)]
    doc.build(story,onFirstPage=frame,onLaterPages=frame)


if __name__=='__main__':main()
