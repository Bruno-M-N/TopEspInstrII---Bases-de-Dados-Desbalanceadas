import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

"""
Parse and convert scikit-learn classification_report to latex 
(Python 3/Booktabs) 
cfreportlatex.py (modified)
Author: Francisco Rodrigues (modified)
https://gist.github.com/FRodrigues21/bec41ee4305c027bcdf9987313182e9b

Code to parse sklearn classification_report
Original: https://gist.github.com/julienr/6b9b9a03bd8224db7b4f
Modified to work with Python 3 and classification report averages
"""

import sys
import collections

def parse_classification_report(clfreport):
    """
    Parse a sklearn classification report into a dict keyed by class name
    and containing a tuple (precision, recall, fscore, support) for each class
    """
    lines = clfreport.split('\n')
    # Remove empty lines
    lines = list(filter(lambda l: not len(l.strip()) == 0, lines))

    # Starts with a header, then score for each class and finally an average
    header = lines[0]
    cls_lines = lines[1:-1]
    avg_line = lines[-1]

    assert header.split() == ['precision', 'recall', 'f1-score', 'support']
    assert avg_line.split()[1] == 'avg'

    # We cannot simply use split because class names can have spaces. So instead
    # figure the width of the class field by looking at the indentation of the
    # precision header
    cls_field_width = len(header) - len(header.lstrip())
    # Now, collect all the class names and score in a dict
    def parse_line(l):
        """Parse a line of classification_report"""
        cls_name = l[:cls_field_width].strip()
        if (len(l[cls_field_width:].split()) == 4):
          precision, recall, fscore, support = l[cls_field_width:].split()
          precision = float(precision)
          recall = float(recall)
          fscore = float(fscore)
          support = int(support)
        else:
          fscore, support = l[cls_field_width:].split()
          precision = str("")
          recall = str("")
          fscore = float(fscore)
          support = int(support)
        return (cls_name, precision, recall, fscore, support)

    data = collections.OrderedDict()
    for l in cls_lines:
        ret = parse_line(l)
        cls_name = ret[0]
        scores = ret[1:]
        data[cls_name] = scores

    # average
    data['avg'] = parse_line(avg_line)[1:]

    return data

def report_to_latex_table(data, model):
    classificationReportFileName = "classification-report-" + model #.tex
    title = f"Classification report para o {model}"
    caption = title + r". \\ \textbf{Fonte -} Autor."
    label = "tab: " + classificationReportFileName  #.tex
    avg_split = False
    out =  f"%{classificationReportFileName}\n"
    out += (r"% A Tabela \ref{" f"{label}" r"} exibe o \textit{" f"{title}"
                 r"}" + "\n")
    out += (r"% \input{Tabelas/" f"{classificationReportFileName}" r"}" + "\n")
    out += (r"% Please add the following required packages to your"
                 r"document preamble:" + "\n")
    out += r"% \usepackage{booktabs}" + "\n"
    out += "\\begin{table}[H]\n"
    out += "    " + "\\centering\n"
    out += "    " + r"\begin{tabular}{@{}ccccc@{}}" + "\n"
    out += "    " + "\\toprule\n"
    out += ("    " + r" & \textbf{Precision} & \textbf{Recall} &"
            + r"\textbf{F-score} & \textbf{Support} \\ \midrule " + "\n")
    for cls, scores in data.items():
        if 'micro' in cls:
            out += "\\midrule\n"
        out += "    " + cls + " & " + " & ".join([str(s) for s in scores])
        out += r" \\ " + "\n"
    out += "    " + "\\end{tabular}\n"
    out += ("    " + r"\caption{" + caption + r"}" + "\n")
    out += ("    " + r"\label{" + label + r"}" + "\n")
    out += "\\end{table}"
    return out
    
def printConfusionMatrix(y_true, y_pred, model):
  # print(y_test)
  # print(y_pred)
  # extract the predicted class labels
  y_pred_class = np.where(y_pred > 0.5, 1, 0)
  confmat = confusion_matrix(y_true = y_true, y_pred = y_pred_class)
  # print(confmat)
  print(f"Matriz de confusão para o {model}")
  print(f"Verdadeiro positivo:  {confmat[0,0]}" 
        f" Falso negativo:      {confmat[0,1]}\n"
        f"Falso positivo:       {confmat[1,0]}" 
        f" Verdadeiro negativo: {confmat[1,1]}")
  
  confusionMatrixFileName = "matriz-confusao-" + model #.tex
  title = f"Matriz de confusão para o {model}"
  caption = title + r". \\ \textbf{Fonte -} Autor."
  label = "tab: " + confusionMatrixFileName  #.tex
  header = (r'& \textbf{} & \multicolumn{2}{c}{\textbf{Rótulo Verdadeiro}}'
            + r'\\ \midrule')
  
  tableTEX =  f"%{confusionMatrixFileName}\n"
  tableTEX += (r"% A Tabela \ref{" f"{label}" r"} exibe a " f"{title}\n")
  tableTEX += (r"% \input{Tabelas/" f"{confusionMatrixFileName}" r"}" + "\n")
  tableTEX += (r"% Please add the following required packages to your"
               r"document preamble:" + "\n")
  tableTEX += r"% \usepackage{booktabs}" + "\n"
  tableTEX += r"% \usepackage{multirow}" + "\n"
  tableTEX += r"\begin{table}[H]" + "\n"
  tableTEX += ("    " + r"\centering" + "\n")
  tableTEX += ("    " + r"\begin{tabular}{@{}cccc@{}}" + "\n")
  tableTEX += ("    " + r"\toprule" + "\n")
  tableTEX += ("    " + header + "\n")
  tableTEX += ("    " + r" &  & \begin{tabular}[c]{@{}c@{}}Classe\\ 0"
               + r"\end{tabular} & \begin{tabular}[c]{@{}c@{}}Classe\\  1"
               + r"\end{tabular} \\" + "\n")
  tableTEX += ("    " + r"\multirow{2}{*}{\textbf{\begin{tabular}[c]{@{}c@{}}"
               + r"Rótulo\\  Predito\end{tabular}}} & "
               + r"\begin{tabular}[c]{@{}c@{}}Classe\\ 0\end{tabular} & "
               + f"{confmat[0, 0]} & {confmat[0, 1]}" + r" \\ " + "\n")
  tableTEX += ("    " + r"& \begin{tabular}[c]{@{}c@{}}Classe\\  "
               + r"1\end{tabular} & "
               + f"{confmat[1, 0]} & {confmat[1, 1]}" + r" \\ \bottomrule"
                + "\n")
  tableTEX += ("    " + r"\end{tabular}" + "\n")
  tableTEX += ("    " + r"\caption{" + caption + r"}" + "\n")
  tableTEX += ("    " + r"\label{" + label + r"}" + "\n")
  tableTEX += "\end{table}" + "\n"

  arquivo = open(pathTab + confusionMatrixFileName + '.tex','w')
  arquivo.write(tableTEX)
  arquivo.close()

def printClassificationReport(y_true, y_pred, model):
  print(f"Classification report para o {model}")
  # extract the predicted class labels
  y_pred_class = np.where(y_pred > 0.5, 1, 0)
  print(classification_report(y_true, y_pred_class))
  
  data = parse_classification_report(classification_report(y_true,
                                                           y_pred_class))
  # print(report_to_latex_table(data, model))
  
  classificationReportFileName = "classification-report-" + model #.tex
  arquivo = open(pathTab + classificationReportFileName + '.tex','w')
  arquivo.write(report_to_latex_table(data, model))
  arquivo.close()