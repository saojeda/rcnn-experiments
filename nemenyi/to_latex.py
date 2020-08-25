import math

import numpy as np


def normalize_by_width(ranks, width):
    return [x * width / len(ranks) for x in ranks]


def get_segments(ranks, cd):
    conected_points = []
    for i in range(len(ranks)):
        p2 = -1
        for j in range(i + 1, len(ranks)):
            if ranks[j] - ranks[i] < cd:
                p2 = j
        if p2 != -1:
            conected_points.append([i, p2])

    found = True

    while found:
        point_to_remove = findInner(conected_points, ranks)
        if point_to_remove != -1:
            conected_points.remove(point_to_remove)
        else:
            found = False

    return conected_points


def findInner(segments, ranks):
    for i in segments:
        for j in segments:
            if i != j:
                if ranks[i[0]] >= ranks[j[0]] and ranks[i[1]] <= ranks[j[1]]:
                    return i
                if ranks[i[0]] <= ranks[j[0]] and ranks[i[1]] >= ranks[j[1]]:
                    return j
    return -1


def writeTex(names, ranks, cd, output_file, caption, width=7):

    names_sorted = [x for _, x in sorted(zip(ranks, names))]
    ranks_sorted = sorted(ranks)

    segments = get_segments(ranks_sorted, cd)

    ranks_normalized = normalize_by_width(ranks_sorted, width - 1)

    lineFormat = (
        "line width=0.5mm, black"
    )

    points = range(1, len(ranks) + 1)
    points = normalize_by_width(points, width - 1)

    normCd = cd * (width - 1.0) / len(names)
    lastRank = len(names) - 1
    margin = 0.5
    leftLabelPosition = margin
    rightLabelPosition = margin + len(ranks) * (width - 1.0) / len(ranks)

    text_script = "\\begin{figure}\n\\centering\n\\begin{tikzpicture}[xscale=2]\n"

    text_script += (
        "\\node (Label) at ({}, 0.7)"
        "{{\\tiny{{CD = {:.2f}}}}}; % the label\n".format(points[0] + normCd / 2, cd)
    )

    text_script += "\\draw[{}] ({},0.5) -- ({},0.5);\n".format(
        lineFormat, points[0], points[0] + normCd
    )

    text_script += (
        "\\foreach \\x in {{{}, {}}} \\"
        "draw[thick,color = black] (\\x, 0.4)"
        " -- (\\x, 0.6);\n \n".format(points[0], points[0] + normCd)
    )

    text_script += "\\draw[gray, thick]({},0) -- ({},0);\n".format(
        points[0], points[lastRank]
    )

    point_names = ",".join(map(str, points))
    text_script += (
        "\\foreach \\x in {{{}}} "
        "\\draw (\\x cm,1.5pt) -- (\\x cm, -1.5pt);\n".format(point_names)
    )

    for idx, p in enumerate(points, 1):
        text_script += "\\node (Label) at ({},0.2){{\\tiny{{{}}}}};\n".format(p, idx)

    startY = -0.25
    deltaY = -0.15
    x1 = 0

    for idx in range(len(segments)):
        seg = segments[idx]
        x1 = ranks_normalized[seg[0]] - 0.05
        x2 = ranks_normalized[seg[1]] + 0.05
        y = startY + deltaY * idx
        text_script += "\\draw[{}]({},{}) -- ({},{});\n".format(
            lineFormat, x1, y, x2, y
        )

    base = 0.25 + 0.2 * len(segments)
    x1 = 0.3

    for idx in range(int(len(names) / 2)):
        text_script += "\\node (Point) at ({}, 0){{}};" "\\node (Label) at ({},-{})".format(
            ranks_normalized[idx], leftLabelPosition, idx * x1 + base
        )
        text_script += "{{\\scriptsize{{{}}}}}; \\draw (Point) |- (Label);\n".format(
            names_sorted[idx]
        )

    for idx in range(len(names) - 1, int((len(names) / 2)) - 1, -1):
        text_script += "\\node (Point) at ({}, 0){{}};" "\\node (Label) at ({},-{})".format(
            ranks_normalized[idx], rightLabelPosition, (len(names) - (idx + 1)) * x1 + base
        )
        text_script += "{{\\scriptsize{{{}}}}}; \\draw (Point) |- (Label);\n".format(
            names_sorted[idx]
        )

    text_script += "\\end{tikzpicture}\n"
    text_script += f"\\caption{{{caption}}}\n"
    text_script += "\\label{fig:nemenyi}\n"
    text_script += "\\end{figure}\n"

    with open(output_file, "w") as f:
        f.write(text_script)
