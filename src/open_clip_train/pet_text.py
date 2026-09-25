"""Conservative, local extraction of clinical sections from PET reports."""
import re

# Anatomy subheadings within Findings are deliberately NOT section boundaries.
_PREFIX = r'^[ \t]*(?:\d+[.)][ \t]*)?'
_TARGET = re.compile(
    _PREFIX + r'(FINDINGS|IMPRESSION)[ \t]*(?::[ \t]*|(?=\r?$))',
    re.IGNORECASE | re.MULTILINE,
)
_BOUNDARY = re.compile(
    _PREFIX + r'(?:'
    r'(?:CLINICAL STATEMENT|CLINICAL HISTORY|HISTORY|INDICATION|TECHNIQUE|'
    r'COMPARISON|CORRELATION|RADIOPHARMACEUTICAL|PROCEDURE|EXAM(?:INATION)?|'
    r'MEDICAL RECORD NUMBER|ACC NUMBER|ACCESSION(?: NUMBER)?|PATIENT(?: NAME)?|'
    r'SUMMARY|ADDENDUM)[ \t]*(?::|(?=\r?$))'
    r'|(?:Electronically[ \t]*Signed|Dictated[ \t]*By|Signed[ \t]*By|'
    r'Report[ \t]+(?:Reviewed|Signed)|Reviewed[ \t]*By|'
    r'The[ \t]+final[ \t]+report[ \t]+was[ \t]+communicated)\b'
    r')', re.IGNORECASE | re.MULTILINE,
)

_FINAL_REPORT = re.compile(
    r'^[ \t*#=_:-]*FINAL[ \t]+REPORT[ \t*#=_:-]*$',
    re.IGNORECASE | re.MULTILINE,
)


# Remove only this dated administrative introduction, not the clinical summary.
_INTEGRATED_SUMMARY_INTRO = re.compile(
    r'^[ \t]*Integrated\s+Imaging\s+Summary\s+for\s+above\s+study\s+and\s+'
    r'CT\s+chest\s+abdomen\s+pelvis\s+performed\s+'
    r'\d{1,2}[-/]\d{1,2}[-/]\d{4}\b[ \t]*[.:]?',
    re.IGNORECASE | re.MULTILINE,
)


def normalize_report_whitespace(text):
    """Collapse horizontal whitespace and remove blank lines, preserving nonempty lines."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return '\n'.join(line for line in text.split('\n') if line)


def extract_pet_sections(report):
    """Return selected text and section names; empty text signals no usable section.

    Exclude SUMMARY sections. Preserve FINDINGS and IMPRESSION headings and source order,
    uncertainty, negation, anatomy subheadings, and distinct repeated sections.
    Normalize whitespace and remove duplicate sections after normalization.
    No generated text or full-report fallback.
    """
    text = normalize_report_whitespace(report)
    targets = list(_TARGET.finditer(text))
    boundaries = sorted(
        [m.start() for m in targets]
        + [m.start() for m in _BOUNDARY.finditer(text)]
        + [m.start() for m in _FINAL_REPORT.finditer(text)]
    )
    selected, names, seen = [], [], set()
    for match in targets:
        end = next((position for position in boundaries if position >= match.end()), len(text))
        body = normalize_report_whitespace(
            _INTEGRATED_SUMMARY_INTRO.sub('', text[match.end():end])
        )
        if not body:
            continue
        name = match.group(1).upper()
        key = (name, body)
        if key in seen:
            continue
        seen.add(key)
        selected.append(f'{name}:\n{body}')
        if name not in names:
            names.append(name)
    return '\n'.join(selected), tuple(names)
