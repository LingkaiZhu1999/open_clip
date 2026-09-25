from open_clip_train.pet_text import extract_pet_sections


def test_sections_exclude_history_technique_and_signature_preserve_anatomy():
    report = ('CLINICAL STATEMENT: known history\nTECHNIQUE: injection\n'
              'FINDINGS:\nLUNGS: no focal uptake.\nOTHER FINDINGS: unchanged.\n'
              'IMPRESSION: Possibly inflammatory; malignancy not excluded.\n'
              'Electronically Signed By: private name\nDictated By: another name')
    text, names = extract_pet_sections(report)
    assert names == ('FINDINGS', 'IMPRESSION')
    assert text == ('FINDINGS:\nLUNGS: no focal uptake.\nOTHER FINDINGS: unchanged.\n'
                    'IMPRESSION:\nPossibly inflammatory; malignancy not excluded.')


def test_alternative_headings_and_crlf():
    text, names = extract_pet_sections('3. findings:\r\nlesion\r\nSUMMARY\r\nuncertain\r\nDictatedBy: name')
    assert names == ('FINDINGS',)
    assert text == 'FINDINGS:\nlesion'


def test_duplicate_sections_and_metadata_boundaries():
    report = ('FINDINGS: lesion\nIMPRESSION: suspicious\nACC NUMBER: secret\n'
              'FINDINGS: lesion\nIMPRESSION: suspicious\nDictated By: name')
    text, names = extract_pet_sections(report)
    assert text == 'FINDINGS:\nlesion\nIMPRESSION:\nsuspicious'
    assert names == ('FINDINGS', 'IMPRESSION')


def test_partial_missing_empty_and_incidental_words():
    assert extract_pet_sections('IMPRESSION: No disease.')[1] == ('IMPRESSION',)
    assert extract_pet_sections('FINDINGS: No lesion.')[1] == ('FINDINGS',)
    assert extract_pet_sections('History: see findings in previous report.') == ('', ())
    assert extract_pet_sections('FINDINGS:\nIMPRESSION:\nDictated By: name') == ('', ())


def test_whitespace_cleanup_preserves_lines_negation_and_numbers():
    text, _ = extract_pet_sections(
        '  FINDINGS:  No   focal\tuptake.  SUV\u00a0\u00a0max 3.5.  \r\n'
        '   LUNGS:   unchanged.\n \n\n\n'
        'IMPRESSION:\t1.  No evidence of recurrence.\n  2.   Possibly inflammatory.  '
    )
    assert text == ('FINDINGS:\nNo focal uptake. SUV max 3.5.\nLUNGS: unchanged.\n'
                    'IMPRESSION:\n1. No evidence of recurrence.\n2. Possibly inflammatory.')


def test_final_report_footer_and_communication_removed():
    for footer in ['FINAL REPORT', '   Final    Report   ', '*** FINAL REPORT ***',
                   'The final report was communicated to someone via email.']:
        text, names = extract_pet_sections('IMPRESSION: No disease.\n' + footer + '\nAdministrative text')
        assert text == 'IMPRESSION:\nNo disease.'
        assert names == ('IMPRESSION',)


def test_clinical_reference_to_final_report_not_blindly_deleted():
    text, _ = extract_pet_sections('FINDINGS: Compare with the final report from the prior study.')
    assert 'Compare with the final report from the prior study.' in text


def test_deduplication_after_whitespace_normalization():
    text, names = extract_pet_sections('FINDINGS: No   lesion.\nFINAL REPORT\nFINDINGS: No lesion.')
    assert text == 'FINDINGS:\nNo lesion.'
    assert names == ('FINDINGS',)


def test_empty_final_report_is_not_a_clinical_section():
    assert extract_pet_sections('IMPRESSION:\nFINAL REPORT') == ('', ())


def test_summary_excluded_alongside_impression():
    text, names = extract_pet_sections(
        'IMPRESSION: Mixed response.\n'
        'SUMMARY: Mixed changes as per the impression above.'
    )
    assert names == ('IMPRESSION',)
    assert text == 'IMPRESSION:\nMixed response.'


def test_dated_integrated_intro_removed_without_losing_clinical_summary():
    for intro in [
        'Integrated Imaging Summary for above study and CT chest abdomen pelvis performed 3-14-2018',
        'Integrated Imaging Summary for above study and\nCT chest abdomen pelvis performed 04/15/2019:',
    ]:
        text, names = extract_pet_sections(
            'FINDINGS: ' + intro + '\nNo new lesions.\nIMPRESSION: Stable findings.'
        )
        assert names == ('FINDINGS', 'IMPRESSION')
        assert text == 'FINDINGS:\nNo new lesions.\nIMPRESSION:\nStable findings.'
    assert extract_pet_sections('SUMMARY: ' + intro) == ('', ())


def test_other_integrated_summary_prose_preserved():
    report = 'IMPRESSION: Integrated imaging summary shows no new lesions.'
    assert extract_pet_sections(report)[0] == 'IMPRESSION:\nIntegrated imaging summary shows no new lesions.'


def test_summary_is_boundary_and_later_target_sections_are_retained():
    text, names = extract_pet_sections(
        'FINDINGS: No new lesions.\n2. Summary\nExcluded content.\n'
        'IMPRESSION: Stable findings.\nSUMMARY: Also excluded.'
    )
    assert text == 'FINDINGS:\nNo new lesions.\nIMPRESSION:\nStable findings.'
    assert names == ('FINDINGS', 'IMPRESSION')
    assert extract_pet_sections('SUMMARY: Summary only.') == ('', ())
