# AizoAutoGrading Testu Atskaite

## Kopsavilkums
- Testu fails: `test_autograd.py`
- Vienību testi: 12
- Integrācijas testi: 4
- Kopā: 16
- Rezultāts: 16 izpildīti veiksmīgi

## Kas Tika Testēts
- `dataNormalize()`
- `GradeTest()`
- `extract_document()`
- `get_topic_boxes()`
- Flask `/imgUpload` GET
- Flask `/imgUpload` POST

## Palaišanas Komanda
```bash
/opt/homebrew/bin/python3 -m pytest test_autograd.py -v
```

## Pēdējais Rezultāts
```text
16 passed
```

## Secinājums
Visi pieprasītie testi ir uzrakstīti failā `test_autograd.py`, un tie iziet veiksmīgi.
