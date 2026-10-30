from app.text_cleanup import repair_pdf_encoding


def test_russian_pdf_encoding_preserves_math():
    original = "Элементарные преобразования системы линейных уравнений сохраняют множество решений"
    broken = original.encode("cp1251").decode("latin1") + "\nx₁ + λx₂ = 0"
    assert repair_pdf_encoding(broken) == original + "\nx₁ + λx₂ = 0"
    assert repair_pdf_encoding(original) == original


def test_normal_latin_text_is_not_reinterpreted():
    original = "Les élèves étudient les équations et les propriétés des systèmes. " * 5
    assert repair_pdf_encoding(original) == original
