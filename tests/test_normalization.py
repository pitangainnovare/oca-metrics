import unittest

from oca_metrics.utils.normalization import (
    extract_year,
    format_output_header_name,
    shorten_openalex_id,
    stz_binary_flag,
    stz_openalex_source_id,
    stz_text,
    stz_doi,
    stz_title,
)


class TestNormalization(unittest.TestCase):

    def test_stz_doi(self):
        self.assertEqual(stz_doi("10.1590/S0101-01012024000100001"), "10.1590/s0101-01012024000100001")
        self.assertEqual(stz_doi("https://doi.org/10.1590/123"), "10.1590/123")
        self.assertEqual(stz_doi("doi:10.1590/123"), "10.1590/123")
        self.assertEqual(stz_doi(None), "")
        self.assertEqual(stz_doi(""), "")

    def test_stz_title(self):
        self.assertEqual(stz_title("Título com Acentuação"), "titulocomacentuacao")
        self.assertEqual(stz_title("Multiple   Spaces"), "multiplespaces")
        self.assertEqual(stz_title(""), "")
        self.assertEqual(stz_title(None), "")

    def test_extract_year(self):
        self.assertEqual(extract_year("2024"), 2024)
        self.assertEqual(extract_year(2023), 2023)
        self.assertEqual(extract_year("invalid"), None)
        self.assertEqual(extract_year("1799"), 1799)
        self.assertEqual(extract_year("2027"), 2027)
        self.assertEqual(extract_year(None), None)

    def test_stz_text(self):
        self.assertEqual(stz_text("  abc  "), "abc")
        self.assertEqual(stz_text(""), "")
        self.assertEqual(stz_text(None), "")

    def test_stz_binary_flag(self):
        self.assertEqual(stz_binary_flag(1), 1)
        self.assertEqual(stz_binary_flag(0), 0)
        self.assertEqual(stz_binary_flag("yes"), 1)
        self.assertEqual(stz_binary_flag("no"), 0)
        self.assertEqual(stz_binary_flag(None), 0)

    def test_stz_openalex_source_id(self):
        self.assertEqual(stz_openalex_source_id("S123"), "https://openalex.org/S123")
        self.assertEqual(stz_openalex_source_id("https://openalex.org/S123"), "https://openalex.org/S123")
        self.assertEqual(stz_openalex_source_id(""), None)
        self.assertEqual(stz_openalex_source_id(None), None)

    def test_format_output_header_name(self):
        self.assertEqual(format_output_header_name("journal_id"), "journal id")
        self.assertEqual(format_output_header_name("top_1pct_share"), "top 1pct share")

    def test_shorten_openalex_id(self):
        self.assertEqual(shorten_openalex_id("https://openalex.org/S123"), "S123")
        self.assertEqual(shorten_openalex_id("S123"), "S123")
        self.assertEqual(shorten_openalex_id(123), 123)
        self.assertEqual(shorten_openalex_id(None), None)


if __name__ == '__main__':
    unittest.main()
