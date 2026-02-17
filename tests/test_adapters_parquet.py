import unittest
import pandas as pd
import duckdb
import json

from oca_metrics.adapters.parquet import ParquetAdapter


class TestParquetAdapter(unittest.TestCase):

    def setUp(self):
        # Create a temporary parquet file for testing
        self.parquet_path = "test_metrics.parquet"
        self.con = duckdb.connect(database=':memory:')
        
        # Create dummy data
        data = {
            'publication_year': [2024, 2024, 2024, 2023, 2018],
            'source_id': ['S1', 'S2', 'S1', 'S3', 'S4'],
            'source_issn_l': ['1234-5678', '8765-4321', '1234-5678', '1111-2222', '9999-0000'],
            'language': ['en', 'en', 'pt', 'en', 'en'],
            'is_merged': [1, 0, 0, 0, 0],
            'oa_individual_works': [
                json.dumps({
                    "W1": {"language": "en"},
                    "W2": {"language": "pt"},
                    "W3": {"language": "es"},
                }),
                None,
                None,
                None,
                None,
            ],
            'field': ['Medicine', 'Medicine', 'Medicine', 'Physics', "China's Socioeconomic Reforms and Governance"],
            'citations_total': [10, 5, 20, 15, 7],
            'citations_window_2y': [2, 1, 4, 3, 1],
            'citations_window_3y': [3, 2, 5, 4, 2],
            'citations_window_5y': [5, 3, 8, 6, 3],
            'citations_2024': [1, 0, 3, 2, 0],
            'citations_2025': [2, 1, 5, 0, 0],
        }
        df = pd.DataFrame(data)
        df.to_parquet(self.parquet_path)
        
        self.adapter = ParquetAdapter(self.parquet_path)

    def tearDown(self):
        import os
        if os.path.exists(self.parquet_path):
            os.remove(self.parquet_path)

    def test_get_categories(self):
        categories = self.adapter.get_categories(2024, 'field')
        self.assertIn('Medicine', categories)
        self.assertNotIn('Physics', categories) # Different year

    def test_get_categories_with_apostrophe_filter(self):
        cat = "China's Socioeconomic Reforms and Governance"
        categories = self.adapter.get_categories(2018, 'field', cat)
        self.assertEqual(categories, [cat])

    def test_get_yearly_citation_columns(self):
        citation_cols = self.adapter.get_yearly_citation_columns()
        self.assertEqual(citation_cols, ['citations_2024', 'citations_2025'])

    def test_compute_baseline(self):
        baseline = self.adapter.compute_baseline(2024, 'field', 'Medicine', [2, 3])
        self.assertIsNotNone(baseline)
        self.assertEqual(baseline['total_docs'], 3)
        self.assertEqual(baseline['total_citations'], 35) # 10 + 5 + 20
        self.assertAlmostEqual(baseline['mean_citations'], 35/3)
        self.assertEqual(baseline['total_citations_window_2y'], 7) # 2 + 1 + 4

    def test_compute_baseline_with_apostrophe_category(self):
        cat = "China's Socioeconomic Reforms and Governance"
        baseline = self.adapter.compute_baseline(2018, 'field', cat, [2, 3, 5])
        self.assertIsNotNone(baseline)
        self.assertEqual(baseline['total_docs'], 1)
        self.assertEqual(baseline['total_citations'], 7)

    def test_compute_thresholds(self):
        # Percentiles: 50
        thresholds = self.adapter.compute_thresholds(2024, 'field', 'Medicine', [2], [50])
        # Citations total: 5, 10, 20. Median (50%) with discrete quantile is 10.
        self.assertEqual(thresholds['C_top50pct'], 10)
        
        # Window 2y: 1, 2, 4. Median is 2.
        self.assertEqual(thresholds['C_top50pct_window_2y'], 2)

    def test_compute_journal_metrics(self):
        thresholds = {
            'C_top50pct': 10,
            'C_top50pct_window_2y': 2
        }
        df_journals = self.adapter.compute_journal_metrics(2024, 'field', 'Medicine', [2], thresholds)
        
        self.assertFalse(df_journals.empty)
        self.assertEqual(len(df_journals), 2) # S1 and S2
        
        # Check S1
        s1 = df_journals[df_journals['journal_id'] == 'S1'].iloc[0]
        self.assertEqual(s1['journal_publications_count'], 2)
        self.assertEqual(s1['journal_citations_total'], 30) # 10 + 20
        
        # Check top counts
        # S1 citations: 10, 20. Threshold 10. Both >= 10. Count = 2.
        self.assertEqual(s1['top_50pct_all_time_publications_count'], 2)
        
        # S1 window 2y: 2, 4. Threshold 2. Both >= 2. Count = 2.
        self.assertEqual(s1['top_50pct_window_2y_publications_count'], 2)
        self.assertEqual(s1['is_journal_multilingual'], 1)
        self.assertEqual(s1['citations_2024'], 4)  # 1 + 3
        self.assertEqual(s1['citations_2025'], 7)  # 2 + 5

        s2 = df_journals[df_journals['journal_id'] == 'S2'].iloc[0]
        self.assertEqual(s2['is_journal_multilingual'], 0)


if __name__ == '__main__':
    unittest.main()
