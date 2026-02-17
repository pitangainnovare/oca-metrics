from unittest.mock import MagicMock

import pandas as pd
import unittest

from oca_metrics.adapters.base import BaseAdapter
from oca_metrics.core import MetricsEngine


class TestMetricsEngine(unittest.TestCase):

    def setUp(self):
        self.mock_adapter = MagicMock(spec=BaseAdapter)
        self.engine = MetricsEngine(self.mock_adapter)
        self.year = 2024
        self.level = "field"
        self.cat_id = "Medicine"
        self.windows = [2, 3]
        self.target_percentiles = [99, 50]
        self.engine.target_percentiles = self.target_percentiles

    def test_process_category_success(self):
        # Mock baseline
        baseline_data = {
            'total_docs': 100,
            'total_citations': 500,
            'mean_citations': 5.0,
            'total_citations_window_2y': 200,
            'mean_citations_window_2y': 2.0,
            'total_citations_window_3y': 300,
            'mean_citations_window_3y': 3.0
        }
        self.mock_adapter.compute_baseline.return_value = pd.Series(baseline_data)

        # Mock thresholds
        thresholds_data = {
            'C_top1pct': 50,
            'C_top1pct_window_2y': 20,
            'C_top1pct_window_3y': 30,
            'C_top50pct': 5,
            'C_top50pct_window_2y': 2,
            'C_top50pct_window_3y': 3
        }
        self.mock_adapter.compute_thresholds.return_value = thresholds_data

        # Mock journal metrics
        journal_data = {
            'journal_id': ['J1', 'J2'],
            'journal_publications_count': [10, 20],
            'journal_citations_total': [100, 50],
            'journal_citations_mean': [10.0, 2.5],
            'journal_citations_mean_window_2y': [4.0, 1.0],
            'journal_citations_mean_window_3y': [6.0, 1.5],
            'top_1pct_all_time_publications_count': [2, 0],
            'top_1pct_window_2y_publications_count': [1, 0],
            'top_1pct_window_3y_publications_count': [1, 0],
            'top_50pct_all_time_publications_count': [8, 5],
            'top_50pct_window_2y_publications_count': [6, 2],
            'top_50pct_window_3y_publications_count': [7, 3]
        }
        self.mock_adapter.compute_journal_metrics.return_value = pd.DataFrame(journal_data)

        # Execute
        df_result = self.engine.process_category(self.year, self.level, self.cat_id, self.windows)

        # Assertions
        self.assertIsNotNone(df_result)
        self.assertEqual(len(df_result), 2)
        self.assertEqual(df_result.iloc[0]['category_id'], self.cat_id)
        self.assertEqual(df_result.iloc[0]['category_level'], self.level)
        self.assertEqual(df_result.iloc[0]['publication_year'], self.year)
        
        # Check normalized impact
        self.assertEqual(df_result.iloc[0]['journal_impact_normalized'], 10.0 / 5.0) # 2.0
        self.assertEqual(df_result.iloc[1]['journal_impact_normalized'], 2.5 / 5.0) # 0.5

        # Check percentile shares
        # J1 top 1% share: 2/10 = 20%
        self.assertEqual(df_result.iloc[0]['top_1pct_all_time_publications_share_pct'], 20.0)
        
        # Check metadata integration (default empty)
        self.assertEqual(df_result.iloc[0]['journal_title'], 'J1')
        self.assertEqual(df_result.iloc[0]['is_scielo'], 0)
        self.assertEqual(df_result.iloc[0]['country'], '')
        self.assertEqual(df_result.iloc[0]['collection'], '')

    def test_process_category_with_metadata(self):
        # Setup mocks similar to success case
        baseline_data = {
            'total_docs': 100,
            'total_citations': 500,
            'mean_citations': 5.0,
            'total_citations_window_2y': 200,
            'mean_citations_window_2y': 2.0,
            'total_citations_window_3y': 300,
            'mean_citations_window_3y': 3.0
        }
        self.mock_adapter.compute_baseline.return_value = pd.Series(baseline_data)
        
        thresholds_data = {
            'C_top1pct': 50,
            'C_top1pct_window_2y': 20,
            'C_top1pct_window_3y': 30,
            'C_top50pct': 5,
            'C_top50pct_window_2y': 2,
            'C_top50pct_window_3y': 3
        }
        self.mock_adapter.compute_thresholds.return_value = thresholds_data
        
        journal_df = pd.DataFrame({
            'journal_id': ['https://openalex.org/S1', 'https://openalex.org/S2'],
            'journal_publications_count': [10, 20],
            'journal_citations_mean': [10.0, 2.5],
            'top_1pct_all_time_publications_count': [1, 0],
            'top_50pct_all_time_publications_count': [5, 2]
        })

        # Add missing columns required by the loop in process_category
        for w in self.windows:
             journal_df[f'journal_citations_mean_window_{w}y'] = 0
             journal_df[f'top_1pct_window_{w}y_publications_count'] = 0
             journal_df[f'top_50pct_window_{w}y_publications_count'] = 0
             
        self.mock_adapter.compute_journal_metrics.return_value = journal_df

        # Metadata
        df_meta = pd.DataFrame({
            'source_id': ['https://openalex.org/S1'],
            'publication_year': [2024],
            'journal_title': ['Journal One'],
            'is_scielo': [1],
            'country': ['Brazil'],
            'scielo_active_valid': [1],
            'scielo_collection_acronym': ['scl'],
        })

        # Execute
        df_result = self.engine.process_category(self.year, self.level, self.cat_id, self.windows, df_meta)

        # Assertions
        self.assertIsNotNone(df_result)
        row1 = df_result[df_result['journal_id'] == 'https://openalex.org/S1'].iloc[0]
        self.assertEqual(row1['journal_title'], 'Journal One')
        self.assertEqual(row1['is_scielo'], 1)
        self.assertEqual(row1['country'], 'Brazil')
        self.assertEqual(row1['collection'], 'scl')
        
        row2 = df_result[df_result['journal_id'] == 'https://openalex.org/S2'].iloc[0]
        self.assertEqual(row2['journal_title'], 'https://openalex.org/S2')
        self.assertEqual(row2['is_scielo'], 0)
        self.assertEqual(row2['country'], '')
        self.assertEqual(row2['collection'], '')

    def test_process_category_no_baseline(self):
        self.mock_adapter.compute_baseline.return_value = None
        df_result = self.engine.process_category(self.year, self.level, self.cat_id, self.windows)
        self.assertIsNone(df_result)

    def test_process_category_no_thresholds(self):
        self.mock_adapter.compute_baseline.return_value = pd.Series({'total_docs': 100})
        self.mock_adapter.compute_thresholds.return_value = {}
        df_result = self.engine.process_category(self.year, self.level, self.cat_id, self.windows)
        self.assertIsNone(df_result)

    def test_process_category_empty_journals(self):
        self.mock_adapter.compute_baseline.return_value = pd.Series({'total_docs': 100})
        self.mock_adapter.compute_thresholds.return_value = {'C_top1pct': 10}
        self.mock_adapter.compute_journal_metrics.return_value = pd.DataFrame()
        df_result = self.engine.process_category(self.year, self.level, self.cat_id, self.windows)
        self.assertIsNone(df_result)


if __name__ == '__main__':
    unittest.main()
