import argparse
import sys
import logging
import datetime
import pandas as pd

from oca_metrics.adapters.parquet import ParquetAdapter
from oca_metrics.core import MetricsEngine
from oca_metrics.utils.csv_schema import (
    get_csv_schema_order,
)
from oca_metrics.utils.metadata import (
    load_global_metadata,
)
from oca_metrics.utils.normalization import (
    format_output_header_name,
    shorten_openalex_id,
)


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute journal-level bibliometrics from data source."
    )
    parser.add_argument("--parquet", help="Path to metrics parquet file.", required=True)
    parser.add_argument("--global-xlsx", help="Path to global metrics excel file.")
    
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=datetime.datetime.now().year)
    
    parser.add_argument("--level", default="field", choices=["domain", "field", "subfield", "topic"])
    parser.add_argument("--category-id", type=str, default=None)
    parser.add_argument("--windows", type=int, nargs="+", default=[2, 3, 5])
    
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--shorten-ids", action="store_true", help="Shorten OpenAlex IDs in output.")
    
    return parser.parse_args()


def main():
    try:
        args = parse_args()
    except SystemExit as e:
        if len(sys.argv) > 1 and ('-h' in sys.argv or '--help' in sys.argv):
            sys.exit(0)
        sys.exit(e.code if e.code != 0 else 2)

    if args.year:
        years = [args.year]
    else:
        years = list(range(args.start_year, args.end_year + 1))
        
    windows = sorted(args.windows)
    level = args.level
    
    try:
        adapter = ParquetAdapter(args.parquet)
        engine = MetricsEngine(adapter)
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        sys.exit(1)
    
    df_meta = load_global_metadata(args.global_xlsx) if args.global_xlsx else pd.DataFrame()

    yearly_citation_cols = adapter.get_yearly_citation_columns()
    schema_keys = get_csv_schema_order(windows, [99, 95, 90, 50], yearly_citation_cols) 
    output_headers = [format_output_header_name(k) for k in schema_keys]
    
    output_file = args.output_file or f"indicators_{level}_{years[0]}-{years[-1]}.csv"
    
    first_item = True
    
    for year in years:
        logger.info(f"Processing year {year}...")
        
        try:
            categories = adapter.get_categories(year, level, args.category_id)
        except Exception as e:
            logger.error(f"Error fetching categories for year {year}: {e}")
            continue
            
        logger.info(f"Found {len(categories)} categories")
        
        for cat_idx, cat_id in enumerate(categories):
            logger.info(f"  [{cat_idx+1}/{len(categories)}] {cat_id}")
            
            df_journals = engine.process_category(year, level, cat_id, windows, df_meta)
            
            if df_journals is None or df_journals.empty:
                continue
            
            if args.shorten_ids:
                if 'journal_id' in df_journals.columns:
                    df_journals['journal_id'] = df_journals['journal_id'].apply(shorten_openalex_id)

                if 'category_id' in df_journals.columns:
                    df_journals['category_id'] = df_journals['category_id'].apply(shorten_openalex_id)

            # Ensure all schema keys exist in the dataframe
            for k in schema_keys:
                if k not in df_journals.columns:
                    df_journals[k] = ""
            
            # Reorder columns according to schema
            df_output = df_journals[schema_keys].copy()
            df_output.columns = output_headers
            
            mode = 'w' if first_item else 'a'
            header = True if first_item else False
            df_output.to_csv(output_file, mode=mode, index=False, header=header, encoding='utf-8')
            first_item = False

    logger.info(f"Done! Results saved to {output_file}")


if __name__ == "__main__":
    main()
