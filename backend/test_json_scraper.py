import os
import sys
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
import unittest
from pathlib import Path

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

# Add backend directory to sys.path
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

try:
    from backend.scraper import (
        BettingSiteRegistry,
        JSONSiteDataCleaner,
        JSONLinkScraper,
        TeamForm,
        build_model,
    )
    from backend.intelligence import build_ensemble
except ImportError:
    from scraper import (
        BettingSiteRegistry,
        JSONSiteDataCleaner,
        JSONLinkScraper,
        TeamForm,
        build_model,
    )
    from intelligence import build_ensemble


class TestJSONSiteScraper(unittest.TestCase):

    def test_01_registry_load(self):
        """Test loading betting sites from football_betting_sites.json."""
        registry = BettingSiteRegistry()
        sites = registry.get_all_sites()
        self.assertGreater(len(sites), 0, "Betting sites registry should load non-empty list of sites.")
        first_site = sites[0]
        self.assertIn("name", first_site)
        self.assertIn("url", first_site)

    def test_02_data_cleaner_odds(self):
        """Test cleaning and normalizing raw odds values (decimal, American, fractional)."""
        cleaner = JSONSiteDataCleaner()
        self.assertEqual(cleaner.clean_odds_val("2.50"), 2.50)
        self.assertEqual(cleaner.clean_odds_val("+150"), 2.50)
        self.assertEqual(cleaner.clean_odds_val("-200"), 1.50)
        self.assertEqual(cleaner.clean_odds_val("3/2"), 2.50)
        self.assertIsNone(cleaner.clean_odds_val("invalid"))
        self.assertIsNone(cleaner.clean_odds_val("0.5"))

    def test_03_data_cleaner_results_matrix(self):
        """Test aggregation, overround computation, and line shopping extraction."""
        raw_results = [
            {
                "name": "Site A",
                "url": "https://sitea.com",
                "status": "success",
                "status_code": 200,
                "latency_ms": 120.0,
                "extracted_odds": {"home": 2.10, "draw": 3.40, "away": 3.60},
            },
            {
                "name": "Site B",
                "url": "https://siteb.com",
                "status": "success",
                "status_code": 200,
                "latency_ms": 150.0,
                "extracted_odds": {"home": 2.25, "draw": 3.30, "away": 3.50},
            },
        ]
        cleaned = JSONSiteDataCleaner.clean_site_scrape_results(raw_results)
        self.assertEqual(cleaned["total_sites"], 2)
        self.assertEqual(cleaned["successful_sites"], 2)
        self.assertEqual(cleaned["sites_with_odds"], 2)

        # Line shopping best odds checks
        ls = cleaned["line_shopping"]
        self.assertEqual(ls["best_home_odds"]["odd"], 2.25)
        self.assertEqual(ls["best_home_odds"]["site"], "Site B")
        self.assertEqual(ls["best_draw_odds"]["odd"], 3.40)
        self.assertEqual(ls["best_draw_odds"]["site"], "Site A")

        # Consensus probabilities checks
        cp = cleaned["consensus_market_prob"]
        self.assertIsNotNone(cp["home"])
        self.assertIsNotNone(cp["draw"])
        self.assertIsNotNone(cp["away"])
        self.assertAlmostEqual(cp["home"] + cp["draw"] + cp["away"], 1.0, delta=0.05)

    def test_04_ensemble_integration(self):
        """Test blending JSON site consensus market prior into EnsembleEngine."""
        home_form = TeamForm("Arsenal", 5, [2, 3, 1, 2, 1], [0, 1, 0, 1, 1])
        away_form = TeamForm("Chelsea", 5, [1, 2, 0, 1, 2], [2, 1, 1, 0, 2])
        model = build_model(home_form, away_form, 1.45, 1.15)

        consensus_odds = {"home": 0.45, "draw": 0.28, "away": 0.27}
        ensemble = build_ensemble(
            home_form, away_form,
            model.expected_home_goals, model.expected_away_goals,
            1.45, 1.15,
            shrinkage_model=model,
            consensus_odds=consensus_odds,
        )
        self.assertGreater(len(ensemble.component_estimates), 1)
        names = [c.name for c in ensemble.component_estimates]
        self.assertIn("JSON Site Consensus Market Prior", names)

    def test_05_scraper_execution(self):
        """Test executing JSONLinkScraper with limit."""
        scraper = JSONLinkScraper()
        report = scraper.scrape_all_sites(limit=3)
        self.assertIn("total_sites", report)
        self.assertIn("successful_sites", report)
        self.assertEqual(report["total_sites"], 3)


if __name__ == "__main__":
    unittest.main()
