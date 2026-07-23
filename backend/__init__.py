"""
PredictBet — Elite Quantitative Football Betting Intelligence System
===============================================================

Package init. Importing this package exposes the key public symbols from
every module so the rest of the codebase can depend on a single namespace
instead of reaching into individual files.

Each submodule import is wrapped in try/except so that missing optional
dependencies do not prevent the rest of the package from loading.
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# scraper
# ---------------------------------------------------------------------------
try:
    from scraper import (
        TeamForm,
        MatchModelResult,
        ScraperCache,
        BetikaClient,
        ESPNScraperClient,
        FootballDataClient,
        EloRatingScraper,
        WikipediaTeamScraper,
        build_model,
        compare_to_market,
        devig_1x2,
        poisson_pmf,
        weighted_shrunk_rate,
        dixon_coles_tau,
        _cache,
        fetch_team_stock_data,
        HeadToHeadFetcher,
        FormMomentumCalculator,
        MultiMarketPredictor,
    )
    _scraper_available = True
except Exception as e:
    logger.warning("scraper module could not be fully imported: %s", e)
    _scraper_available = False

# ---------------------------------------------------------------------------
# intelligence
# ---------------------------------------------------------------------------
try:
    from intelligence import (
        EnsembleResult,
        EstimatorOutput,
        PredictionLedger,
        build_ensemble,
        generate_scouting_narrative,
        suggest_stake,
        StakeSuggestion,
        RISK_NOTE,
        ConfidenceTier,
        AggressiveStakeEngine,
        PredictionCard,
        AutoPredictionPipeline,
        generate_aggressive_narrative,
    )
    _intelligence_available = True
except Exception as e:
    logger.warning("intelligence module could not be fully imported: %s", e)
    _intelligence_available = False

# ---------------------------------------------------------------------------
# market_pipeline
# ---------------------------------------------------------------------------
try:
    from market_pipeline import (
        XGDataClient,
        TeamNewsClient,
        MultiBookmakerOddsAggregator,
        BookmakerQuote,
        BestPrice,
        LeagueParameterCalibrator,
        TeamHomeAdvantageEstimator,
        ResultsSyncer,
        VersionedPredictionLedger,
        OddsMovementTracker,
        blend_with_market_probability,
    )
    _market_pipeline_available = True
except Exception as e:
    logger.warning("market_pipeline module could not be fully imported: %s", e)
    _market_pipeline_available = False

# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------
try:
    from backtest import (
        SettledBet,
        load_bets_from_csv,
        load_bets_from_json,
        bets_with_devigged_market,
        hit_rate,
        brier_and_log_loss,
        calibration_by_bucket,
        flat_stake_roi,
        capped_kelly_bankroll_simulation,
        significance_vs_market,
        closing_line_value,
        baseline_comparisons,
        full_accuracy_report,
        print_report,
    )
    _backtest_available = True
except Exception as e:
    logger.warning("backtest module could not be fully imported: %s", e)
    _backtest_available = False

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------
__all__ = []

if _scraper_available:
    __all__ += [
        "TeamForm",
        "MatchModelResult",
        "ScraperCache",
        "BetikaClient",
        "ESPNScraperClient",
        "FootballDataClient",
        "EloRatingScraper",
        "WikipediaTeamScraper",
        "build_model",
        "compare_to_market",
        "devig_1x2",
        "poisson_pmf",
        "weighted_shrunk_rate",
        "dixon_coles_tau",
        "_cache",
        "fetch_team_stock_data",
        "HeadToHeadFetcher",
        "FormMomentumCalculator",
        "MultiMarketPredictor",
    ]

if _intelligence_available:
    __all__ += [
        "EnsembleResult",
        "EstimatorOutput",
        "PredictionLedger",
        "build_ensemble",
        "generate_scouting_narrative",
        "suggest_stake",
        "StakeSuggestion",
        "RISK_NOTE",
        "ConfidenceTier",
        "AggressiveStakeEngine",
        "PredictionCard",
        "AutoPredictionPipeline",
        "generate_aggressive_narrative",
    ]

# ---------------------------------------------------------------------------
# pipeline (automated workflow)
# ---------------------------------------------------------------------------
try:
    from pipeline import (
        AutomatedPredictionPipeline,
        FixtureIngestor,
        FixtureEnricher,
        MarketEvaluator,
        BetRefusalEngine,
        InstitutionalReportGenerator,
        FixtureEnrichment,
        PipelineResult,
    )
    _pipeline_available = True
    __all__ += [
        "AutomatedPredictionPipeline",
        "FixtureIngestor",
        "FixtureEnricher",
        "MarketEvaluator",
        "BetRefusalEngine",
        "InstitutionalReportGenerator",
        "FixtureEnrichment",
        "PipelineResult",
    ]
except Exception as e:
    logger.warning("pipeline module could not be imported: %s", e)
    _pipeline_available = False

if _market_pipeline_available:
    __all__ += [
        "XGDataClient",
        "TeamNewsClient",
        "MultiBookmakerOddsAggregator",
        "BookmakerQuote",
        "BestPrice",
        "LeagueParameterCalibrator",
        "TeamHomeAdvantageEstimator",
        "ResultsSyncer",
        "VersionedPredictionLedger",
        "OddsMovementTracker",
        "blend_with_market_probability",
    ]

if _backtest_available:
    __all__ += [
        "SettledBet",
        "load_bets_from_csv",
        "load_bets_from_json",
        "bets_with_devigged_market",
        "hit_rate",
        "brier_and_log_loss",
        "calibration_by_bucket",
        "flat_stake_roi",
        "capped_kelly_bankroll_simulation",
        "significance_vs_market",
        "closing_line_value",
        "baseline_comparisons",
        "full_accuracy_report",
        "print_report",
    ]
