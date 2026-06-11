"""
Wrapper: scrape each extra league as a separate subprocess.
Ensures Chrome driver fully restarts between leagues.
"""
import subprocess
import sys
import time

LEAGUES = [
    # Skip Eredivisie - already done
    "TUR-Super Lig",
    "POR-Primeira Liga",
    "BEL-First Division A",
    "ENG-Championship",
    "SCO-Premiership",
    "USA-MLS",
    "ITA-Serie B",
    "FRA-Ligue 2",
    "DEN-Superliga",
    "NOR-Eliteserien",
    "SUI-Super League",
    "BRA-Serie A",
    "ARG-Primera Division",
    "MEX-Liga MX",
    "JPN-J1 League",
    "KOR-K League 1",
    "CZE-First League",
    "GRE-Super League",
    "SAU-Pro League",
    "GER-2. Bundesliga",
]

results = {}
for league in LEAGUES:
    print(f"\n{'#'*60}")
    print(f"# Starting: {league}")
    print(f"{'#'*60}")
    try:
        proc = subprocess.run(
            [sys.executable, "scripts/scrape_extra_leagues.py", "--league", league],
            timeout=180,
            capture_output=False,
        )
        results[league] = "OK" if proc.returncode == 0 else f"EXIT {proc.returncode}"
    except subprocess.TimeoutExpired:
        results[league] = "TIMEOUT"
        print(f"  TIMEOUT after 180s, skipping")
    except Exception as e:
        results[league] = f"ERROR: {e}"
        print(f"  ERROR: {e}")

    time.sleep(3)

print(f"\n{'='*60}")
print("SUMMARY:")
for league, status in results.items():
    print(f"  {league:30s} {status}")
