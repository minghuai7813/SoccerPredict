"""Analyze which leagues the synthetic-stats players belong to."""
import json, sys, re
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.encoding import fix_console_encoding
fix_console_encoding()

rosters = json.loads(
    Path("d:/CursorProjects/SoccerProject/data/wc2026_rosters.json")
    .read_text(encoding="utf-8")
)

FBREF_LEAGUES = {
    # Tier 2 - FBref covered
    "Eredivisie": ["psv", "ajax", "feyenoord", "az ", "nec", "twente", "utrecht",
        "heracles", "pec zwolle", "rkc", "volendam", "sparta rotterdam",
        "sc telstar", "vvv", "den bosch", "almere city"],
    "Turkish Super Lig": ["galatasaray", "fenerbahce", "besiktas", "trabzonspor",
        "konyaspor", "rizespor", "basaksehir", "kayserispor", "caykur rizespor",
        "alanyaspor"],
    "Liga Portugal": ["benfica", "sporting", "porto", "braga", "vitoria guimaraes",
        "farense", "chaves", "torreense", "estrela", "casa pia", "gil vicente",
        "tondela", "vizela"],
    "Belgian JPL": ["club brugge", "anderlecht", "gent", "genk", "standard liege",
        "charleroi", "union saint", "sint-truiden", "beveren", "zulte waregem",
        "mechelen", "dender"],
    "Championship": ["sheffield united", "burnley", "leeds", "hull", "stoke",
        "swansea", "derby", "middlesbrough", "millwall", "coventry", "ipswich",
        "peterborough", "wrexham", "rotherham", "barnsley", "norwich",
        "charlton", "watford", "portsmouth"],
    "Scottish Prem": ["celtic", "rangers", "hearts", "hibernian", "motherwell",
        "ross county"],
    "Bundesliga 2": ["hamburg", "st pauli", "hannover", "karlsruher",
        "fortuna dusseldorf", "holstein kiel"],
    "MLS": ["inter miami", "lafc", "columbus crew", "nashville", "chicago fire",
        "philadelphia union", "new york city", "seattle", "minnesota",
        "portland timbers", "colorado rapids", "atlanta united",
        "toronto", "vancouver whitecaps", "orlando city", "dallas",
        "fc cincinnati", "austin fc", "charlotte fc", "new england",
        "san diego", "real salt lake"],
    "Serie B": ["sassuolo", "pisa", "cremonese", "frosinone", "sampdoria"],
    "Ligue 2": ["bastia", "sochaux", "nancy", "montpellier"],
    "Danish Superliga": ["copenhagen", "brondby", "midtjylland", "nordsjaelland",
        "silkeborg", "aarhus"],
    "Norwegian Eliteserien": ["viking", "bodo/glimt", "molde", "sarpsborg"],
    "Swiss Super League": ["fc zurich", "young boys", "servette", "lugano",
        "st gallen"],
    "Brazilian Serie A": ["flamengo", "palmeiras", "sao paulo", "gremio",
        "atletico mineiro", "botafogo", "internacional", "bragantino",
        "fluminense", "corinthians", "vasco da gama", "santos"],
    "Argentine Primera": ["river plate", "boca juniors", "independiente", "racing",
        "san lorenzo", "lanus", "talleres"],
    "Liga MX": ["club america", "unam", "pumas", "chivas", "cruz azul", "toluca",
        "atlas", "tijuana", "pachuca", "mazatlan", "leon", "santos laguna"],
    "J-League": ["kashima", "fc tokyo", "sanfrecce hiroshima", "albirex niigata",
        "machida zelvia", "yokohama"],
    "K-League": ["ulsan", "jeonbuk", "daejeon", "gangwon", "fc seoul"],
    "Saudi Pro": ["al hilal", "al nassr", "al ahli", "al ittihad", "al qadsiah",
        "al ettifaq", "al fayha", "al ula"],
    "Czech First League": ["slavia prague", "sparta prague", "viktoria plzen",
        "hradec kralove"],
    "Greek Super League": ["olympiacos", "panathinaikos", "aris", "kifisia",
        "atromitos", "larisa"],
    "Eredivisie 2": ["sc telstar", "vvv-venlo", "den bosch"],
}

def classify_league(club):
    cl = club.lower().strip()
    for league, patterns in FBREF_LEAGUES.items():
        for p in patterns:
            if p in cl:
                return league
    return "Unknown/Small"

# Analyze
league_counts = Counter()
league_players = {}
total_synthetic = 0

for team, players in rosters.items():
    for p in players:
        if p.get("match_status") == "matched":
            continue
        total_synthetic += 1
        club = p.get("club", "")
        league = classify_league(club)
        league_counts[league] += 1
        if league not in league_players:
            league_players[league] = []
        league_players[league].append(f"{p['name']} ({club}) [{team}]")

print(f"Total synthetic-data players: {total_synthetic}\n")
print(f"{'League':<30s} {'Count':>5s}  {'FBref?':>6s}")
print("-" * 50)
fbref_total = 0
for league, count in league_counts.most_common():
    is_fbref = league != "Unknown/Small"
    if is_fbref:
        fbref_total += count
    print(f"{league:<30s} {count:5d}  {'YES' if is_fbref else 'NO':>6s}")

print(f"\nFBref-scrapable: {fbref_total}/{total_synthetic} ({fbref_total/total_synthetic*100:.0f}%)")
print(f"Truly unscrapable: {total_synthetic - fbref_total}")

print(f"\n--- Unknown/Small league players ---")
for info in league_players.get("Unknown/Small", []):
    print(f"  {info}")
