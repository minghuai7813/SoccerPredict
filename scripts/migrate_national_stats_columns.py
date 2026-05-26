"""
Add expanded columns to player_stats_national (SQLite ALTER).
为 player_stats_national 添加扩展列（SQLite ALTER）。
"""

import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "oracle_mvp.db"

NEW_COLUMNS = [
    ("shots", "INTEGER"),
    ("shots_on_target", "INTEGER"),
    ("key_passes", "INTEGER"),
    ("through_balls", "INTEGER"),
    ("crosses", "INTEGER"),
    ("progressive_passes", "INTEGER"),
    ("dribble_attempts", "INTEGER"),
    ("dribble_success", "INTEGER"),
    ("carries_count", "INTEGER"),
    ("progressive_carry_distance", "REAL"),
    ("blocks", "INTEGER"),
    ("clearances", "INTEGER"),
    ("aerial_duels_won", "INTEGER"),
    ("aerial_duels_lost", "INTEGER"),
    ("ball_recoveries", "INTEGER"),
    ("fouls_committed", "INTEGER"),
    ("fouls_won", "INTEGER"),
    ("tackle_attempts", "INTEGER"),
    ("pressures", "INTEGER"),
    ("counter_pressures", "INTEGER"),
    ("actions_under_pressure", "INTEGER"),
    ("yellow_cards", "INTEGER"),
    ("red_cards", "INTEGER"),
    ("goalkeeper_saves", "INTEGER"),
    ("is_starter", "INTEGER"),
]


def main() -> None:
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(player_stats_national)")
    existing = {row[1] for row in cur.fetchall()}
    for col, typ in NEW_COLUMNS:
        if col in existing:
            print(f"  skip {col}")
            continue
        cur.execute(f"ALTER TABLE player_stats_national ADD COLUMN {col} {typ}")
        print(f"  added {col}")
    conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
